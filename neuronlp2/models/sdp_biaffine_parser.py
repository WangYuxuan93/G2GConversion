# -*- coding:utf-8 -*-

__author__ = 'max'

import os
import json
from overrides import overrides
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronlp2.io import get_logger
from neuronlp2.nn import TreeCRF, VarGRU, VarRNN, VarLSTM, VarFastLSTM
from neuronlp2.nn import BiAffine, BiLinear, CharCNN, BiAffine_v2
from neuronlp2.nn.self_attention import AttentionEncoderConfig, AttentionEncoder
from neuronlp2.nn.graph_attention_network import GraphAttentionNetworkConfig, GraphAttentionNetwork
from neuronlp2.nn.dropout import drop_input_independent
from torch.autograd import Variable
from transformers import *
import itertools
from neuronlp2.nn.cpg_lstm import CPG_LSTM
from neuronlp2.models.semsyn_roberta import SemSynRobertaConfig, SemSynRobertaModel
from neuronlp2.lca_in_dag import LCADAG


class PositionEmbeddingLayer(nn.Module):
    def __init__(self, embedding_size, dropout_prob=0, max_position_embeddings=256):
        super(PositionEmbeddingLayer, self).__init__()
        """Adding position embeddings to input layer
        """
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_tensor, debug=False):
        """
        input_tensor: (batch, seq_len, input_size)
        """
        seq_length = input_tensor.size(1)
        batch_size = input_tensor.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_tensor.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size,-1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_tensor + position_embeddings
        embeddings = self.dropout(embeddings)
        if debug:
            print ("input_tensor:",input_tensor)
            print ("position_embeddings:",position_embeddings)
            print ("embeddings:",embeddings)
        return embeddings

"""
oldLabelsEmbeddding
jeffrey 2021-9-12
"""
# class oldLabelsEmbeddingLayer(nn.Module):
#
# 	def __init__(self, embedding_size, dropout_prob=0,max_labels=1000):
# 		super(oldLabelsEmbeddingLayer, self).__init__()
# 		self.labels_embeddings = nn.Embedding(max_labels, embedding_size)
# 		self.dropout = nn.Dropout(dropout_prob)
#
# 	def forward(self, input_tensor, debug=False):
# 		"""
# 		input_tensor: (batch, seq_len, input_size)
# 		"""
# 		# seq_length = input_tensor.size(1)
# 		# batch_size = input_tensor.size(0)
# 		embeddings = self.labels_embeddings(input_tensor)
# 		embeddings = self.dropout(embeddings)
# 		if debug:
# 			print ("input_tensor:",input_tensor)
# 			print ("labels_embeddings:",embeddings)
# 		return embeddings

def load_elmo(path):
    from allennlp.modules.elmo import Elmo, batch_to_ids
    options_file = os.path.join(path, "options.json")
    weight_file = os.path.join(path, "weights.hdf5")
    if not os.path.exists(options_file):
        print ("Did not find options.json in {}".format(path))
    if not os.path.exists(weight_file):
        print ("Did not find weights.hdf5 in {}".format(path))
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    conf = json.loads(open(options_file, 'r').read())
    output_size = conf['lstm']['projection_dim'] * 2
    return elmo, output_size

class SDPBiaffineParser(nn.Module):
    def __init__(self, hyps, num_pretrained, num_words, num_chars, num_pos, num_labels,
                 device=torch.device('cpu'),
                 embedd_word=None, embedd_char=None, embedd_pos=None,
                 use_pretrained_static=True, use_random_static=False,
                 use_elmo=False, elmo_path=None,
                 pretrained_lm='none', lm_path=None, lm_config=None, num_src_labels=52,
                 num_lans=1, log_name='Network'):
        super(SDPBiaffineParser, self).__init__()
        self.hyps = hyps
        self.device = device

        # for input embeddings
        use_pos = hyps['input']['use_pos']
        use_char = hyps['input']['use_char']
        word_dim = hyps['input']['word_dim']
        pos_dim = hyps['input']['pos_dim']
        char_dim = hyps['input']['char_dim']
        self.pos_dim = pos_dim
        self.use_pretrained_static = use_pretrained_static
        self.use_random_static = use_random_static
        self.use_elmo = use_elmo and elmo_path is not None
        self.pretrained_lm = pretrained_lm
        self.only_pretrain_static = use_pretrained_static and not use_random_static

        # for biaffine layer
        self.arc_mlp_dim = hyps['biaffine']['arc_mlp_dim']
        self.rel_mlp_dim = hyps['biaffine']['rel_mlp_dim']
        p_in = hyps['biaffine']['p_in']
        self.p_in = p_in
        p_out = hyps['biaffine']['p_out']
        activation = hyps['biaffine']['activation']
        self.act_func = activation
        # for input encoder
        input_encoder_name = hyps['input_encoder']['name']
        hidden_size = hyps['input_encoder']['hidden_size']
        num_layers = hyps['input_encoder']['num_layers']
        p_rnn = hyps['input_encoder']['p_rnn']
        self.p_rnn = p_rnn
        self.lan_emb_as_input = False
        lan_emb_size = hyps['input_encoder']['lan_emb_size']
        #self.end_word_id = end_word_id

        logger = get_logger(log_name)
        self.logger = logger
        model = "{}-{}".format(hyps['model'], input_encoder_name)
        logger.info("Network: %s, hidden=%d, act=%s" % (model, hidden_size, activation))
        logger.info("##### Embeddings (POS tag: %s, Char: %s) #####" % (use_pos, use_char))
        logger.info("dropout(in, out): (%.2f, %.2f)" % (p_in, p_out))
        logger.info("Use Randomly Init Word Emb: %s" % (use_random_static))
        logger.info("Use Pretrained Word Emb: %s" % (use_pretrained_static))
        logger.info("##### Input Encoder (Type: %s, Layer: %d, Hidden: %d) #####" % (input_encoder_name, num_layers, hidden_size))
        logger.info("Langauge embedding as input: %s (size: %d)" % (self.lan_emb_as_input, lan_emb_size))
        # Initialization
        # to collect all params other than langauge model
        self.basic_parameters = []
        # collect all params for language model
        self.lm_parameters = []
        # for Pretrained LM
        # if self.pretrained_lm == 'sroberta':
            # config = SemSynRobertaConfig.from_pretrained(lm_config)
            # config.graph["num_rel_labels"] = num_src_labels
            # self.lm_encoder = SemSynRobertaModel.from_pretrained(lm_path, config=config)
            # self.lm_parameters.append(self.lm_encoder)
            # logger.info("[LM] Pretrained Language Model Type: SemSynRoberta")
            # logger.info("[LM] Pretrained Language Model Path: %s" % (lm_path))
            # lm_hidden_size = self.lm_encoder.config.hidden_size
        if self.pretrained_lm != 'none':
            self.lm_encoder = AutoModel.from_pretrained(lm_path)
            self.lm_parameters.append(self.lm_encoder)
            logger.info("[LM] Pretrained Language Model Type: %s" % (self.lm_encoder.config.model_type))
            logger.info("[LM] Pretrained Language Model Path: %s" % (lm_path))
            lm_hidden_size = self.lm_encoder.config.hidden_size
            #assert lm_hidden_size == word_dim
            #lm_hidden_size = 768
        else:
            self.lm_encoder = None
            lm_hidden_size = 0
        # for ELMo
        if self.use_elmo:
            self.elmo_encoder, elmo_hidden_size = load_elmo(elmo_path)
            self.lm_parameters.append(self.elmo_encoder)
            logger.info("[ELMo] Pretrained ELMo Path: %s" % (elmo_path))
        else:
            self.elmo_encoder = None
            elmo_hidden_size = 0
        # for pretrianed static word embedding
        if self.use_pretrained_static:
            if self.only_pretrain_static:
                num_pretrained = num_words
                logger.info("[Pretrained Static] Only use Pretrained static embeddings. (size=%d)" % num_pretrained)
            else:
                logger.info("[Pretrained Static] Pretrained static embeddings size: %d" % num_pretrained)
            self.pretrained_word_embed = nn.Embedding(num_pretrained, word_dim, _weight=embedd_word, padding_idx=1)
            self.basic_parameters.append(self.pretrained_word_embed)
            pretrained_static_size = word_dim
        else:
            self.pretrained_word_embed = None
            pretrained_static_size = 0
        # for randomly initialized static word embedding
        if self.use_random_static:
            logger.info("[Random Static] Randomly initialized static embeddings size: %d" % num_words)
            self.random_word_embed = nn.Embedding(num_words, word_dim, padding_idx=1)
            self.basic_parameters.append(self.random_word_embed)
            random_static_size = word_dim
        else:
            self.random_word_embed = None
            random_static_size = 0

        self.language_embed = None

        #self.word_embed.weight.requires_grad=False
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if use_pos else None
        if self.pos_embed is not None:
            self.basic_parameters.append(self.pos_embed)
        if use_char:
            self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
            self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)
            self.basic_parameters.append(self.char_embed)
            self.basic_parameters.append(self.char_cnn)
        else:
            self.char_embed = None
            self.char_cnn = None

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels

        enc_dim = lm_hidden_size + elmo_hidden_size + pretrained_static_size + random_static_size
        if use_char:
            enc_dim += char_dim
        if use_pos:
            enc_dim += pos_dim

        self.input_encoder_name = input_encoder_name
        if input_encoder_name == 'Linear':
            self.input_encoder = nn.Linear(enc_dim, hidden_size)
            self.position_embedding_layer = PositionEmbeddingLayer(enc_dim, dropout_prob=0,
                                                                max_position_embeddings=256)
            self.basic_parameters.append(self.input_encoder)
            self.basic_parameters.append(self.position_embedding_layer)
            self.enc_out_dim = hidden_size
        elif input_encoder_name == 'FastLSTM':
            self.input_encoder = VarFastLSTM(enc_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)
            self.basic_parameters.append(self.input_encoder)
            self.enc_out_dim = hidden_size * 2
            logger.info("dropout(p_rnn): (%.2f, %.2f)" % (p_rnn[0], p_rnn[1]))
        elif input_encoder_name == 'CPGLSTM':
            self.input_encoder = CPG_LSTM(enc_dim, hidden_size, lan_emb_size, num_layers=num_layers,
                                    batch_first=True, bidirectional=True, dropout_in=p_rnn[0], dropout_out=p_rnn[1])
            self.basic_parameters.append(self.input_encoder)
            self.enc_out_dim = hidden_size * 2
            logger.info("dropout(p_rnn): (%.2f, %.2f)" % (p_rnn[0], p_rnn[1]))
            logger.info("Langauge embedding size: %d" % lan_emb_size)
        elif input_encoder_name == 'Transformer':
            num_attention_heads = hyps['input_encoder']['num_attention_heads']
            intermediate_size = hyps['input_encoder']['intermediate_size']
            hidden_act = hyps['input_encoder']['hidden_act']
            dropout_type = hyps['input_encoder']['dropout_type']
            embedding_dropout_prob = hyps['input_encoder']['embedding_dropout_prob']
            hidden_dropout_prob = hyps['input_encoder']['hidden_dropout_prob']
            inter_dropout_prob = hyps['input_encoder']['inter_dropout_prob']
            attention_probs_dropout_prob = hyps['input_encoder']['attention_probs_dropout_prob']
            use_input_layer = hyps['input_encoder']['use_input_layer']
            use_sin_position_embedding = hyps['input_encoder']['use_sin_position_embedding']
            freeze_position_embedding = hyps['input_encoder']['freeze_position_embedding']
            initializer = hyps['input_encoder']['initializer']
            if not use_input_layer and not enc_dim == hidden_size:
                print ("enc_dim ({}) does not match hidden_size ({}) with no input layer!".format(enc_dim, hidden_size))
                exit()

            self.attention_config = AttentionEncoderConfig(input_size=enc_dim,
                                                    hidden_size=hidden_size,
                                                    num_hidden_layers=num_layers,
                                                    num_attention_heads=num_attention_heads,
                                                    intermediate_size=intermediate_size,
                                                    hidden_act=hidden_act,
                                                    dropout_type=dropout_type,
                                                    embedding_dropout_prob=embedding_dropout_prob,
                                                    hidden_dropout_prob=hidden_dropout_prob,
                                                    inter_dropout_prob=inter_dropout_prob,
                                                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                    use_input_layer=use_input_layer,
                                                    use_sin_position_embedding=use_sin_position_embedding,
                                                    freeze_position_embedding=freeze_position_embedding,
                                                    max_position_embeddings=256,
                                                    initializer=initializer,
                                                    initializer_range=0.02)
            self.input_encoder = AttentionEncoder(self.attention_config)
            self.basic_parameters.append(self.input_encoder)
            self.enc_out_dim = hidden_size
            logger.info("dropout(emb, hidden, inter, att): (%.2f, %.2f, %.2f, %.2f)" % (embedding_dropout_prob,
                                hidden_dropout_prob, inter_dropout_prob, attention_probs_dropout_prob))
            logger.info("Use Sin Position Embedding: %s (Freeze it: %s)" % (use_sin_position_embedding, freeze_position_embedding))
            logger.info("Use Input Layer: %s" % use_input_layer)

        elif input_encoder_name == 'None':
            self.input_encoder = None
            self.enc_out_dim = enc_dim
        else:
            self.input_encoder = None
            self.enc_out_dim = enc_dim

        # jeffrey 2021-9-12
        self.labels_ebbed = nn.Embedding(num_src_labels, self.rel_mlp_dim, padding_idx=0)  # padding的时候读取的向量是第几个
        self.pattern_ebbed = nn.Embedding(10+1,self.rel_mlp_dim,padding_idx=0)
        self.basic_parameters.append(self.labels_ebbed)
        self.basic_parameters.append(self.pattern_ebbed)

        # for biaffine scorer
        self.init_biaffine()
        logger.info("##### Biaffine #####")
        logger.info("MLP dim: Arc=%d, Rel=%d" % (self.arc_mlp_dim, self.rel_mlp_dim))
        logger.info("Activation function: %s" % (activation))

        assert activation in ['elu', 'leaky_relu', 'tanh', 'None']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'None':
            self.activation = None
        self.criterion_arc = nn.BCELoss(reduction='none')
        self.criterion_label = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)
        logger.info('# of Parameters: %d' % (sum([param.numel() for param in self.parameters()])))

    def init_biaffine(self):
        hid_size = self.enc_out_dim
        if self.arc_mlp_dim == -1:
            self.arc_attention = BiAffine_v2(hid_size, bias_x=True, bias_y=False)
            self.basic_parameters.append(self.arc_attention)
        else:
            self.arc_h = nn.Linear(hid_size+3*self.pos_dim, self.arc_mlp_dim)
            self.arc_c = nn.Linear(hid_size+3*self.pos_dim, self.arc_mlp_dim)
            #self.arc_attention = BiAffine(arc_mlp_dim, arc_mlp_dim)
            self.arc_attention = BiAffine_v2(self.arc_mlp_dim, bias_x=True, bias_y=False)

            self.basic_parameters.append(self.arc_h)
            self.basic_parameters.append(self.arc_c)
            self.basic_parameters.append(self.arc_attention)

        if self.rel_mlp_dim == -1:
            self.rel_attention = BiAffine_v2(hid_size, n_out=self.num_labels, bias_x=True, bias_y=True)
            self.basic_parameters.append(self.rel_attention)
        else:
            self.rel_h = nn.Linear(hid_size+3*self.pos_dim, self.rel_mlp_dim)
            self.rel_c = nn.Linear(hid_size+3*self.pos_dim, self.rel_mlp_dim)
            #self.rel_attention = BiLinear(rel_mlp_dim, rel_mlp_dim, self.num_labels)
            self.rel_attention = BiAffine_v2(self.rel_mlp_dim, n_out=self.num_labels, bias_x=True, bias_y=True)
            self.basic_parameters.append(self.rel_h)
            self.basic_parameters.append(self.rel_c)
            self.basic_parameters.append(self.rel_attention)

    def _basic_parameters(self):
        params = [p.parameters() for p in self.basic_parameters]
        #print (params)
        return itertools.chain(*params)

    def _lm_parameters(self):
        if not self.lm_parameters:
            return None
        if len(self.lm_parameters) == 1:
            return self.lm_parameters[0].parameters()
        else:
            params = [p.parameters() for p in self.lm_parameters]
            return itertools.chain(*params)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_char is None and self.char_embed is not None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)
        if self.random_word_embed is not None:
            nn.init.uniform_(self.random_word_embed.weight, -0.1, 0.1)
        if self.language_embed is not None:
            nn.init.uniform_(self.language_embed.weight, -0.1, 0.1)
        # 初始化embedding Jeffrey
        nn.init.uniform_(self.labels_ebbed.weight,-0.1,0.1)
        nn.init.uniform_(self.pattern_ebbed.weight,-0.1,0.1)
        with torch.no_grad():
            if self.pretrained_word_embed is not None:
                self.pretrained_word_embed.weight[self.pretrained_word_embed.padding_idx].fill_(0)
            if self.random_word_embed is not None:
                self.random_word_embed.weight[self.random_word_embed.padding_idx].fill_(0)
            if self.char_embed is not None:
                self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)
            self.labels_ebbed.weight[0].fill_(0)
            self.pattern_ebbed.weight[0].fill_(0)

        if self.arc_mlp_dim != -1 and self.rel_mlp_dim != -1:
            if self.act_func == 'leaky_relu':
                nn.init.kaiming_uniform_(self.arc_h.weight, a=0.1, nonlinearity='leaky_relu')
                nn.init.kaiming_uniform_(self.arc_c.weight, a=0.1, nonlinearity='leaky_relu')
                nn.init.kaiming_uniform_(self.rel_h.weight, a=0.1, nonlinearity='leaky_relu')
                nn.init.kaiming_uniform_(self.rel_c.weight, a=0.1, nonlinearity='leaky_relu')
            else:
                nn.init.xavier_uniform_(self.arc_h.weight)
                nn.init.xavier_uniform_(self.arc_c.weight)
                nn.init.xavier_uniform_(self.rel_h.weight)
                nn.init.xavier_uniform_(self.rel_c.weight)

            nn.init.constant_(self.arc_h.bias, 0.)
            nn.init.constant_(self.arc_c.bias, 0.)
            nn.init.constant_(self.rel_h.bias, 0.)
            nn.init.constant_(self.rel_c.bias, 0.)

        if self.input_encoder_name == 'Linear':
            nn.init.xavier_uniform_(self.input_encoder.weight)
            nn.init.constant_(self.input_encoder.bias, 0.)


    def _lm_embed(self, input_ids=None, first_index=None, src_heads=None, src_types=None, debug=False):
        """
        Input:
            input_ids: (batch, max_bpe_len)
            first_index: (batch, seq_len)
        """
        # (batch, max_bpe_len, hidden_size)
        if src_heads is not None and src_types is not None:
            # (batch, max_bpe_len, hidden_size)
            lm_output = self.lm_encoder(input_ids, heads=src_heads, rels=src_types)[0]
        else:
            # (batch, max_bpe_len, hidden_size)
            lm_output = self.lm_encoder(input_ids)[0]
        size = list(first_index.size()) + [lm_output.size()[-1]]
        # (batch, seq_len, hidden_size)
        output = lm_output.gather(1, first_index.unsqueeze(-1).expand(size))
        if debug:
            print (lm_output.size())
            print (output.size())
        return output


    def _embed(self, input_word, input_pretrained, input_char, input_pos, bpes=None,
                first_idx=None, input_elmo=None, lan_id=None, src_heads=None, src_types=None):
        batch_size, seq_len = input_word.size()
        word_embeds = []
        if self.random_word_embed is not None:
            random_embed = self.random_word_embed(input_word)
            word_embeds.append(random_embed)
        if self.pretrained_word_embed is not None:
            if self.only_pretrain_static:
                pretrained_embed = self.pretrained_word_embed(input_word)
            else:
                pretrained_embed = self.pretrained_word_embed(input_pretrained)
            word_embeds.append(pretrained_embed)
        if self.lm_encoder is not None:
            lm_embed = self._lm_embed(bpes, first_idx, src_heads=src_heads, src_types=src_types)
            word_embeds.append(lm_embed)
        if self.elmo_encoder is not None:
            elmo_embed = self.elmo_encoder(input_elmo)['elmo_representations'][0]
            word_embeds.append(elmo_embed)
        if len(word_embeds) == 1:
            enc_word = word_embeds[0]
        else:
            enc_word = torch.cat(word_embeds, dim=-1)

        if self.lan_emb_as_input:
            # (lan_emb_size)
            lan_emb = self.language_embed(lan_id)
            #print (lan_id, '\n', lan_emb)
            lan_emb = lan_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
            enc_word = torch.cat([enc_word, lan_emb], dim=2)

        if self.char_embed is not None:
            # [batch, length, char_length, char_dim]
            char = self.char_cnn(self.char_embed(input_char))
            #char = self.dropout_in(char)
            # concatenate word and char [batch, length, word_dim+char_filter]
            enc_word = torch.cat([enc_word, char], dim=2)

        if self.pos_embed is not None:
            # [batch, length, pos_dim]
            enc_pos = self.pos_embed(input_pos)
            # apply dropout on input
            #pos = self.dropout_in(pos)
            if self.training:
                #print ("enc_word:\n", enc_word)
                # mask by token dropout
                enc_word, enc_pos = drop_input_independent(enc_word, enc_pos, self.p_in)
                #print ("enc_word (a):\n", enc_word)
            enc = torch.cat([enc_word, enc_pos], dim=2)
        else:
            enc = self.dropout_in(enc_word)
        return enc

    def _input_encoder(self, embeddings, mask=None, lan_id=None):

        #print ("input_word:\n", input_word)
        #print ("input_pretrained:\n", input_pretrained)

        # apply dropout word on input
        #word = self.dropout_in(word)

        ht = None
        # output from rnn [batch, length, hidden_size]
        if self.input_encoder_name == 'Linear':
            # sequence shared mask dropout
            enc = self.dropout_in(embeddings.transpose(1, 2)).transpose(1, 2)
            enc = self.position_embeembeddingsdding_layer(enc)
            output = self.input_encoder(enc)
        elif self.input_encoder_name == 'Transformer':
            # sequence shared mask dropout
            # apply this dropout in transformer after added position embedding
            #enc = self.dropout_in(enc.transpose(1, 2)).transpose(1, 2)
            all_encoder_layers = self.input_encoder(embeddings, mask)
            # [batch, length, hidden_size]
            output = all_encoder_layers[-1]
        elif self.input_encoder_name == 'CPGLSTM':
            enc = self.dropout_in(embeddings.transpose(1, 2)).transpose(1, 2)
            lan_emb = self.language_embed(lan_id)
            #print (lan_emb)
            output, ht = self.input_encoder(lan_emb, enc, mask)
        elif self.input_encoder_name == 'FastLSTM':
            # for 'FastLSTM'
            # sequence shared mask dropout
            #enc = self.dropout_in(embeddings.transpose(1, 2)).transpose(1, 2)
            enc = embeddings
            output, ht = self.input_encoder(enc, mask)
        elif self.input_encoder_name == 'None':
            output = embeddings
            self.encoder_output = output
            return output, ht

        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        self.encoder_output = output
        return output, ht

    def _arc_mlp(self, hidden_i,hidden_j):
        if self.arc_mlp_dim == -1:
            arc_h, arc_c = hidden_i, hidden_j
        else:
            if self.activation:
                # output size [batch, length, arc_mlp_dim]
                arc_c = self.activation(self.arc_c(hidden_i))
                arc_h = self.activation(self.arc_h(hidden_j))
            else:
                arc_c = self.arc_c(hidden_i)
                arc_h = self.arc_h(hidden_j)


        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        return arc_h, arc_c

    def _rel_mlp(self, hidden_i,hidden_j):
        if self.rel_mlp_dim == -1:
            rel_h, rel_c = hidden_i, hidden_j
        else:
            if self.activation:
                # output size [batch, length, rel_mlp_dim]
                rel_c = self.activation(self.rel_c(hidden_i))
                rel_h = self.activation(self.rel_h(hidden_j))

            else:
                rel_c = self.rel_c(hidden_i)
                rel_h = self.rel_h(hidden_j)

        # apply dropout on rel
        # [batch, length, dim] --> [batch, 2 * length, dim]
        rel = torch.cat([rel_h, rel_c], dim=1)
        rel = self.dropout_out(rel.transpose(1, 2)).transpose(1, 2)
        rel_h, rel_c = rel.chunk(2, 1)
        rel_h = rel_h.contiguous()
        rel_c = rel_c.contiguous()

        return rel_h, rel_c

    def accuracy(self, arc_logits, rel_logits, heads, rels, mask, debug=False):
        """
        arc_logits: (batch, seq_len, seq_len)
        rel_logits: (batch, n_rels, seq_len, seq_len)
        heads: (batch, seq_len)
        rels: (batch, seq_len)
        mask: (batch, seq_len)
        """
        total_arcs = heads.sum()
        # (batch, seq_len)
        arc_preds = arc_logits.ge(0.5).float()
        # (batch_size, seq_len, seq_len, n_rels)
        transposed_rel_logits = rel_logits.permute(0, 2, 3, 1)
        # (batch_size, seq_len, seq_len)
        rel_preds = transposed_rel_logits.argmax(-1)

        ones = torch.ones_like(heads)
        zeros = torch.zeros_like(heads)
        arc_correct = (torch.where(arc_preds==heads, ones, zeros) * heads).sum()
        rel_correct = (torch.where(rel_preds==rels, ones, zeros) * heads * arc_preds).sum()
        # ========================== f1 ================================
        arc_preds = arc_preds * mask
        arc_pred_num = arc_preds.sum()
        # =============================================================

        if debug:
            print ("arc_logits:\n", arc_logits)
            print ("arc_preds:\n", arc_preds)
            print ("heads:\n", heads)
            print ("rel_preds:\n", rel_preds)
            print ("rels:\n", rels)
            print ("mask:\n", mask)
            print ("total_arcs:\n", total_arcs)
            print ("arc_correct:\n", arc_correct)
            print ("rel_correct:\n", rel_correct)

        return arc_correct.cpu().numpy(), rel_correct.cpu().numpy(), total_arcs.cpu().numpy(),arc_pred_num

    def _argmax(self, logits):
        """
        Input:
            logits: (batch, seq_len, seq_len), as probs
        Output:
            graph_matrix: (batch, seq_len, seq_len), in onehot
        """
        # (batch, seq_len)
        index = logits.argmax(-1).unsqueeze(2)
        # (batch, seq_len, seq_len)
        graph_matrix = torch.zeros_like(logits).int()
        graph_matrix.scatter_(-1, index, 1)
        return graph_matrix.detach()

    def get_neaest_par(self,heads):
        # BFS(i)
        # BFS(j)
        # 同时求交集
        batch,q_len,q_len = heads.size()
        heads = heads.cpu().numpy().tolist()
        batch_par = np.zeros((batch,q_len,q_len,q_len))
        batch_pattern = np.zeros((batch,q_len,q_len))
        for i in range(batch):
            lca = LCADAG(heads[i])
            patterns = lca.get_all_pattern()
            par = lca.get_ancestor()
            batch_par[i]= par
            batch_pattern[i] = patterns
        return torch.from_numpy(np.array(batch_par)).to(self.device),torch.from_numpy(np.array(batch_pattern)).to(self.device)


    def forward(self, input_word, input_pretrained, input_char, input_pos, heads, rels,
                bpes=None, first_idx=None, input_elmo=None, mask=None, lan_id=None,
                src_heads=None, src_types=None,
                origin_heads=None,origin_types=None,
                info1=None,info2=None):
        # Pre-process
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=heads.device).gt(0).float().unsqueeze(0) * mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))

        # (batch, seq_len, embed_size)
        embeddings = self._embed(input_word, input_pretrained, input_char, input_pos,
                                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id,
                                src_heads=src_heads, src_types=src_types)
        # (batch, seq_len, hidden_size)
        encoder_output, _ = self._input_encoder(embeddings, mask=mask, lan_id=lan_id)
        # *********** pattern embedding **************
        # (batch,seq_len,seq_len,hidden_size)
        encoder_output_i = encoder_output.unsqueeze(2).repeat(1,1,seq_len,1) #(i<-j),i的表示
        # encoder_output_j = encoder_output.unsqueeze(2).expand(2,seq_len) #(j->i),j的表示

        node_labeling = torch.sum(self.labels_ebbed(origin_types),dim=2)
        head_num = torch.sum(origin_heads,dim=-1)
        # (batch,seq_len,hidden_size)
        # 需要解决nan
        node_labeling = node_labeling.div(head_num.unsqueeze(-1).float())  #每个节点label的平均
        node_labeling = torch.where(torch.isnan(node_labeling),torch.full_like(node_labeling,0),node_labeling)
        # (batch,seq_len,seq_len) distant_embedding
        # (batch,seq_len,seq_len,seq_len) nearest_par_labeling
        # nearest_par_labeling,distant_embedding = self.get_neaest_par(origin_heads)
        nearest_par_labeling,distant_embedding = info1,info2
        nearest_par_embedd = torch.sum(self.labels_ebbed(nearest_par_labeling.long()),dim=-2)
        par_num = torch.sum(nearest_par_labeling,dim=-1)
        # 需要解决nan
        nearest_par_embedd = nearest_par_embedd.div(par_num.unsqueeze(-1).float())   #(batch,seq_len,seq_len,hidden_size)
        nearest_par_embedd = torch.where(torch.isnan(nearest_par_embedd), torch.full_like(nearest_par_embedd, 0), nearest_par_embedd)
        distant_embedd = self.pattern_ebbed(distant_embedding.long())
        # 进行embedding 合并
        merge1 = torch.cat((encoder_output_i,node_labeling.unsqueeze(2).repeat(1,1,seq_len,1)),dim=-1)
        merge2 = torch.cat((merge1,nearest_par_embedd),dim=-1)
        embedding_i = torch.cat((merge2,distant_embedd),dim=-1)
        embedding_j = torch.cat((merge2,distant_embedd.transpose(1,2)),dim=-1)


        # ***********pattern embedding ***********

        # (batch, seq_len, arc_mlp_dim)
        arc_h, arc_c = self._arc_mlp(embedding_i,embedding_j)

        # (batch, seq_len, seq_len)
        arc_logits = self.arc_attention(arc_c, arc_h.transpose(1,2),seq_len)
        # arc_attention1 jeffrey 2021-9-12
        # arc_embedding = self.arc_ebbed((origin_heads>0).type(torch.long))
        # arc_logits1 = arc_embedding @ torch.unsqueeze(arc_c,-1)

        #print ('graph_matrix:\n', graph_matrix)
        #print ('arc_loss:', arc_losses[-1].sum())

        # (batch, length, rel_mlp_dim)
        rel_h, rel_c = self._rel_mlp(embedding_i,embedding_j)
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attention(rel_c, rel_h.transpose(1,2),seq_len)
        # rel_attention1 jeffrey 2021-9-12

        # rel_embedding = self.labels_ebbed(origin_types)
        # rel_logits1 = rel_embedding @ torch.unsqueeze(rel_c,-1)
        # self.logger.info(rel_logits.shape)
        # self.logger.info(rel_logits1.shape)

        # 插入计算
        arc_logits = arc_logits  # + arc_logits1.squeeze(-1)+rel_logits1.squeeze(-1)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_mask = mask_3D.eq(0)
            arc_logits = arc_logits.masked_fill(minus_mask, float('-inf'))
        # arc_loss shape [batch, length_c]
        arc_logits = torch.sigmoid(arc_logits)
        arc_loss = self.criterion_arc(arc_logits, heads.float())
        # mask invalid position to 0 for sum loss
        if mask is not None:
            arc_loss = arc_loss * mask_3D
        # [batch, length - 1] -> [batch] remove the symbolic root
        arc_loss = arc_loss[:, 1:].sum(dim=1)

        #rel_loss = self.criterion(out_type.transpose(1, 2), rels)
        rel_loss = self.criterion_label(rel_logits, rels)
        if mask is not None:
            rel_loss = rel_loss * mask_3D
        rel_loss = rel_loss * heads
        rel_loss = rel_loss[:, 1:].sum(dim=1)

        statistics = self.accuracy(arc_logits, rel_logits, heads, rels, mask_3D)

        return (arc_loss, rel_loss), statistics


    def decode(self, input_word, input_pretrained, input_char, input_pos, mask=None,
               bpes=None, first_idx=None, input_elmo=None, lan_id=None, leading_symbolic=0,target_mask=None,
               src_heads=None, src_types=None,origin_heads=None,origin_types=None,info1=None,info2=None):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and rels.

        """
        logger = get_logger("Decode")
        # Pre-process
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=input_word.device).gt(0).float().unsqueeze(0) * mask
        mask_3D = (root_mask.unsqueeze(-1) * mask.unsqueeze(1))
        # (batch, seq_len, embed_size)
        embeddings = self._embed(input_word, input_pretrained, input_char, input_pos,
                                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id,
                                src_heads=src_heads, src_types=src_types)
        # (batch, seq_len, hidden_size)
        encoder_output, _ = self._input_encoder(embeddings, mask=mask, lan_id=lan_id)

        # decode 也需要加入辅助信息
        encoder_output_i = encoder_output.unsqueeze(2).repeat(1, 1, seq_len, 1)  # (i<-j),i的表示
        # encoder_output_j = encoder_output.unsqueeze(2).expand(2,seq_len) #(j->i),j的表示

        node_labeling = torch.sum(self.labels_ebbed(origin_types), dim=2)
        head_num = torch.sum(origin_heads, dim=-1)
        # (batch,seq_len,hidden_size)
        # 需要解决nan
        node_labeling = node_labeling.div(head_num.unsqueeze(-1).float())  # 每个节点label的平均
        node_labeling = torch.where(torch.isnan(node_labeling), torch.full_like(node_labeling, 0), node_labeling)
        # (batch,seq_len,seq_len) distant_embedding
        # (batch,seq_len,seq_len,seq_len) nearest_par_labeling
        #nearest_par_labeling, distant_embedding = self.get_neaest_par(origin_heads)
        nearest_par_labeling, distant_embedding = info1,info2
        nearest_par_embedd = torch.sum(self.labels_ebbed(nearest_par_labeling.long()), dim=-2)
        par_num = torch.sum(nearest_par_labeling, dim=-1)
        # 需要解决nan
        nearest_par_embedd = nearest_par_embedd.div(par_num.unsqueeze(-1).float())  # (batch,seq_len,seq_len,hidden_size)
        nearest_par_embedd = torch.where(torch.isnan(nearest_par_embedd), torch.full_like(nearest_par_embedd, 0), nearest_par_embedd)
        distant_embedd = self.pattern_ebbed(distant_embedding.long())
        # 进行embedding 合并
        merge1 = torch.cat((encoder_output_i, node_labeling.unsqueeze(2).repeat(1, 1, seq_len, 1)), dim=-1)
        merge2 = torch.cat((merge1, nearest_par_embedd), dim=-1)
        embedding_i = torch.cat((merge2, distant_embedd), dim=-1)

        embedding_j = torch.cat((merge2, distant_embedd.transpose(1, 2)), dim=-1)


        # (batch, seq_len, arc_mlp_dim)
        arc_h, arc_c = self._arc_mlp(embedding_i,embedding_j)
        # (batch, seq_len, seq_len)

        arc_logits = self.arc_attention(arc_c, arc_h.transpose(1,2),seq_len)

        # (batch, length, rel_mlp_dim)
        rel_h, rel_c = self._rel_mlp(embedding_i,embedding_j)
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attention(rel_c, rel_h.transpose(1,2),seq_len)
        # (batch, n_rels, seq_len_c, seq_len_h)
        # => (batch, length_h, length_c, num_labels)
        #rel_logits = rel_logits.permute(0, 3, 2, 1)

        if mask is not None:
            minus_mask = mask_3D.eq(0)
            arc_logits.masked_fill_(minus_mask, float('-inf'))

        arc_preds = torch.sigmoid(arc_logits).ge(0.5).float()  # 多个arc
        # (batch_size, len_c, len_h, n_rels)
        transposed_type_logits = rel_logits.permute(0, 2, 3, 1)  # permute重新排列张量
        # (batch_size, seq_len, seq_len)
        # 没做softmax
        #type_preds = transposed_type_logits.argmax(-1)  # 在只有一个label的情况下找到最大的索引
        type_preds = F.softmax(transposed_type_logits,dim=-1)
        # jeffrey: 加一个mask 过滤不在目标标签体系中的label:
        if target_mask is not None:
            target_mask = torch.Tensor(target_mask)
            target_mask = target_mask.to(self.device)
            s = target_mask.size()[0]
            c = target_mask.view([1, 1, 1,s])
            d = c.repeat([batch_size, seq_len, seq_len,1])
            type_preds = type_preds * d  # mask
            # logger.info(type_preds[0, 1, 2, :])
        type_preds = type_preds.argmax(dim=-1)
        return arc_preds * mask_3D, type_preds


    def get_probs(self, input_word, input_pretrained, input_char, input_pos, mask=None,
                bpes=None, first_idx=None, input_elmo=None, lan_id=None, leading_symbolic=0):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and rels.

        """
        # Pre-process
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=input_word.device).gt(0).float().unsqueeze(0) * mask

        # (batch, seq_len, embed_size)
        embeddings = self._embed(input_word, input_pretrained, input_char, input_pos,
                                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id)
        # (batch, seq_len, hidden_size)
        encoder_output, _ = self._input_encoder(embeddings, mask=mask, lan_id=lan_id)

        # (batch, seq_len, arc_mlp_dim)
        arc_h, arc_c = self._arc_mlp(encoder_output)
        # (batch, seq_len, seq_len)
        arc_logits = self.arc_attention(arc_c, arc_h)

        # (batch, length, rel_mlp_dim)
        rel_h, rel_c = self._rel_mlp(encoder_output)
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attention(rel_c, rel_h)
        # (batch, n_rels, seq_len_c, seq_len_h)
        # => (batch, length_h, length_c, num_labels)
        rel_logits = rel_logits.permute(0,3,2,1)

        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            arc_logits.masked_fill_(minus_mask, float('-inf'))
        # arc_loss shape [batch, length_h, length_c]
        arc_probs = F.softmax(arc_logits, dim=1)
        # rel_loss shape [batch, length_h, length_c, num_labels]
        rel_probs = F.softmax(rel_logits, dim=3)

        return arc_probs, rel_probs

    def get_logits(self, input_word, input_pretrained, input_char, input_pos, mask=None,
                bpes=None, first_idx=None, input_elmo=None, lan_id=None, leading_symbolic=0):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and rels.

        """
        # Pre-process
        batch_size, seq_len = input_word.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=input_word.device).gt(0).float().unsqueeze(0) * mask

        # (batch, seq_len, embed_size)
        embeddings = self._embed(input_word, input_pretrained, input_char, input_pos,
                                bpes=bpes, first_idx=first_idx, input_elmo=input_elmo, lan_id=lan_id)
        # (batch, seq_len, hidden_size)
        encoder_output, _ = self._input_encoder(embeddings, mask=mask, lan_id=lan_id)

        # (batch, seq_len, arc_mlp_dim)
        arc_h, arc_c = self._arc_mlp(encoder_output)
        # (batch, seq_len, seq_len)
        arc_logits = self.arc_attention(arc_c, arc_h)

        # (batch, length, rel_mlp_dim)
        rel_h, rel_c = self._rel_mlp(encoder_output)
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attention(rel_c, rel_h)
        # (batch, n_rels, seq_len_c, seq_len_h)
        # => (batch, length_h, length_c, num_labels)
        rel_logits = rel_logits.permute(0,3,2,1)

        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            arc_logits.masked_fill_(minus_mask, float('-inf'))

        return arc_logits, rel_logits