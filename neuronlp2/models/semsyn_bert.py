# coding=utf-8

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import PretrainedConfig, BertPreTrainedModel
from neuronlp2.models.modeling_bert import (BertEmbeddings, BertSelfOutput, BertAttention, BertIntermediate, BertOutput, BertLayer, BertPooler)
from transformers.activations import ACT2FN

from neuronlp2.models.gate import HighwayGateLayer, ConstantGateLayer
from neuronlp2.models.graph_convolution import GCNLayer, RGCNLayer
from neuronlp2.models.graph_attention import GATLayer
from neuronlp2.models.gnn_encoder import GNNEncoder
from neuronlp2.models.graph_mask_encoder import BertGraphMaskLayer

#import logging
#logger = logging.getLogger(__name__)
from neuronlp2.io import get_logger
logger = get_logger(__name__)

class SemSynBertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        num_labels=2,
        fusion_type="joint",
        graph=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.num_labels = num_labels

        # GNN options
        self.fusion_type = fusion_type
        self.graph = graph


class PalBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.graph["num_attention_heads"] != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.graph["num_attention_heads"])
            )

        input_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size

        self.num_attention_heads = config.graph["num_attention_heads"]
        self.attention_head_size = int(input_size / config.graph["num_attention_heads"])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class PalGNNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.do_pal_project = config.graph["do_pal_project"]
        self.encoder_type = config.graph["encoder"]
        self.use_output_layer = "use_output_layer" in config.graph and config.graph["use_output_layer"]
        self.use_ff_layer = "use_ff_layer" in config.graph and config.graph["use_ff_layer"]
        
        if isinstance(config.hidden_act, str):
            self.hidden_act_fn = ACT2FN[config.hidden_act]
        else:
            self.hidden_act_fn = config.hidden_act

        if config.graph["do_pal_project"]:
            self.dense_down = nn.Linear(config.hidden_size, config.graph["lowrank_size"])
            #output_size = 2*config.graph["lowrank_size"] if (config.graph["use_rel_embedding"] and config.rel_combine_type=="concat") else config.graph["lowrank_size"]
            output_size = config.graph["lowrank_size"]
            self.dense_up = nn.Linear(output_size, config.hidden_size)
        
        if self.encoder_type == "GCN":
            self.attention = GCNLayer(config)
        elif self.encoder_type == "RGCN":
            self.attention = RGCNLayer(config)
        elif self.encoder_type == "GAT":
            self.attention = GATLayer(config)
        elif self.encoder_type == "ATT": # vanilla attention
            self.attention = PalBertSelfAttention(config)
        elif self.encoder_type == "LIN": # linear 
            self.attention = None

        if self.use_output_layer:
            self.attention_output = BertSelfOutput(config)
        if self.use_ff_layer:
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)


    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        heads=None, 
        rels=None,
    ):

        input_tensor = self.dense_down(hidden_states) if self.do_pal_project else hidden_states
        if self.encoder_type == "GCN":
            attention_hiddens = self.attention(input_tensor, attention_mask, heads=heads, rels=rels)
        elif self.encoder_type == "RGCN":
            attention_hiddens = self.attention(input_tensor, attention_mask, heads=heads, rels=rels)
        elif self.encoder_type == "GAT":
            attention_hiddens = self.attention(input_tensor, attention_mask, heads=heads, rels=rels)
        elif self.encoder_type == "ATT":
            attention_hiddens = self.attention(input_tensor, attention_mask)[0]
        elif self.encoder_type == "LIN":
            # for linear we add act in between
            attention_hiddens = self.hidden_act_fn(input_tensor)
        
        attention_hiddens = self.dense_up(attention_hiddens) if self.do_pal_project else attention_hiddens
        if self.encoder_type != "LIN":
            attention_hiddens = self.hidden_act_fn(attention_hiddens)

        if self.use_output_layer:
            attention_output = self.attention_output(attention_hiddens, hidden_states)
        else:
            attention_output = attention_hiddens

        if self.use_ff_layer:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        else:
            layer_output = attention_output
        
        return layer_output


class ResidualBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_fusion_gate = config.graph["use_fusion_gate"]
        if self.use_fusion_gate:
            if config.graph["fusion_gate"] == "ConstantGateLayer":
                self.gate = ConstantGateLayer(config.graph["fusion_gate_const"])
            else:
                self.gate = eval(config.graph["fusion_gate"])(config.hidden_size)

    def forward(self, hidden_states, input_tensor, res_layer):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.use_fusion_gate:
            hidden_states = self.gate(hidden_states + input_tensor, res_layer)
        else:
            hidden_states = hidden_states + input_tensor + res_layer
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class ResidualGNNBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = ResidualBertOutput(config)

        self.fusion_type = config.fusion_type
        self.res_layer = PalGNNLayer(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=True,
        heads=None,
        rels=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        res_output = self.res_layer(hidden_states, attention_mask, heads, rels)
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, res_output)

        outputs = (layer_output,) + outputs
        return outputs


class SemSynBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_encoder = config.graph["encoder"]
        self.fusion_type = config.fusion_type
        self.structured_layers = config.graph["structured_layers"] if config.graph["structured_layers"] is not None else [i for i in range(config.num_hidden_layers)]
        self.use_rel_embedding = self.graph_encoder != "RGCN" and config.graph["use_rel_embedding"]
        self.data_flow = config.graph["data_flow"]

        if self.fusion_type == "mask":
            self.layer = nn.ModuleList([BertGraphMaskLayer(config) if i in self.structured_layers
                                    else BertLayer(config) for i in range(config.num_hidden_layers)])
        elif self.fusion_type in ["inter", "top"]:
            self.layer = nn.ModuleList([BertLayer(config) for i in range(config.num_hidden_layers)])
        elif self.fusion_type == "residual":
            self.layer = nn.ModuleList([ResidualGNNBertLayer(config) if i in self.structured_layers
                                    else BertLayer(config) for i in range(config.num_hidden_layers)])

        if self.fusion_type == "top":
            self.gnn_encoder = GNNEncoder(config)

        self.inter_gnn_layers = None
        if self.fusion_type == "inter":
            self.inter_gnn_layers = nn.ModuleList([PalGNNLayer(config) if i in self.structured_layers
                                    else None for i in range(config.num_hidden_layers)])
            self.use_fusion_gate = config.graph["use_fusion_gate"]
            if self.use_fusion_gate:
                if config.graph["fusion_gate"] == "ConstantGateLayer":
                    self.gates = nn.ModuleList([ConstantGateLayer(config.graph["fusion_gate_const"]) if i in self.structured_layers
                                    else None for i in range(config.num_hidden_layers)])
                else:
                    self.gates = nn.ModuleList([eval(config.graph["fusion_gate"])(config.hidden_size) if i in self.structured_layers
                                    else None for i in range(config.num_hidden_layers)])

        if self.use_rel_embedding:
            if self.graph_encoder == "GAT":
                # should have same size as one attention head
                in_out_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size
                config.graph["rel_embed_size"] = int(in_out_size / config.graph["num_attention_heads"])
            self.num_rel_labels = config.graph["num_rel_labels"]
            self.num_all_rel_labels = self.num_rel_labels * 2 if self.data_flow == "bidir" else self.num_rel_labels
            self.rel_embeddings = nn.Embedding(self.num_all_rel_labels, config.graph["rel_embed_size"], padding_idx=0)
            self.rel_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.show_info()

    def show_info(self):
        self.use_output_layer = "use_output_layer" in self.config.graph and self.config.graph["use_output_layer"]
        self.use_ff_layer = "use_ff_layer" in self.config.graph and self.config.graph["use_ff_layer"]
        logger.info("###### SemSynBERT Encoder ######")
        logger.info("graph_encoder = {}, fusion_type = {}".format('GraphMask' if self.fusion_type =="mask" else self.graph_encoder, self.fusion_type))
        if self.fusion_type in ["residual", "top"]:
            logger.info("use_output_layer = {}, use_ff_layer = {}".format(self.use_output_layer, self.use_ff_layer))
        logger.info("data_flow = {}, top num_layers = {}, structured_layers = {}".format(self.data_flow, 
                                                    self.config.graph["num_layers"] if self.fusion_type=="top" else "N/A",
                                                    self.structured_layers if self.fusion_type!="top" else "N/A"))
        logger.info("lowrank_size = {} (hidden = {}), num_attention_heads = {}".format(self.config.graph["lowrank_size"] if self.config.graph["do_pal_project"] else "N/A",
                                                self.config.hidden_size, self.config.graph["num_attention_heads"] if self.graph_encoder in ["GAT","ATT"] else "N/A"))
        logger.info("data_flow_gate = {}".format(self.config.graph["data_flow_gate"] if self.config.graph["use_data_flow_gate"] 
                                                    and self.graph_encoder=="GCN" else "N/A"))
        logger.info("fusion_gate = {} (const = {})".format(self.config.graph["fusion_gate"] if self.config.graph["use_fusion_gate"] and self.fusion_type!="top" else "N/A",
                                                self.config.graph["fusion_gate_const"] if self.config.graph["fusion_gate"] == "ConstantGateLayer" else "N/A"))
        logger.info("rel_embed_size = {}, num_basic_matrix = {}".format(self.config.graph["rel_embed_size"] if self.use_rel_embedding and self.graph_encoder!="RGCN" else "N/A",
                                                self.config.graph["num_basic_matrix"] if self.graph_encoder=="RGCN" else "N/A"))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=False,
        heads=None,
        rels=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # preprocess
        # convert rels to embedding
        if self.fusion_type in ["top","residual","inter"] and self.use_rel_embedding:
            if self.data_flow == "c2h":
                # do not need to add offset since no h2c label
                rels = rels.permute(0,2,1)
            if self.data_flow == "bidir":
                zeros = torch.zeros_like(rels, dtype=torch.long, device=rels.device)
                reverse_rels = rels.permute(0,2,1) + self.num_rel_labels
                reverse_rels = torch.where(reverse_rels>self.num_rel_labels, reverse_rels, zeros)
                rels = self.rel_embeddings(rels) + self.rel_embeddings(reverse_rels)
            else:
                rels = self.rel_embeddings(rels)
            rels = self.rel_dropout(rels)

        # convert heads to mask like format
        if self.fusion_type in ["top","residual","inter"] and self.graph_encoder == "GAT":
            if self.data_flow == "c2h":
                heads = heads.permute(0, 2, 1)
            elif self.data_flow == "bidir":
                heads = heads + heads.permute(0, 2, 1)
                # deal with entries == 2
                heads = torch.where(heads>0, torch.ones_like(heads), torch.zeros_like(heads))
            # (batch, 1, seq_len, seq_len), valid arc = 0, invalid arc = -10000.0
            heads = heads.unsqueeze(1)
            heads = (1.0 - heads) * -10000.0

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                if self.fusion_type not in ["inter", "top"] and i in self.structured_layers:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask,
                        head_mask=layer_head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        output_attentions=output_attentions,
                        heads=heads,
                        rels=rels
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=attention_mask,
                        head_mask=layer_head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        output_attentions=output_attentions,
                    )
            hidden_states = layer_outputs[0]

            if self.fusion_type == "inter" and i in self.structured_layers:
                inter_hidden_states = self.inter_gnn_layers[i](
                    hidden_states,
                    attention_mask=attention_mask,
                    heads=heads,
                    rels=rels,
                )
                if self.use_fusion_gate:
                    hidden_states = self.gates[i](hidden_states, inter_hidden_states)
                else:
                    hidden_states = 0.5*(hidden_states + inter_hidden_states)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.fusion_type == "top":
            hidden_states = self.gnn_encoder(
                                    hidden_states, 
                                    attention_mask=attention_mask, 
                                    heads=heads, 
                                    rels=rels)

        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class SemSynBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = SemSynBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        heads = heads.to_dense() if heads is not None and heads.is_sparse else heads
        rels = rels.to_dense() if rels is not None and rels.is_sparse else rels

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        #if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]


class SemSynBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = SemSynBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        controller="main",
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        heads = heads.to_dense() if heads is not None and heads.is_sparse else heads
        rels = rels.to_dense() if rels is not None and rels.is_sparse else rels

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class SemSynBertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = SemSynBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        heads = heads.to_dense() if heads is not None and heads.is_sparse else heads
        rels = rels.to_dense() if rels is not None and rels.is_sparse else rels
        if heads is not None:
            if len(heads.size()) == 4:
                #   (batch, n_choices, seq_len, seq_len)
                # =>(batch*n_choices , seq_len, seq_len)
                heads = heads.view(-1, heads.size(-2), heads.size(-1))
            elif len(heads.size()) == 5:
                #    (batch, n_choices, n_mask, seq_len, seq_len)
                # => (batch*n_choices, n_mask, seq_len, seq_len)
                heads = heads.view(-1, heads.size(-3), heads.size(-2), heads.size(-1))
        rels = rels.view(-1, rels.size(-2), rels.size(-1)) if rels is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


class SemSynBertForQuestionAnswering(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = SemSynBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        heads = heads.to_dense() if heads is not None and heads.is_sparse else heads
        rels = rels.to_dense() if rels is not None and rels.is_sparse else rels
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output



class SemSynBertForArgumentLabel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = SemSynBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.activation = nn.SELU()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        predicate_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )

        sequence_output = outputs[0]
        # (batch, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)

        # change 0/1 to bool matrix, must assure that each line has exactly one 1
        predicate_hidden = sequence_output[(predicate_mask == 1)] 

        predicate_hidden = predicate_hidden.unsqueeze(1).repeat(1, sequence_output.size(1), 1)
        #print ("predicate_hidden:", predicate_hidden.size())
        output = torch.cat([sequence_output, predicate_hidden], dim=-1)
        output = self.activation(self.dense(output))
        # (batch, seq_len, num_labels)
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.contiguous().view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                #print ("active_labels:\n", active_labels)
                #print ("active_logits:\n", torch.argmax(active_logits, -1))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class SemSynBertForPredicateSense(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = SemSynBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        predicate_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        heads=None,
        rels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            heads=heads,
            rels=rels,
        )

        sequence_output = outputs[0]
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.contiguous().view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output