import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_utils import apply_chunking_to_forward
from transformers import RobertaModel, RobertaConfig, PreTrainedModel
from .modeling_bert import BiaffineAttention
from neuronlp2.nn import VarFastLSTM

import logging
logger = logging.getLogger(__name__)


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaForArgumentLabel(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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


class RobertaForPredicateSense(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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


class RobertaForSDP(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.arc_mlp_dim = 512
        self.rel_mlp_dim = 128

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.input_encoder = VarFastLSTM(config.hidden_size, config.hidden_size, num_layers=3, batch_first=True, bidirectional=True, dropout=(0.33,0.33))
        self.activation = nn.LeakyReLU(0.1)
        self.mlp_dropout = nn.Dropout2d(p=0.33)

        self.init_biaffine()
        self.init_weights()

    def init_biaffine(self):
        self.arc_h = nn.Linear(self.config.hidden_size*2, self.arc_mlp_dim)
        self.arc_c = nn.Linear(self.config.hidden_size*2, self.arc_mlp_dim)
        self.arc_attention = BiaffineAttention(self.arc_mlp_dim, bias_x=True, bias_y=False)

        self.rel_h = nn.Linear(self.config.hidden_size*2, self.rel_mlp_dim)
        self.rel_c = nn.Linear(self.config.hidden_size*2, self.rel_mlp_dim)
        self.rel_attention = BiaffineAttention(self.rel_mlp_dim, n_out=self.num_labels, bias_x=True, bias_y=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.arc_h.weight, a=0.1, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.arc_c.weight, a=0.1, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.rel_h.weight, a=0.1, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.rel_c.weight, a=0.1, nonlinearity='leaky_relu')

        nn.init.constant_(self.arc_h.bias, 0.)
        nn.init.constant_(self.arc_c.bias, 0.)
        nn.init.constant_(self.rel_h.bias, 0.)
        nn.init.constant_(self.rel_c.bias, 0.)


    def _arc_mlp(self, hidden):
        # output size [batch, length, arc_mlp_dim]
        arc_h = self.activation(self.arc_h(hidden))
        arc_c = self.activation(self.arc_c(hidden))

        # apply dropout on arc
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        arc = self.mlp_dropout(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        return arc_h, arc_c

    def _rel_mlp(self, hidden):
        # output size [batch, length, rel_mlp_dim]
        rel_h = self.activation(self.rel_h(hidden))
        rel_c = self.activation(self.rel_c(hidden))

        # apply dropout on rel
        # [batch, length, dim] --> [batch, 2 * length, dim]
        rel = torch.cat([rel_h, rel_c], dim=1)
        rel = self.mlp_dropout(rel.transpose(1, 2)).transpose(1, 2)
        rel_h, rel_c = rel.chunk(2, 1)
        rel_h = rel_h.contiguous()
        rel_c = rel_c.contiguous()

        return rel_h, rel_c


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        first_ids=None,
        word_mask=None,
        heads=None,
        rels=None
    ):
        batch_size, seq_len = first_ids.size()
        # (batch, seq_len), seq mask, where at position 0 is 0
        root_mask = torch.arange(seq_len, device=input_ids.device).gt(0).float().unsqueeze(0) * word_mask
        # (batch, seq_len, seq_len)
        mask_3D = (root_mask.unsqueeze(-1) * word_mask.unsqueeze(1))

        return_dict = None

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        size = list(first_ids.size()) + [sequence_output.size()[-1]]
        # (batch, seq_len, hidden_size)
        output = sequence_output.gather(1, first_ids.unsqueeze(-1).expand(size))
        #print ("first_ids:", first_ids)
        #print ("output:", output.size())
        encoder_output, _ = self.input_encoder(output, word_mask)
        output = self.mlp_dropout(encoder_output.transpose(1, 2)).transpose(1, 2)

        # (batch, seq_len, arc_mlp_dim)
        arc_h, arc_c = self._arc_mlp(encoder_output)
        # (batch, seq_len, seq_len)
        arc_logits = self.arc_attention(arc_c, arc_h)
        # mask invalid position to -inf for log_softmax
        if mask_3D is not None:
            minus_mask = mask_3D.eq(0)
            arc_logits = arc_logits.masked_fill(minus_mask, float('-inf'))
        #print ("arc_logits:\n", arc_logits)
        arc_logits = torch.sigmoid(arc_logits)

        # (batch, length, rel_mlp_dim)
        rel_h, rel_c = self._rel_mlp(encoder_output)
        # (batch, n_rels, seq_len, seq_len)
        rel_logits = self.rel_attention(rel_c, rel_h)

        # (batch, seq_len, seq_len, n_rels)
        transposed_rel_logits = rel_logits.permute(0, 2, 3, 1)
        
        #print ("transposed_rel_logits:\n", transposed_rel_logits)

        loss = None
        if heads is not None and rels is not None:
            arc_loss_fct = nn.BCELoss(reduction='none')
            arc_loss = arc_loss_fct(arc_logits, heads.float())
            # mask invalid position to 0 for sum loss
            if mask_3D is not None:
                arc_loss = arc_loss * mask_3D
            # [batch, length - 1] -> [batch] remove the symbolic root
            arc_loss = arc_loss[:, 1:].sum(dim=1)

            rel_locc_fct = nn.CrossEntropyLoss(reduction='none')
            rel_loss = rel_locc_fct(rel_logits, rels)
            if mask_3D is not None:
                rel_loss = rel_loss * mask_3D
            rel_loss = rel_loss * heads
            rel_loss = rel_loss[:, 1:].sum(dim=1)

            loss = 0.95 * arc_loss.mean() + 0.05 * rel_loss.mean()

        output = (arc_logits, transposed_rel_logits) + outputs[2:]
        return ((loss,) + output) if loss is not None else output