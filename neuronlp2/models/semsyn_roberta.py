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

from transformers import PretrainedConfig
from neuronlp2.models.modeling_bert import BertPooler
from neuronlp2.models.modeling_roberta import RobertaPreTrainedModel, RobertaEmbeddings
from neuronlp2.models.semsyn_bert import SemSynBertEncoder
from transformers.activations import ACT2FN
from .modeling_bert import BiaffineAttention
from neuronlp2.nn import VarFastLSTM

import logging
logger = logging.getLogger(__name__)


class SemSynRobertaConfig(PretrainedConfig):
    model_type = "roberta"

    def __init__(
        self,
        vocab_size=50265,
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
        layer_norm_eps=1e-05,
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


class SemSynRobertaModel(RobertaPreTrainedModel):
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

        self.embeddings = RobertaEmbeddings(config)
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


class SemSynRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = SemSynRobertaModel(config)
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


class SemSynRobertaForMultipleChoice(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = SemSynRobertaModel(config)
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


class SemSynRobertaForQuestionAnswering(RobertaPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = SemSynRobertaModel(config, add_pooling_layer=False)
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



class SemSynRobertaForArgumentLabel(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = SemSynRobertaModel(config, add_pooling_layer=False)
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


class SemSynRobertaForPredicateSense(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = SemSynRobertaModel(config, add_pooling_layer=False)
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


class SemSynRobertaForSDP(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.arc_mlp_dim = 512
        self.rel_mlp_dim = 128

        self.roberta = SemSynRobertaModel(config, add_pooling_layer=False)
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
        rels=None,
        src_heads=None,
        src_rels=None,
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
            heads=src_heads,
            rels=src_rels,
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