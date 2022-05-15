# coding=utf-8
import math
import torch
from torch import nn

#from transformers.modeling_bert import (BertOutput, BertIntermediate, BertSelfOutput)
#from transformers.models.bert.modeling_bert import (BertOutput, BertIntermediate, BertSelfOutput)
from neuronlp2.models.modeling_bert import (BertOutput, BertIntermediate, BertSelfOutput)
from neuronlp2.models.gate import HighwayGateLayer

def convert_batched_graph_to_masks(heads, n_mask=3, mask_types=["parent","child"]):
    """
    heads: (batch, seq_len, seq_len)
    """
    parent_masks = []
    child_masks = []
    if "parent" in mask_types:
        origin_parents = heads
        parents = torch.ones_like(origin_parents)
    if "child" in mask_types:
        origin_childs = heads.permute(0,2,1)
        childs = torch.ones_like(origin_childs)
    for dist in range(n_mask):
        if "parent" in mask_types:
            parents = torch.matmul(parents, origin_parents)
            parent_masks.append(parents.unsqueeze(1))
        if "child" in mask_types:
            childs = torch.matmul(childs, origin_childs)
            child_masks.append(childs.unsqueeze(1))
    
    # len(mask_types) * n_mask * [(batch, 1, seq_len, seq_len)]
    masks = parent_masks + child_masks
    # (batch, len(mask_types) * n_mask, seq_len, seq_len)
    attention_mask = torch.cat(masks, dim=1)
    return attention_mask


def convert_graph_to_masks(heads, n_mask=3, mask_types=["parent","child"]):
    """
    heads: (seq_len, seq_len)
    """
    parent_masks = []
    child_masks = []
    if "parent" in mask_types:
        origin_parents = heads
        parents = torch.ones_like(origin_parents)
    if "child" in mask_types:
        origin_childs = heads.permute(1,0)
        childs = torch.ones_like(origin_childs)
    for dist in range(n_mask):
        if "parent" in mask_types:
            parents = torch.matmul(parents, origin_parents)
            parent_masks.append(parents)
        if "child" in mask_types:
            childs = torch.matmul(childs, origin_childs)
            child_masks.append(childs)
    
    # len(mask_types) * n_mask * [(seq_len, seq_len)]
    masks = parent_masks + child_masks
    # (len(mask_types) * n_mask, seq_len, seq_len)
    attention_mask = torch.stack(masks, dim=0)
    return attention_mask


class GraphMaskAttention(nn.Module):
    def __init__(self, config):
        super(GraphMaskAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # task
        self.use_task_aggregation = config.graph["use_task_aggregation"]
        if self.use_task_aggregation:
            self.query_task = nn.Linear(self.all_head_size, 1)
            self.key_task = nn.Linear(self.all_head_size, self.all_head_size)
            self.value_task = nn.Linear(self.all_head_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) # shape(batch_size, seq_length, num_att, att_size)
        return x.permute(0, 2, 1, 3) # shape(batch_size, num_att, seq_length, att_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
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

        # shape(batch_size, num_att, seq_length, att_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape(batch_size, num_att, seq_length, seq_length)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            #print ("attention_scores=", attention_scores.size())
            #print ("attention_mask=", attention_mask.size())
            # Apply syntax masks
            attention_scores = torch.unsqueeze(attention_scores, dim=2) + torch.unsqueeze(attention_mask, dim=1)
                        # batch_size x num_att x 1 x seq_length x seq_length + batch_size x 1 x nmask x seq_length x seq_length
                        # batch_size x heads_num x nmask x seq_length x seq_length

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, torch.unsqueeze(value_layer, dim=2))
                            # attention_probs: batch_size x heads_num x nmask x seq_length x seq_length
                            # value_layer: batch_size x heads_num x seq_length x per_head_size
                            # context_layer: batch_size x heads_num x nmask x seq_length x per_head_size

        if self.use_task_aggregation:
            # context_layer: batch_size x seq_length x nmask x heads_num x per_head_size
            context_layer = context_layer.permute(0, 3, 2, 1, 4).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            # context_layer: batch_size x seq_length x nmask x all_head_size
            
            # key_task value_task: batch_size x seq_length x nmask x all_head_size
            key_task_layer = self.key_task(context_layer)
            value_task_layer = self.value_task(context_layer)

            #attention_task_scores = torch.matmul(key_task_layer, self.query_task.transpose(-1, -2)).squeeze() # batch_size x seq_length x nmask
            attention_task_scores = self.query_task(key_task_layer).squeeze(-1) # batch_size x seq_length x nmask
            attention_task_probs = nn.Softmax(dim=-1)(attention_task_scores) # batch_size x seq_length x nmask
            context_layer = torch.matmul(value_task_layer.transpose(-1, -2), attention_task_probs.unsqueeze(-1))
                        # [batch_size x seq_lentgh x all_head_size x nmask] x [batch_size x seq_length x nmask x 1]
                        #=[batch_size x seq_length x all_head_size x 1]
            context_layer = context_layer.squeeze() # batch_size x seq_length x all_head_size

        else:
            # Follow syntax-BERT, we sum all outputs generated by different syntax masks
            norm_factor = math.sqrt(context_layer.size(2))
            context_layer = torch.sum(context_layer, dim=2)
                            # batch_size x heads_num x seq_length x per_head_size
            context_layer = torch.div(context_layer, norm_factor)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        # batch_size x seq_length x all_head_size
        return outputs


class BertGraphMaskAttention(nn.Module):
    def __init__(self, config):
        super(BertGraphMaskAttention, self).__init__()
        self.self = GraphMaskAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertGraphMaskLayer(nn.Module):
    def __init__(self, config):
        super(BertGraphMaskLayer, self).__init__()
        self.attention = BertGraphMaskAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertGraphMaskAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        head_mask=None, 
        encoder_hidden_states=None, 
        encoder_attention_mask=None,
        output_attentions=False,
        heads=None,
        rels=None,
    ):
        # replace the attention_mask with heads (stores graph masks)
        attention_mask = heads
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertGraphMaskEncoder(nn.Module):
    def __init__(self, config):
        super(BertGraphMaskEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertGraphMaskLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
