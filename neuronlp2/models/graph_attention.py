# coding=utf-8
import math
import torch
from torch import nn

from neuronlp2.models.gate import HighwayGateLayer

import logging
logger = logging.getLogger(__name__)


class GATLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.num_attention_heads = config.graph["num_attention_heads"]
		self.in_out_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size
		self.use_rel_embedding = config.graph["use_rel_embedding"]

		if self.in_out_size % self.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (self.in_out_size, self.num_attention_heads))

		self.attention_head_size = int(self.in_out_size / self.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(self.in_out_size, self.all_head_size)
		self.key = nn.Linear(self.in_out_size, self.all_head_size)
		self.value = nn.Linear(self.in_out_size, self.all_head_size)

		self.dropout = nn.Dropout(config.graph["attention_probs_dropout_prob"])

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(
		self, 
		hidden_states,
		attention_mask=None,
		heads=None,
		rels=None,
		debug=False,
	):
		"""
		hidden_states: (batch, seq_len, input_size)
		heads: (batch, 1, seq_len, seq_len), valid arc = 0, invalid arc = -10000.0
		rels: (batch, seq_len, seq_len, rel_embed_size)
		"""

		# (batch, seq_len, seq_len, hid_size)
		dep_rel_matrix = rels

		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		# (batch, n_heads, seq_len, seq_len)
		attention_scores = torch.matmul(query_layer,
										key_layer.transpose(-1, -2))

		rel_attention_scores = 0
		if self.use_rel_embedding:
			# query_layer:    (batch, num_heads, seq_len, 1,       hid_size)
			# dep_rel_matrix: (batch, 1,         seq_len, seq_len, hid_size)
			rel_attention_scores = query_layer[:, :, :, None, :] * dep_rel_matrix[:, None, :, :, :]
			rel_attention_scores = torch.sum(rel_attention_scores, -1)

		attention_scores = (attention_scores + rel_attention_scores) / math.sqrt(self.attention_head_size)

		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + heads

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)
		# (batch, num_heads, seq_len, seq_len) x (batch, num_heads, seq_len, hid_size)
		context_layer = torch.matmul(attention_probs,
									 value_layer)

		if self.use_rel_embedding:
			# attention_probs: (batch, num_heads, seq_len, seq_len, 1)
			# dep_rel_matrix:  (batch, 1        , seq_len, seq_len, hid_size)
			val_edge = attention_probs[:, :, :, :, None] * dep_rel_matrix[:, None, :, :, :]
			context_layer = context_layer + torch.sum(val_edge, -2)

		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)

		return context_layer


