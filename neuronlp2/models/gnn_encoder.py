# coding=utf-8
import math
import torch
from torch import nn

#from transformers.modeling_bert import (BertOutput, BertIntermediate, BertSelfOutput)
#from transformers.models.bert.modeling_bert import (BertOutput, BertIntermediate, BertSelfOutput)
from neuronlp2.models.modeling_bert import (BertOutput, BertIntermediate, BertSelfOutput)
from neuronlp2.models.gate import HighwayGateLayer
from neuronlp2.models.graph_convolution import GCNLayer, RGCNLayer
from neuronlp2.models.graph_attention import GATLayer


class GNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_encoder = config.graph["encoder"]
        self.fusion_type = config.fusion_type
        self.use_rel_embedding = self.graph_encoder != "RGCN" and config.graph["use_rel_embedding"]
        self.data_flow = config.graph["data_flow"]
        self.num_layers = config.graph["num_layers"]

        self.layer = nn.ModuleList([GNNEncoderLayer(config) for i in range(self.num_layers)])

        if self.use_rel_embedding:
            if self.graph_encoder == "GAT":
                # should have same size as one attention head
                in_out_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size
                config.graph["rel_embed_size"] = int(in_out_size / config.graph["num_attention_heads"])
            self.num_rel_labels = config.graph["num_rel_labels"]
            self.num_all_rel_labels = self.num_rel_labels * 2 if self.data_flow == "bidir" else self.num_rel_labels
            self.rel_embeddings = nn.Embedding(self.num_all_rel_labels, config.graph["rel_embed_size"], padding_idx=0)
            self.rel_dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        heads=None,
        rels=None,
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                heads,
                rels
            )

        return hidden_states


class GNNEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.do_pal_project = config.graph["do_pal_project"]
        self.encoder_type = config.graph["encoder"]
        self.use_output_layer = "use_output_layer" in config.graph and config.graph["use_output_layer"]
        self.use_ff_layer = "use_ff_layer" in config.graph and config.graph["use_ff_layer"]

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
        else:
            print ("###### Invalid GNN encoder_type: {} ######".format(self.encoder_type))
            exit()

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
        
        if self.do_pal_project:
            attention_hiddens = self.dense_up(attention_hiddens)

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
