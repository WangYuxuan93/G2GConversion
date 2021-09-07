# coding=utf-8
import torch
from torch import nn

from neuronlp2.models.gate import HighwayGateLayer

import logging
logger = logging.getLogger(__name__)


class GCNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_self_weight = config.graph["use_self_weight"]
        self.use_rel_embedding = config.graph["use_rel_embedding"]
        self.use_data_flow_gate = config.graph["use_data_flow_gate"]
        self.data_flow = config.graph["data_flow"]
        input_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size
        output_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size

        self.adj_weight = nn.Linear(input_size, output_size, bias=False)
        if self.use_self_weight:
            self.self_weight = nn.Linear(input_size, output_size, bias=False)
        if self.use_rel_embedding:
            self.rel_weight = nn.Linear(config.graph["rel_embed_size"], output_size, bias=False)
        if self.data_flow == "bidir":
            self.reverse_adj_weight = nn.Linear(input_size, output_size, bias=False)
            if self.use_data_flow_gate:
                self.adj_gate = eval(config.graph["data_flow_gate"])(output_size)

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
        heads: (batch, seq_len, seq_len)
        rels: (batch, seq_len, seq_len, rel_embed_size)
        """
        
        # (batch, seq_len, output_size)
        adj_layer = self.adj_weight(hidden_states)

        # (batch, seq_len, seq_len)
        # use the predicted heads from other parser
        adj_matrix = heads.float()
        # modifier to dependent, this cause multi-heads
        if self.data_flow == "c2h":
            # remask pads at the end of each row because it's permuted
            adj_matrix = adj_matrix.permute(0,2,1)

        if debug:
            torch.set_printoptions(profile="full")
            print ("adj_matrix:\n", adj_matrix)

        # (batch, seq_len, output_size)
        context_layer = torch.matmul(adj_matrix, adj_layer)
        if self.data_flow == "bidir":
            reverse_adj_layer = self.reverse_adj_weight(hidden_states)
            reverse_adj_matrix = adj_matrix.permute(0,2,1)
            reverse_context_layer = torch.matmul(reverse_adj_matrix, reverse_adj_layer)

        # divide by the number of neighbors
        # (batch, seq_len)
        num_neighbors = adj_matrix.sum(-1)
        ones = torch.ones_like(num_neighbors, device=context_layer.device)
        num_neighbors = torch.where(num_neighbors>0,num_neighbors,ones)
        if debug:
            print ("num_neighbors:\n", num_neighbors)
        # divide by the number of neighbors
        context_layer = context_layer / num_neighbors.unsqueeze(-1)
        if self.data_flow == "bidir":
            num_neighbors = reverse_adj_matrix.sum(-1)
            ones = torch.ones_like(num_neighbors, device=context_layer.device)
            num_neighbors = torch.where(num_neighbors>0,num_neighbors,ones)
            # divide by the number of neighbors
            reverse_context_layer = reverse_context_layer / num_neighbors.unsqueeze(-1)
            if self.use_data_flow_gate:
                context_layer = self.adj_gate(context_layer, reverse_context_layer)
            else:
                context_layer = context_layer + reverse_context_layer

        if self.use_self_weight:
            self_layer = self.self_weight(hidden_states)
            context_layer += self_layer

        if self.use_rel_embedding:
            # (batch ,seq_len, seq_len, rel_embed_size) => (batch ,seq_len, seq_len, output_size)
            rel_matrix = self.rel_weight(rels)
            context_layer += rel_matrix.sum(-2)

        # context_layer = [self_layer] + adj_layer + [rev_adj_layer] + [rel_layer]
        return context_layer


class RGCNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_basic_matrix = config.graph["num_basic_matrix"]
        self.num_rel_labels = config.graph["num_rel_labels"]
        self.data_flow = config.graph["data_flow"]
        self.input_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size
        self.output_size = config.graph["lowrank_size"] if config.graph["do_pal_project"] else config.hidden_size

        self.num_all_rel_labels = self.num_rel_labels * 2 if self.data_flow == "bidir" else self.num_rel_labels

        self.basic_matrix = nn.Parameter(torch.Tensor(self.num_basic_matrix,
                                                self.input_size,
                                                self.output_size))
        self.rel_weight = nn.Parameter(torch.Tensor(self.num_all_rel_labels,
                                                self.num_basic_matrix))
        self.self_weight = nn.Linear(self.input_size, self.output_size, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        #nn.init.zeros_(self.weight)
        nn.init.xavier_uniform_(self.rel_weight)
        nn.init.xavier_uniform_(self.basic_matrix)

    def forward(
            self, 
            hidden_states,
            attention_mask=None,
            heads=None,
            rels=None,
            debug=False
        ):
        """
        hidden_states: (batch, seq_len, input_size)
        rels: (batch, seq_len, seq_len)

        matmul(x, w)
        x: [batch_size, 1, seq_len, h] => [batch_size, n_out, seq_len, h], stack n_out times
        w: [n_out, h, h]) => [batch_size, n_out, h, h], stack batch_size times
        output: [batch_size, n_out, seq_len, h]
        """
        batch_size, seq_len, _ = list(hidden_states.size())
        device = hidden_states.device

        # preprocess
        if self.data_flow == "c2h":
            rels = rels.permute(0,2,1)
        rel_mask = rels.gt(0).long()
        rel_tensor = torch.zeros(batch_size, self.num_all_rel_labels, seq_len, seq_len, dtype=torch.long, device=rels.device)
        # (batch, num_label, seq_len, seq_len)
        rel_tensor.scatter_(1, rels.unsqueeze(1), 1)
        rel_tensor = rel_tensor * rel_mask.unsqueeze(1)

        if self.data_flow == "bidir":
            zeros = torch.zeros_like(rels, dtype=torch.long, device=rels.device)
            reverse_rels = rels.permute(0,2,1) + self.num_rel_labels
            reverse_rels = torch.where(reverse_rels>self.num_rel_labels, reverse_rels, zeros)
            if debug:
                torch.set_printoptions(profile="full")
                print ("rels:\n", rels)
                print ("reverse_rels:\n", reverse_rels)
            reverse_rel_mask = reverse_rels.gt(0).long()
            reverse_rel_tensor = torch.zeros(batch_size, self.num_all_rel_labels, seq_len, seq_len, dtype=torch.long, device=rels.device)
            # (batch, num_label, seq_len, seq_len)
            reverse_rel_tensor.scatter_(1, reverse_rels.unsqueeze(1), 1)
            reverse_rel_tensor = reverse_rel_tensor * reverse_rel_mask.unsqueeze(1)
            rel_tensor += reverse_rel_tensor

        # (num_label, num_basic, input_size, output_size)
        weight = self.rel_weight.unsqueeze(-1).unsqueeze(-1).expand([-1,-1,self.input_size,self.output_size])
        # (num_label, num_basic, input_size, output_size)
        rel_matrix = self.basic_matrix.unsqueeze(0) * weight
        # (num_label, input_size, output_size)
        rel_matrix = rel_matrix.sum(1)
        # (batch, 1, seq_len, input_size) x (1, num_label, input_size, output_size)
        # => (batch, num_label, seq_len, output_size)
        r_hidden_states = torch.matmul(hidden_states.unsqueeze(1), rel_matrix.unsqueeze(0))

        # (batch, num_label, seq_len, seq_len) x (batch, num_label, seq_len, output_size)
        # (batch, num_label, seq_len, output_size)
        r_context_layer = torch.matmul(rel_tensor.float(), r_hidden_states)

        # (batch, num_label, seq_len)
        num_neighbors = rel_tensor.sum(-1)
        ones = torch.ones_like(num_neighbors, device=r_context_layer.device)
        num_neighbors = torch.where(num_neighbors>0,num_neighbors,ones)
        if debug:
            print ("num_neighbors:\n", num_neighbors)
        # divide by the number of neighbors
        # (batch, num_label, seq_len, output_size)
        r_context_layer = r_context_layer / num_neighbors.unsqueeze(-1)
        # (batch, seq_len, output_size)
        context_layer = r_context_layer.sum(1)

        self_layer = self.self_weight(hidden_states)
        context_layer = self_layer + context_layer

        return context_layer


