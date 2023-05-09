import torch
from torch import nn

import math


class QNetwork(nn.Module):
    def __init__(self, hidden_size, n_node):
        super(QNetwork, self).__init__()
        self.linear_layer_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_layer_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_layer_3 = nn.Linear(hidden_size, n_node, bias=True)

    def forward(self, hidden):
        hidden = self.linear_layer_1(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.linear_layer_2(hidden)
        hidden = torch.tanh(hidden)
        q_value = self.linear_layer_3(hidden)

        return q_value


class Ggnn(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(Ggnn, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size

        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def gnn_cell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = nn.functional.linear(inputs, self.w_ih, self.b_ih)
        gh = nn.functional.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = new_gate + input_gate * (hidden - new_gate)
        return hy

    def forward(self, alias_inputs: torch.Tensor, A: torch.Tensor, items: torch.Tensor):
        hidden = self.embedding(items)
        for i in range(self.step):
            hidden = self.gnn_cell(A, hidden)

        get_f = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get_f(i) for i in torch.arange(len(alias_inputs)).long()])

        return seq_hidden


class SoftAttetion(nn.Module):
    def __init__(self, hidden_size):
        super(SoftAttetion, self).__init__()
        self.hidden_size = hidden_size

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # batch_size x seq_length x 1
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        return a


class MbGrl(nn.Module):
    def __init__(self, opt, n_node):
        super(MbGrl, self).__init__()
        self.hidden_size = opt.hidden_size
        self.embedding = nn.Embedding(n_node + 1, opt.hidden_size)
        self.gnn = Ggnn(hidden_size=opt.hidden_size, step=opt.step)
        self.soft_atten_srr = SoftAttetion(hidden_size=opt.hidden_size)
        self.soft_atten_bfrl = SoftAttetion(hidden_size=opt.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_item_embedding(self, alias_inputs:torch.Tensor,adj_matr:torch.Tensor, items:torch.Tensor,):
        item_embedding = self.embedding(items)
        for i in range(self.gnn.step):
            hidden = self.gnn.gnn_cell(adj_matr, item_embedding)

        get_f = lambda i: hidden[i][alias_inputs[i]]
        seq_hidden = torch.stack([get_f(i) for i in torch.arange(len(alias_inputs)).long()])

        return seq_hidden

    def get_session_represent(self, hidden:torch.Tensor, mask:torch.Tensor):
        return self.soft_atten_srr(hidden, mask)

    def get_state_represent(self, hidden:torch.Tensor, mask:torch.Tensor):
        return self.soft_atten_bfrl(hidden, mask)

    def forward(self, session_represent:torch.Tensor):
        item_embedding = self.embedding.weight[1:]
        score = torch.matmul(session_represent, item_embedding.transpose(1, 0))
        return score
