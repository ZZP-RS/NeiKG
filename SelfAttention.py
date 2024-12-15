import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Attention_Layer(nn.Module):
    def __init__(self,in_dim):
        super(Attention_Layer, self).__init__()
        self.in_dim = in_dim

        self.Q_linear = nn.Linear(in_dim, in_dim, bias=False)
        self.K_linear = nn.Linear(in_dim, in_dim, bias=False)
        self.V_linear = nn.Linear(in_dim, in_dim, bias=False)

        self.norm_fact = 1 / sqrt(self.in_dim)


    def forward(self,inputs):
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs)
        V = self.V_linear(inputs)

        att = torch.matmul(Q, K.transpose(1, 0)) * self.norm_fact
        att = torch.nn.functional.softmax(att, dim=-1)

        outputs = torch.matmul(att, V)

        return outputs