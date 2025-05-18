import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from .utils.util import check, init
import torch_geometric.utils as gu
from light_malib.utils.logger import Logger

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SparseSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SparseSelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
        #                      .view(1, 1, n_agent + 1, n_agent + 1))

        # self.att_bp = None

    def forward(self, key, value, query, atten_masks):
        '''
        The n_heads implementation is wrong
        '''
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (B*L, nh, hs)
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).reshape(-1, self.n_head, D // self.n_head)  
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).reshape(-1, self.n_head, D // self.n_head) 
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).reshape(-1, self.n_head, D // self.n_head) 


        # atten_masks: B, L, L
        assert len(atten_masks.shape)==3 and atten_masks.shape[0]==B and atten_masks.shape[1]==L and atten_masks.shape[2]==L, "{} B: {} L: {}".format(atten_masks.shape,B,L)
    
        ret= torch.nonzero(atten_masks, as_tuple=True)        
        batch_idx, heads, tails=ret

        offsets=batch_idx*L
        # E
        heads=offsets+heads
        tails=offsets+tails

        # E, nh, hs
        qs=q[heads]        
        ks=k[tails]
        vs=v[tails]

        # E, nh
        weights=(ks*qs).sum(dim=-1)
        
        # E, nh
        attens=gu.softmax(weights,heads)
        # E, nh, 1
        attens=attens.unsqueeze(-1)

        # B*L, nh, hs
        y = gu.scatter(attens*vs, heads, dim=0, dim_size=B*L, reduce="sum")
        # B, L, D
        y = y.reshape(B, L, D)

        # output projection
        y = self.proj(y)
        return y