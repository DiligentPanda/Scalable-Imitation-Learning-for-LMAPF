import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, use_edge_feats=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        
        self.use_edge_feats=use_edge_feats
        edge_feats_dim=5
        if use_edge_feats:
            self.edge_att_proj = nn.Sequential(
                init_(nn.Linear(edge_feats_dim, n_embd)),
                nn.ReLU(),
                nn.LayerNorm(normalized_shape=(n_embd)),
                init_(nn.Linear(n_embd, n_head)),
            )
            # TODO: 
            # self.edge_feats_proj = nn.Sequential(
            #     init_(nn.Linear(edge_feats_dim, n_embd)),
            #     nn.ReLU(),
            #     nn.LayerNorm(normalized_shape=(n_embd)),
            #     init_(nn.Linear(n_embd, n_embd)),
            # )

    
    def forward(self, key, value, query, atten_masks, edge_feats=None):
        B, L, D = query.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head) 
        

        k = k.transpose(1, 2)  # (B, nh, L1, hs)
        q = q.transpose(1, 2)  # (B, nh, L0, hs)
        v = v.transpose(1, 2)  # (B, nh, L1, hs)
        
        L0 = q.shape[2]
        L1 = k.shape[2]

        # causal attention: (B, nh, L0, hs) x (B, nh, hs, L1) -> (B, nh, L0, L1)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)
 
        # atten_masks: B, L0, L1
        assert len(atten_masks.shape)==3 and atten_masks.shape[0]==B and atten_masks.shape[1]==L0 and atten_masks.shape[2]==L1, "{} B: {} L0: {} L1: {}".format(atten_masks.shape, B, L0, L1)
        atten_masks = atten_masks.reshape(B,1,L0,L1).repeat(1,self.n_head,1,1)
        
        if self.use_edge_feats:
            # B,A,A,nh
            edge_att = self.edge_att_proj(edge_feats)
            # B,nh,A,A
            edge_att = edge_att.permute(0,3,1,2)
            att+=edge_att
        
        att = att.masked_fill(atten_masks==0, float('-inf'))

        # if self.masked:
        #     att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        # B, L0, L1
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L0, L1) x (B, nh, L1, hs) -> (B, nh, L0, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, L0, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head, use_edge_feats):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, use_edge_feats)
        self.use_edge_feats = use_edge_feats
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, atten_masks, edge_feats=None):
        x = self.ln1(x + self.attn(x, x, x, atten_masks, edge_feats))
        x = self.ln2(x + self.mlp(x))
        return x