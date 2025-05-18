import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric.utils as gu

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
    def __init__(self, n_embd, n_head):
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
        
    
        edge_feats_dim=5
        self.edge_att_proj = nn.Sequential(
            init_(nn.Linear(edge_feats_dim, n_embd)),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=(n_embd)),
            init_(nn.Linear(n_embd, n_head)),
        )
        # TODO: 
        self.edge_key_proj = nn.Sequential(
            init_(nn.Linear(edge_feats_dim, n_embd)),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=(n_embd)),
            init_(nn.Linear(n_embd, n_embd)),
        )
        self.edge_val_proj = nn.Sequential(
            init_(nn.Linear(edge_feats_dim, n_embd)),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=(n_embd)),
            init_(nn.Linear(n_embd, n_embd)),
        )
    
    def forward(self, key, value, query, batch_indices, head_indices, tail_indices, edge_feats, atten_masks):
        B, L, D = query.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).reshape(-1, self.n_head, D // self.n_head)  
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).reshape(-1, self.n_head, D // self.n_head)  
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).reshape(-1, self.n_head, D // self.n_head)  

        # E
        heads = batch_indices*L+head_indices
        # E
        tails = batch_indices*L+tail_indices

        # E, nh, hs
        qs=q[heads]        
        ks=k[tails]
        vs=v[tails]
        
        edge_ks=self.edge_key_proj(edge_feats).reshape(-1, self.n_head, D // self.n_head)  
        edge_vs=self.edge_val_proj(edge_feats).reshape(-1, self.n_head, D // self.n_head)  
        
        ks+=edge_ks
        vs+=edge_vs
        
        # E, nh
        weights=(ks*qs).sum(dim=-1)
        
        # E, nh
        edge_weights=self.edge_att_proj(edge_feats)
        weights+=edge_weights
        
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

class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, n_embd, n_head):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, batch_indices, head_indices, tail_indices, edge_feats, atten_masks):
        x = self.ln1(x + self.attn(x, x, x, batch_indices, head_indices, tail_indices, edge_feats, atten_masks))
        x = self.ln2(x + self.mlp(x))
        return x