import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from .utils.util import check, init
import torch_geometric.utils as gu
import torch_geometric.nn as gnn
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
    
class DAGSparseSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super().__init__()
        
        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_agent = n_agent
        
        self.obs_proj = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd))
        
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(2*n_embd, n_embd))
        self.value = init_(nn.Linear(2*n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = nn.Sequential(
            nn.LayerNorm(n_embd),
            init_(nn.Linear(2*n_embd, n_embd), activate=True), 
            nn.GELU(), 
            nn.LayerNorm(n_embd)
        )
        
        # onlys upport batch size 1 now.
        self.cache_k = torch.zeros((1, n_agent, self.n_head, self.n_embd))
        self.cache_v = torch.zeros((1, n_agent, self.n_head, self.n_embd))
    
    def forward(self, observations, actions, atten_masks, start_pos=-1):
        '''
        observations: B, L, D
        actions: B, L, D
        atten_masks: B, L, L 
        
        
        '''
        # pure observations will be used as query
        
        # will be used for key and value
        aug_observations=torch.cat([observations, actions], dim=-1)
        
        B, L, D = observations.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (B*L, nh, hs)
        k = self.key(aug_observations).view(B, L, self.n_head, D // self.n_head) 
        q = self.query(observations).view(B, L, self.n_head, D // self.n_head)
        v = self.value(aug_observations).view(B, L, self.n_head, D // self.n_head)

        if start_pos>=0:
            # use cache
            if B>self.cache_k.shape[0]:
                self.cache_k = torch.zeros((1, self.n_agent, self.n_head, self.n_embd))
                self.cache_v = torch.zeros((1, self.n_agent, self.n_head, self.n_embd))
                
            if self.cache_k.device!=q.device:
                self.cache_k = self.cache_k.to(q.device)
                self.cache_v = self.cache_v.to(q.device)
                
            self.cache_k[:B, start_pos: start_pos+L] = k
            self.cache_v[:B, start_pos: start_pos+L] = v
        
            k = self.cache_k[:B, :start_pos+L]
            v = self.cache_v[:B, :start_pos+L]
        
        k = k.transpose(1, 2)  # (B, nh, L1, hs)
        q = q.transpose(1, 2)  # (B, nh, L0, hs)
        v = v.transpose(1, 2)  # (B, nh, L1, hs)

        L0 = q.shape[2]
        L1 = k.shape[2]

        # atten_masks: B, L0, L1
        assert len(atten_masks.shape)==3 and atten_masks.shape[0]==B and atten_masks.shape[1]==L0 and atten_masks.shape[2]==L1, "{} B: {} L0: {} L1: {}".format(atten_masks.shape, B, L0, L1)   
        
        # !!! attention_masks should not contain self-attention
        idxs1=torch.arange(L0,dtype=torch.int32,device=q.device)
        idxs2=idxs1+start_pos
        atten_masks[:,idxs1,idxs2]=0
        
        ret= torch.nonzero(atten_masks, as_tuple=True)        


        # Logger.error("ret: {}".format(ret))
        
        
        # import time
        # time.sleep(10)
        
        batch_idx, heads, tails=ret

        if len(batch_idx)!=0:

            offsets=batch_idx*L1
            # E
            heads=offsets+heads+start_pos
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
            y = y.reshape(B, L0, D)
        else:
            y = torch.zeros((B, L0, D), device=q.device, dtype=q.dtype)

        # since there is self-attention, we need to add observation directly
        observations=self.obs_proj(observations)
        y = torch.cat([y, observations],dim=-1)

        # output projection
        y = self.proj(y)
        return y
