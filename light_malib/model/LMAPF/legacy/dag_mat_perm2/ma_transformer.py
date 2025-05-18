import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from .utils.util import check, init, self_atten_embed, self_atten_split
from .utils.transformer_act import discrete_autoregreesive_act
from .utils.transformer_act import discrete_parallel_act
from .utils.transformer_act import continuous_autoregreesive_act
from .utils.transformer_act import continuous_parallel_act
from light_malib.utils.logger import Logger

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_agent = n_agent
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None
        
        # onlys upport batch size 1 now.
        self.cache_k = torch.zeros((1, n_agent, self.n_head, self.n_embd))
        self.cache_v = torch.zeros((1, n_agent, self.n_head, self.n_embd))

    def forward(self, key, value, query, atten_masks, start_pos=-1):
        B, L, D = query.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head) 
        
        if start_pos>=0:
            # use cache
            if B>self.cache_k.shape[0]:
                self.cache_k = torch.zeros((1, self.n_agent, self.n_head, self.n_embd))
                self.cache_v = torch.zeros((1, self.n_agent, self.n_head, self.n_embd))
                
            if self.cache_k.device!=query.device:
                self.cache_k = self.cache_k.to(query.device)
                self.cache_v = self.cache_v.to(query.device)
                
            self.cache_k[:B, start_pos: start_pos+L] = k
            self.cache_v[:B, start_pos: start_pos+L] = v
        
            k = self.cache_k[:B, :start_pos+L]
            v = self.cache_v[:B, :start_pos+L]
        
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

class DAGSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super().__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_agent = n_agent
        
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(2*n_embd, n_embd))
        self.value = init_(nn.Linear(2*n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
   
        self.value_self = init_(nn.Linear(n_embd, n_embd))
        self.key_self = init_(nn.Linear(n_embd, n_embd))
   
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
        #                      .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None
        
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
        
        B, L0, D = observations.size()
        if start_pos>=0:
            L1 = L0 + start_pos
        else:
            L1 = L0
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(aug_observations).view(B, L0, self.n_head, D // self.n_head)
        v = self.value(aug_observations).view(B, L0, self.n_head, D // self.n_head) 
        q = self.query(observations).view(B, L0, self.n_head, D // self.n_head)
        
        self_k = self.key_self(observations).view(B, L0, self.n_head, D // self.n_head)
        self_v = self.value_self(observations).view(B, L0, self.n_head, D // self.n_head)
        
        if start_pos>=0:
            
            if B>self.cache_k.shape[0]:
                self.cache_k = torch.zeros((B, self.n_agent, self.n_head, self.n_embd))
                self.cache_v = torch.zeros((B, self.n_agent, self.n_head, self.n_embd))
            
            if self.cache_k.device!=observations.device:
                self.cache_k = self.cache_k.to(observations.device)
                self.cache_v = self.cache_v.to(observations.device)
            
            # use cache
            k = self.cache_k[:B, :start_pos+L0]
            v = self.cache_v[:B, :start_pos+L0]
        
        k = k.transpose(1, 2)  # (B, nh, L1, hs)
        q = q.transpose(1, 2)  # (B, nh, L0, hs)
        v = v.transpose(1, 2)  # (B, nh, L1, hs)
        
        self_k = self_k.transpose(1, 2) # (B, nh, L0, hs)
        self_v = self_v.transpose(1, 2) # (B, nh, L0, hs)
        
        # (B, nh, L0)
        self_att_weights = (q * self_k).sum(dim=-1)
        # causal attention: (B, nh, L0, hs) x (B, nh, hs, L1) -> (B, nh, L0, L1)
        att_weights = q @ k.transpose(-2, -1)

        idxs1=torch.arange(L0,dtype=torch.int32,device=q.device)
        if start_pos>=0:
            idxs2=idxs1+start_pos
        else:
            idxs2=idxs1
            
        att_weights = self_atten_embed(att_weights, self_att_weights, idxs1, idxs2)

        # causal attention: (B, nh, L0, hs) x (B, nh, hs, L1) -> (B, nh, L0, L1)
        att = att_weights * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)
 
        # atten_masks: B, L0, L1
        assert len(atten_masks.shape)==3 and atten_masks.shape[0]==B and atten_masks.shape[1]==L0 and atten_masks.shape[2]==L1, "{} B: {} L0: {} L1: {}".format(atten_masks.shape, B, L0, L1)
        
        atten_masks = atten_masks.reshape(B,1,L0,L1).repeat(1,self.n_head,1,1)
        # B, H, L0
        # atten_masks_masks=(atten_masks.sum(dim=-1)==0)
        # atten_masks[atten_masks_masks] = 1
        att = att.masked_fill(atten_masks==0, float('-inf'))

        # if self.masked:
        #     att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        
        # TODO: 0 self-attention might be a source of unstable, we explicit add something, e.g. self/global features?
        # e.g. replace the diagonal of atten with dot product of self obs
        # B, H, L0, L1
        att = F.softmax(att, dim=-1)
        
        att, self_att = self_atten_split(att, idxs1, idxs2)

        y = att @ v  # (B, nh, L0, L1) x (B, nh, L1, hs) -> (B, nh, L0, hs)
        
        # (B, nh, L0, 1) * (B, nh, L0, hs) -> (B, nh, L0, hs)
        y_self = self_att.unsqueeze(-1) * self_v
        y += y_self
        
        y = y.transpose(1, 2).contiguous().view(B, L0, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y
    
    def update_cache(self, observations, actions, start_pos):
        assert start_pos>=0
        
        B, L0, D = observations.size()
        
        # will be used for key and value
        aug_observations=torch.cat([observations, actions], dim=-1)
        
        if B>self.cache_k.shape[0]:
            self.cache_k = torch.zeros((B, self.n_agent, self.n_head, self.n_embd))
            self.cache_v = torch.zeros((B, self.n_agent, self.n_head, self.n_embd))
            
        if self.cache_k.device!=observations.device:
            self.cache_k = self.cache_k.to(observations.device)
            self.cache_v = self.cache_v.to(observations.device)
            
        k = self.key(aug_observations).view(B, L0, self.n_head, D // self.n_head)
        v = self.value(aug_observations).view(B, L0, self.n_head, D // self.n_head) 
            
        self.cache_k[:B, start_pos: start_pos+L0] = k
        self.cache_v[:B, start_pos: start_pos+L0] = v

    
class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent,masked=False):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=masked)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, atten_masks):
        x = self.ln1(x + self.attn(x, x, x, atten_masks))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent,masked=True):
        super(DecodeBlock, self).__init__()
        self.attn = DAGSelfAttention(n_embd, n_head, n_agent, masked)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )
        
    def forward(self, x, rep_enc, atten_masks, start_pos=-1):
        # TODO(rivers): we can proj rep_env first
        x = self.ln1(rep_enc + self.attn(rep_enc, x, atten_masks, start_pos))
        x = self.ln2(x + self.mlp(x))
        return x
    
    def update_cache(self, x, rep_enc, start_pos):
        self.attn.update_cache(rep_enc, x, start_pos)
        
# class DecodeBlock(nn.Module):
#     """ an unassuming Transformer block """

#     def __init__(self, n_embd, n_head, n_agent,masked=True):
#         super(DecodeBlock, self).__init__()

#         self.ln1 = nn.LayerNorm(n_embd)
#         self.ln2 = nn.LayerNorm(n_embd)
#         self.ln3 = nn.LayerNorm(n_embd)
#         self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=masked)
#         self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=masked)
#         self.mlp = nn.Sequential(
#             init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
#             nn.GELU(),
#             init_(nn.Linear(1 * n_embd, n_embd))
#         )

#     def forward(self, x, rep_enc, atten_masks, start_pos=-1):
#         # TODO(rivers): we should think about this design more
#         x = self.ln1(x + self.attn1(x, x, x, atten_masks, start_pos))
#         # TODO(rivers): why not merge x and rep_enc here?
#         x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc, atten_masks=atten_masks, start_pos=start_pos))
#         x = self.ln3(x + self.mlp(x))
#         return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        if self.encode_state:
            self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                            init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        else:
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                            init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])

    def forward(self, state, obs, atten_masks):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        rep = self.ln(x)
        for block in self.blocks:
            rep = block(rep, atten_masks)

        return rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.ModuleList([DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self):
        if self.action_type != 'Discrete':
            nn.init.zeros_(self.log_std)

    # state, action, and return
    def forward(self, action, obs_rep, obs, atten_masks, start_pos=-1):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep, atten_masks, start_pos)
            logit = self.head(x)

        return logit


    def update_cache(self, action, obs_rep, obs, atten_masks, start_pos):
        assert start_pos>=0
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)
        assert len(self.blocks)==1, 'only support 1 block now'
        for block in self.blocks:
            block.update_cache(x, obs_rep, start_pos)

# class MultiAgentTransformer(nn.Module):

#     def __init__(self, state_dim, obs_dim, action_dim, n_agent,
#                  n_block, n_embd, n_head, encode_state=False,
#                  action_type='Discrete', dec_actor=False, share_actor=False):
#         super(MultiAgentTransformer, self).__init__()

#         self.n_agent = n_agent
#         self.action_dim = action_dim
#         self.action_type = action_type

#         # state unused
#         state_dim = 37

#         self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
#         self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
#                                self.action_type, dec_actor=dec_actor, share_actor=share_actor)

#     def zero_std(self):
#         if self.action_type != 'Discrete':
#             self.decoder.zero_std()

#     def forward(self, state, obs, action, available_actions=None):
#         # state: (batch*n_agent, state_dim)
#         # obs: (batch*n_agent, obs_dim)
#         # action: (batch*n_agent,1)
#         # available_actions: (batch*n_agent, act_dim)

#         assert obs.shape[0]%self.n_agent==0
#         batch_size = obs.shape[0]//self.n_agent
        
#         v_loc, obs_rep = self.encoder(state, obs)
#         if self.action_type == 'Discrete':
#             action = action.long()
#             action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
#                                                         self.n_agent, self.action_dim, available_actions)
#         else:
#             action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
#                                                           self.n_agent, self.action_dim)

#         return action_log, v_loc, entropy

#     def get_actions(self, state, obs, available_actions=None, deterministic=False):
#         assert obs.shape[0]%self.n_agent==0
#         batch_size = obs.shape[0]//self.n_agent
        
#         v_loc, obs_rep = self.encoder(state, obs)
#         if self.action_type == "Discrete":
#             output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
#                                                                            self.n_agent, self.action_dim,
#                                                                            available_actions, deterministic)
#         else:
#             output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
#                                                                              self.n_agent, self.action_dim,
#                                                                              deterministic)

#         return output_action, output_action_log, v_loc

#     def get_values(self, state, obs, available_actions=None):
#         v_tot, obs_rep = self.encoder(state, obs)
#         return v_tot
