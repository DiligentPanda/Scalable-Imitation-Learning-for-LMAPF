import torch.nn as nn
import torch
import sys
import numpy as np
from light_malib.utils.episode import EpisodeKey
from .re_embed import ReEmbed
from .transformer import EncodeBlock
from .positional_encoding import get_2d_sincos_pos_embed
from light_malib.utils.logger import Logger

def get_edge_feats(curr_positions, FoV_h, FoV_w, max_atten_dist):
    '''
    curr_positions: B, A, 2
    '''
    h=FoV_h//2
    w=FoV_w//2
    
    B,A,_=curr_positions.shape
    
    # 2
    diffs=(curr_positions[:,:,None,:]-curr_positions[:,None,:,:]).float()
    # B,A,A,1
    l_inf_dists,_=diffs.abs().max(dim=-1,keepdim=True)
    # B,A,A
    atten_masks=l_inf_dists.squeeze(-1)<=max_atten_dist
    # K
    batch_indices, head_indices, tail_indices=torch.nonzero(atten_masks, as_tuple=True)    

    l_1_dists=torch.norm(diffs,p=1,dim=-1,keepdim=True)/(h+w)
    l_2_dists=torch.norm(diffs,p=2,dim=-1,keepdim=True)/((h**2+w**2)**0.5)
    # B,A,A,5
    edge_feats=torch.concat([diffs,l_inf_dists,l_1_dists,l_2_dists],dim=-1)
    
    # K,5
    edge_feats=edge_feats[batch_indices,head_indices,tail_indices]
    
    return batch_indices,head_indices,tail_indices,edge_feats,atten_masks

class Critic(nn.Module):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        super().__init__()
    
        self.in_dim=4
        self.hidden_dim=32
        self.FOV_height=11
        self.FOV_width=11
        
        self.num_encode_blocks=model_config["num_encode_blocks"]
        
        self.backbone=nn.Sequential(
            nn.BatchNorm2d(self.in_dim),
            # 11,11 -> 9,9
            nn.Conv2d(self.in_dim,self.hidden_dim,3),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            # 9,9 -> 7,7
            nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            # 7,7- > 5,5
            nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            # 5,5 -> 3,3
            nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            # 3,3 -> 1,1
            nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
        )
        
        self.comm=nn.ModuleList()
        for i in range(self.num_encode_blocks):
            encode_block = EncodeBlock(n_embd=self.hidden_dim,n_head=8)
            self.comm.append(encode_block)
        
        self.out=nn.Linear(self.hidden_dim,1)
        
        self.rnn_layer_num=1
        self.rnn_state_size=1
        
    def forward(self, **kwargs):
        global_observations=kwargs.get(EpisodeKey.CUR_GLOBAL_STATE,None)
        observations=kwargs.get(EpisodeKey.CUR_STATE,None)
        critic_rnn_states=kwargs.get(EpisodeKey.CRITIC_RNN_STATE,None)
        rnn_masks=kwargs.get(EpisodeKey.DONE,None)
        
        assert len(observations.shape)==2
        
        B=global_observations.shape[0]
        A=observations.shape[0]//B
                
        # target_positions=observations[...,-2:].long()
        curr_positions=observations[...,-4:-2].long()
        # priorities=observations[...,-5]
        # perm_indices=observations[...,-7].long()
        reverse_perm_indices=observations[...,-6].long()
        observations=observations[...,:-7]
        
        batch_indices=torch.arange(B,dtype=torch.long,device=observations.device)*A
        batch_indices=batch_indices.reshape(B,1).repeat(1,A).reshape(-1)
        
        # perm_indices+=batch_indices
        reverse_perm_indices+=batch_indices
        
        # B*A,F -> B*A,C,H,W
        observations=observations.view(-1,self.in_dim,self.FOV_height,self.FOV_width)
        # B*A,h,1,1
        h = self.backbone(observations)
        # # B*A,h,5 -> B*A,5,h
        # h=h[...,[0,1,1,1,2],[1,0,1,2,1]].permute(0,2,1).contiguous()
        # B*A,h
        h = h.reshape(h.shape[0],-1)
           
        # B*A,1
        values = self.out(h)
        
        values = values[reverse_perm_indices]
        
        return values, critic_rnn_states
    
class Actor(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization         
        ) -> None:
        super().__init__()
        
        self.num_encode_blocks=model_config["num_encode_blocks"]
        
        self.in_dim=4
        self.hidden_dim=32
        self.FOV_height=11
        self.FOV_width=11
        self.num_heads=8
         
        self.backbone=nn.Sequential(
            nn.BatchNorm2d(self.in_dim),
            # 11,11 -> 9,9
            nn.Conv2d(self.in_dim,self.hidden_dim,3),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            # 9,9 -> 7,7
            nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            # 7,7- > 5,5
            nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            # 5,5 -> 3,3
            nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            # 3,3 -> 1,1
            nn.Conv2d(self.hidden_dim,self.hidden_dim,3)
        )
        
        self.max_atten_dist=model_config["max_atten_dist"]
        
        self.comm=nn.ModuleList()
        for i in range(self.num_encode_blocks):
            encode_block = EncodeBlock(n_embd=self.hidden_dim,n_head=8)
            self.comm.append(encode_block)
        
        self.out=nn.Linear(self.hidden_dim,5)
        
        self.rnn_layer_num=1
        self.rnn_state_size=1
            
    def forward(self, **kwargs):

        global_observations=kwargs.get(EpisodeKey.CUR_GLOBAL_OBS,None)
        observations=kwargs.get(EpisodeKey.CUR_OBS,None)
        actor_rnn_states=kwargs.get(EpisodeKey.ACTOR_RNN_STATE,None)
        rnn_masks=kwargs.get(EpisodeKey.DONE,None)
        action_masks=kwargs.get(EpisodeKey.ACTION_MASK,None)
        actions=kwargs.get(EpisodeKey.ACTION,None)
        explore=kwargs.get("explore")
        
        B=global_observations.shape[0]
        A=observations.shape[0]//B
        
        # target_positions=observations[...,-2:].long()
        curr_positions=observations[...,-4:-2].long()
        # priorities=observations[...,-5]
        perm_indices=observations[...,-7].long()
        reverse_perm_indices=observations[...,-6].long()
        observations=observations[...,:-7]
        
        batch_indices=torch.arange(B,dtype=torch.long,device=observations.device)*A
        batch_indices=batch_indices.reshape(B,1).repeat(1,A).reshape(-1)
        
        perm_indices+=batch_indices
        reverse_perm_indices+=batch_indices
        
        # B*A,F -> B*A,C,H,W
        observations=observations.view(-1,self.in_dim,self.FOV_height,self.FOV_width)
        # B*A,h,1,1
        h = self.backbone(observations)
        # # B*A,h,5 -> B*A,5,h
        # h=h[...,[0,1,1,1,2],[1,0,1,2,1]].permute(0,2,1).contiguous()
        # B*A,h
        h = h.reshape(h.shape[0],-1)
            
        # B*A
        batch_indices,head_indices,tail_indices,edge_feats,atten_masks = get_edge_feats(curr_positions.reshape(B,A,2), self.FOV_height, self.FOV_width, self.max_atten_dist)

        
        h = h.reshape(B,A,-1)
        for i in range(self.num_encode_blocks):
            h = self.comm[i](h, batch_indices, head_indices, tail_indices, edge_feats, atten_masks)
        h = h.reshape(B*A,-1)
           
        # B*A,5
        logits=self.out(h)
        
        illegal_action_mask = 1-action_masks
        logits=logits-1e10*illegal_action_mask
        
        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)            
            dist_entropy = None
        else:
            actions = actions[perm_indices]
            dist_entropy = dist.entropy()
        
        action_log_probs = dist.log_prob(actions)
        
        actions=actions[reverse_perm_indices]
        action_log_probs=action_log_probs[reverse_perm_indices]
        if dist_entropy is not None:
            dist_entropy=dist_entropy[reverse_perm_indices]
        logits=logits[reverse_perm_indices]
        
        return actions, actor_rnn_states, action_log_probs, dist_entropy, logits
