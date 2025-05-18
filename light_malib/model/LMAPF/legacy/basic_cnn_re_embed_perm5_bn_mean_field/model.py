import torch.nn as nn
import torch
import sys
import numpy as np
from light_malib.utils.episode import EpisodeKey
from .re_embed import ReEmbed
from .mean_field import MeanField

def fold(tensor,num_agents):
    if tensor is None:
        return None
    assert tensor.shape[0]%num_agents==0
    B=tensor.shape[0]//num_agents
    tensor=tensor.reshape(B,num_agents,*tensor.shape[1:])
    return tensor
        
def unfold(tensor,num_agents):
    if tensor is None:
        return None
    assert tensor.shape[1]==num_agents
    tensor=tensor.reshape(-1,*tensor.shape[2:])
    return tensor


class Critic(nn.Module):
    def __init__(self, model_config, observation_space, action_space, custom_config, initialization):
        super().__init__()
    
        self.in_dim=4
        self.hidden_dim=32
        self.FOV_height=11
        self.FOV_width=11
        
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
        # B*A,h
        h = h.reshape(h.shape[0],-1)
        # B*A,1
        values = self.out(h)
        
        values = values[reverse_perm_indices]
        
        return values, critic_rnn_states
    
    
class ReEmbedBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.re_embed=ReEmbed()
        
        self.ln = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
        )
        
        self.proj=nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
        )
        
    def forward(self, global_feats, curr_positions, local_feats):
        ctx_feats = self.re_embed(global_feats, curr_positions, local_feats)
        local_feats = ctx_feats + self.proj(local_feats)
        local_feats = self.ln(local_feats) 
        return local_feats

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
        
        self.num_re_embed_blocks=model_config["num_re_embed_blocks"]
        
        self.in_dim=4
        self.hidden_dim=32
        self.FOV_height=11
        self.FOV_width=11
        self.global_in_dim=3
         
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

        
        if self.num_re_embed_blocks!=0:     
            self.global_backbone=nn.Sequential(
                nn.BatchNorm2d(self.global_in_dim),
                nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=1,padding=0),
            )     
        
        self.comm=nn.ModuleList()
        for i in range(self.num_re_embed_blocks):
            re_embed_block=ReEmbedBlock(self.hidden_dim)
            self.comm.append(re_embed_block)
        
        self.out=nn.Linear(self.hidden_dim,5)
        
        self.mean_field=MeanField(iterations=10)
        
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

        # target_positions=observations[...,-2:].long()
        curr_positions=observations[...,-4:-2].long()
        # priorities=observations[...,-5]
        
        B=global_observations.shape[0]
        A=observations.shape[0]//B
                
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
        
        if self.num_re_embed_blocks!=0:
            # B, H, W
            gh = self.global_backbone(global_observations)
        
        for i in range(self.num_re_embed_blocks):
            h = self.comm[i](gh, curr_positions, h)
           
        # B*A,5
        logits=self.out(h)
        
        illegal_action_mask = 1-action_masks
        # logits=logits-1e10*illegal_action_mask
        
        logits=logits.reshape(B,A,-1)
        illegal_action_mask=illegal_action_mask.reshape(B,A,-1)
        curr_positions=curr_positions.reshape(B,A,-1)

        if actions is None:
            logits=self.mean_field(logits, illegal_action_mask, curr_positions)
            logits=logits.reshape(B*A,-1)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = None
        else:
            actions = actions[perm_indices]
            logits, training_logits=self.mean_field.forward_training(logits, illegal_action_mask, curr_positions, actions.reshape(B,A))
            logits = logits.reshape(B*A,-1)
            dist = torch.distributions.Categorical(logits=logits)
            dist_entropy = dist.entropy()
            logits = training_logits.reshape(B,1).repeat(1,A).reshape(B*A,1)
            
        action_log_probs = dist.log_prob(actions)
        
        actions=actions[reverse_perm_indices]
        action_log_probs=action_log_probs[reverse_perm_indices]
        if dist_entropy is not None:
            dist_entropy=dist_entropy[reverse_perm_indices]
        logits=logits[reverse_perm_indices]
        
        return actions, actor_rnn_states, action_log_probs, dist_entropy, logits
