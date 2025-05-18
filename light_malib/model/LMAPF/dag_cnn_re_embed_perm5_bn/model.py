import torch.nn as nn
import torch
import sys
import numpy as np
from light_malib.utils.episode import EpisodeKey
from .re_embed import ReEmbed
import torch.nn.functional as F
from light_malib.utils.logger import Logger
import random

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
        
        # TODO: residual?
        if self.FOV_height==11:
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
        # FOV 21x21 -> 1x1
        elif self.FOV_height==21:
            raise NotImplementedError
            self.backbone=nn.Sequential(
                nn.LayerNorm([self.in_dim,self.FOV_height,self.FOV_width]),
                # 21,21 -> 19,19
                nn.Conv2d(self.in_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-2,self.FOV_width-2]),
                # 19,19 -> 9,9
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3,stride=2),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-12,self.FOV_width-12]),
                # 9,9- > 7,7
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-14,self.FOV_width-14]),
                # 7,7 -> 5,5
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-16,self.FOV_width-16]),
                # 5,5 -> 3,3
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-18,self.FOV_width-18]),
                # 3,3 -> 1,1
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            )
        else:
            raise NotImplementedError
        
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
                
        perm_indices=observations[...,-7].long()
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

class FoVEmbedBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
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
        
    def forward(self, local_features):
        '''
        local_features: B(*A), Feature, FOV_h, FOV_w
        '''
        return self.backbone(local_features)
    
class ReEmbedBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.re_embed=ReEmbed()
        
        # self.ln = nn.Sequential(
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.hidden_dim),
        # )
        
        # self.proj=nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.hidden_dim),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(self.hidden_dim),
        # )
        
    def forward(self, global_feats, curr_positions, local_feats):
        ctx_feats = self.re_embed(global_feats, curr_positions, local_feats)
        return ctx_feats
        # local_feats = ctx_feats + self.proj(local_feats)
        # local_feats = self.ln(local_feats) 
        # return local_feats

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
        self.map_height=32
        self.map_width=32
        
        self.action_dim=5
         
        # TODO: residual?
        if self.FOV_height==11:
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
        # FOV 21x21 -> 1x1
        elif self.FOV_height==21:
            raise NotImplementedError
            self.backbone=nn.Sequential(
                nn.LayerNorm([self.in_dim,self.FOV_height,self.FOV_width]),
                # 21,21 -> 19,19
                nn.Conv2d(self.in_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-2,self.FOV_width-2]),
                # 19,19 -> 9,9
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3,stride=2),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-12,self.FOV_width-12]),
                # 9,9- > 7,7
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-14,self.FOV_width-14]),
                # 7,7 -> 5,5
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-16,self.FOV_width-16]),
                # 5,5 -> 3,3
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-18,self.FOV_width-18]),
                # 3,3 -> 1,1
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            )
        else:
            raise NotImplementedError
        
        if self.num_re_embed_blocks!=0:     
            self.global_backbone=nn.Sequential(
                nn.BatchNorm2d(self.global_in_dim),
                nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=1,padding=0),
                # nn.LayerNorm([self.global_in_dim,self.map_height,self.map_width]),
                # nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=3,padding=1),
                # nn.ReLU(),
                # nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
                # nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=2,dilation=2),
                # nn.ReLU(),
                # nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
                # nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=4,dilation=4),
                # nn.ReLU(),
                # nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
                # nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=8,dilation=8),
                # nn.ReLU(),
                # nn.LayerNorm(self.hidden_dim)
            )     
        
        self.re_embed_block=ReEmbedBlock(self.hidden_dim)
        self.FOV_embed_block=FoVEmbedBlock(self.hidden_dim+self.action_dim+1, self.hidden_dim)
        
        self.proj=nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.ln = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
        )
        
        self.out=nn.Linear(self.hidden_dim,self.action_dim)
        
        self.rnn_layer_num=1
        self.rnn_state_size=1
        
        # self.num_robots=200
        # self.map_size=[32,32]
        # self.action_choices=np.array([[0,1],[1,0],[0,-1],[-1,0],[0,0]],dtype=np.int32)
        # self.action_choices=self.action_choices.reshape(-1).tolist()
        
        # sys.path.insert(0,"lmapf_lib/MAPFCompetition2023/build")
        
        self.sampling_alpha=1.0
        self.min_sampling_alpha=0.0
        self.decay_factor=1.0
        self.decay_step=20000 # 20000
        self.step_ctr=0
        
        self.num_batches=4
            
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
        self.map_height=global_observations.shape[2]
        self.map_width=global_observations.shape[3]
                
        B=global_observations.shape[0]
        A=observations.shape[0]//B
                
        perm_indices=observations[...,-7].long()
        reverse_perm_indices=observations[...,-6].long()
        observations=observations[...,:-7]
        
        batch_indices=torch.arange(B,dtype=torch.long,device=observations.device)*A
        batch_indices=batch_indices.reshape(B,1).repeat(1,A).reshape(-1)
        
        perm_indices+=batch_indices
        reverse_perm_indices+=batch_indices
        
        if actions is not None:
            actions = actions[perm_indices]
        
        
        # B*A,F -> B*A,C,H,W
        observations=observations.view(-1,self.in_dim,self.FOV_height,self.FOV_width)
        # B*A,h,1,1
        h = self.backbone(observations)
        # # B*A,h,5 -> B*A,5,h
        # h=h[...,[0,1,1,1,2],[1,0,1,2,1]].permute(0,2,1).contiguous()
        # B*A,h
        h = h.reshape(h.shape[0],-1)
        
        # B, H, W
        gh = self.global_backbone(global_observations)
        ctx_feats = self.re_embed_block(gh, curr_positions, h)
        
        h = self.proj(h)
           
        
        # reshape
        curr_positions = curr_positions.reshape(B,A,2)
        h = h.reshape(B,A,-1)
        ctx_feats = ctx_feats.reshape(B,A,-1,self.FOV_height,self.FOV_width)
        action_masks = action_masks.reshape(B,A,-1)
        if actions is not None:
            actions = actions.reshape(B,A)
        
        if actions is None:
            actions, action_log_probs, dist_entropy, action_logits = self.seq_act(curr_positions, h, ctx_feats, action_masks, explore)
        else:
            actions, action_log_probs, dist_entropy, action_logits = self.parallel_act(actions, curr_positions, h, ctx_feats, action_masks)
            
        # actions=actions.reshape(B*A,*actions.shape[2:])
        # action_log_probs=action_log_probs.reshape(B*A,*action_log_probs.shape[2:])
        # if dist_entropy is not None:
        #     dist_entropy=dist_entropy.reshape(B*A,*dist_entropy.shape[2:])
        # action_logits=action_logits.reshape(B*A,*action_logits.shape[2:])
        
        actions=actions[reverse_perm_indices]
        action_log_probs=action_log_probs[reverse_perm_indices]
        if dist_entropy is not None:
            dist_entropy=dist_entropy[reverse_perm_indices]
        action_logits=action_logits[reverse_perm_indices]
        
        return actions, actor_rnn_states, action_log_probs, dist_entropy, action_logits
    
    # this will be only used in the sampling
    def seq_act(self, curr_positions, h, ctx_feats, action_masks, explore):
        '''
        actions: [B,A]
        curr_positions: [B,A,2]
        h: [B,A,-1]
        ctx_feats: [B,A,C,Fh,Fw]
        action_masks: [B,A,action_dim]
        '''
        pass
        B,A=curr_positions.shape[:2]
        device=curr_positions.device
        
        actions=torch.zeros((B,A), dtype=curr_positions.dtype, device=device)
        action_log_probs=torch.zeros((B,A), dtype=h.dtype, device=device)
        action_logits=torch.zeros((B,A,self.action_dim), dtype=h.dtype, device=device)
        
        batch_size=A//self.num_batches
        num_batches=(A+batch_size-1)//batch_size        
        action_dependency = ActionDependency(curr_positions, self.FOV_height, self.FOV_width, self.action_dim, self.map_height, self.map_width)
        
        for batch_idx in range(num_batches):
            s_idx=batch_idx*batch_size
            e_idx=min((batch_idx+1)*batch_size,A)
            
            a=torch.arange(s_idx,e_idx,device=device,dtype=torch.int32)
            
            local_feats_a = action_dependency.get(a)
            h_a = h[:,a]
            ctx_feats_a = ctx_feats[:,a]
            
            feats_a = torch.cat([ctx_feats_a, local_feats_a],dim=2)
            
            feats_a = feats_a.reshape(B*len(a),-1,self.FOV_height,self.FOV_width)
            h_ctx_a = self.FOV_embed_block(feats_a)
            h_ctx_a = h_ctx_a.reshape(B*len(a),-1)
            h_a = h_a.reshape(B*len(a),-1)
            h_a = self.ln(h_a + h_ctx_a)
            
            # B*A,5
            logits_a=self.out(h_a)
            
            logits_a=logits_a.reshape(B,len(a),-1)
            
            illegal_action_mask_a = 1-action_masks[:,a]
            logits_a=logits_a-1e10*illegal_action_mask_a
            
            dist_a = torch.distributions.Categorical(logits=logits_a)
            # TODO: shape
            actions_a = dist_a.sample() if explore else dist_a.probs.argmax(dim=-1)
            action_log_probs_a = dist_a.log_prob(actions_a)
            
            
            actions[:,a]=actions_a
            action_log_probs[:,a]=action_log_probs_a
            action_logits[:,a]=logits_a
            
            action_dependency.encode(a, actions_a)
        
        actions=actions.reshape(B*A)
        action_log_probs=action_log_probs.reshape(B*A,-1)
        action_logits=action_logits.reshape(B*A,-1)
        
        return actions, action_log_probs, None, action_logits
    
    # this will be only used in the training
    def parallel_act(self, expert_actions, curr_positions, h, ctx_feats, action_masks):
        '''
        actions: [B,A]
        curr_positions: [B,A,2]
        h: [B,A,-1]
        ctx_feats: [B,A,C,Fh,Fw]
        '''
        device=expert_actions.device
        
        if self.step_ctr==0:
            Logger.warning("sampling rate is set to {}".format(self.sampling_alpha))
        
        self.step_ctr+=1
        if self.step_ctr%self.decay_step==0:
            self.sampling_alpha=max(self.sampling_alpha-self.decay_factor,self.min_sampling_alpha)
            Logger.warning("sampling rate is set to {}".format(self.sampling_alpha))
        
        # TODO: we should use action mask
        
        B,A=expert_actions.shape
        device=expert_actions.device
        
        dist_entropys=[]
        actions=[]
        action_log_probs=[]
        action_logits=[]
        
        # local_feats = torch.zeros((B, A, self.action_dim+1, self.FOV_height,self.FOV_width),dtype=torch.float32,device=device)
        action_dependency = ActionDependency(curr_positions, self.FOV_height, self.FOV_width, self.action_dim, self.map_height, self.map_width)
        batch_size=A//self.num_batches
        num_batches=(A+batch_size-1)//batch_size        
        action_dependency = ActionDependency(curr_positions, self.FOV_height, self.FOV_width, self.action_dim, self.map_height, self.map_width)
        
        for batch_idx in range(num_batches):
            s_idx=batch_idx*batch_size
            e_idx=min((batch_idx+1)*batch_size,A)
            
            a=torch.arange(s_idx,e_idx,device=device,dtype=torch.int32)
            
            local_feats_a = action_dependency.get(a)
            h_a = h[:,a]
            ctx_feats_a = ctx_feats[:,a]
                        
            feats_a = torch.cat([ctx_feats_a, local_feats_a],dim=2)
            feats_a = feats_a.reshape(B*len(a),-1,self.FOV_height,self.FOV_width)
            h_ctx_a = self.FOV_embed_block(feats_a)
            h_ctx_a = h_ctx_a.reshape(B*len(a),-1)
            h_a = h_a.reshape(B*len(a),-1)
            h_a = self.ln(h_a + h_ctx_a)
            
            # B*A,5
            logits_a=self.out(h_a)
            
            logits_a=logits_a.reshape(B,len(a),-1)
            
            illegal_action_mask_a = 1-action_masks[:,a]
            logits_a=logits_a-1e10*illegal_action_mask_a
            
            dist_a = torch.distributions.Categorical(logits=logits_a)
            dist_entropy_a = dist_a.entropy()

            expert_actions_a = expert_actions[:,a]
            predcited_actions_a = dist_a.probs.argmax(dim=-1)
            teach_forcing = 1 if random.random()<=self.sampling_alpha else 0
            actions_a=predcited_actions_a*(1-teach_forcing)+expert_actions_a*teach_forcing
            
            action_log_probs_a = dist_a.log_prob(actions_a)
            
            actions.append(actions_a)
            action_log_probs.append(action_log_probs_a)
            action_logits.append(logits_a)
            dist_entropys.append(dist_entropy_a)
            
            # we should mix here with expert actions?
            action_dependency.encode(a, actions_a.detach())
        
        # local_feats = local_feats.reshape(B, A,self.action_dim+1,self.FOV_height,self.FOV_width)
        
        # Logger.error("{} {}".format(ctx_feats.shape,local_feats.shape))
        # import time
        # time.sleep(10)
        
        # feats = torch.concat([ctx_feats, local_feats],dim=2)
        
        # feats = feats.reshape(B*A,*feats.shape[2:])
        
        # h_ctx = self.FOV_embed_block(feats)
        # h_ctx = h_ctx.reshape(B*A,-1)
        # h = h.reshape(B*A,-1)
        # action_masks = action_masks.reshape(B*A,-1)
        # actions=actions.reshape(B*A,-1)
        
        # h = self.ln(h + h_ctx)
        
        # # B*A,5
        # logits=self.out(h)
        # raise NotImplementedError("we should treat multiagent as an RNN, we should consider counpouding errors, so we should not always use the gt labers in the training. Rather, we should addd noises to the label, e.g. selecting predicted labels with certain probs by teacher forcing.")
        # logits=logits.reshape(-1,5)
        
        actions=torch.concat(actions,dim=1).reshape(B*A)
        action_logits=torch.concat(action_logits,dim=1).reshape(B*A,-1)
        action_log_probs=torch.concat(action_log_probs,dim=1).reshape(B*A)
        dist_entropys=torch.concat(dist_entropys,dim=1).reshape(B*A)
        
        return actions, action_log_probs, dist_entropys, action_logits
    
# this is a feature encoder class
class ActionDependency:
    def __init__(self, curr_positions, FOV_height, FOV_width, action_dim, map_height, map_width):
        '''
        curr_positions: [B,A,2]
        '''
        self.FOV_height=FOV_height
        self.FOV_width=FOV_width
        self.action_dim=action_dim
        self.map_height=map_height
        self.map_width=map_width
        
        self.B,self.A=curr_positions.shape[:2]
        device=curr_positions.device
        
        curr_positions=curr_positions.long()
        offsets_y=torch.arange(-(self.FOV_height//2),(self.FOV_height+1)//2,dtype=torch.int32,device=device)
        offsets_x=torch.arange(-(self.FOV_width//2),(self.FOV_width+1)//2,dtype=torch.int32,device=device)
        local_view_offsets=torch.stack(torch.meshgrid([offsets_y,offsets_x],indexing="ij"),dim=-1)
        padded_graph_offsets=torch.tensor([self.FOV_height//2,self.FOV_width//2],dtype=torch.int32,device=device)
        
        # B, A, FOV_height, FOV_weight, 2
        local_views=curr_positions[:,:,None,None,:]+local_view_offsets
        offsetted_local_views=local_views+padded_graph_offsets
        # B, A, 2
        offsetted_curr_positions=curr_positions+padded_graph_offsets
        
        padded_global_feats = torch.zeros((self.B, self.action_dim+1, self.map_height+self.FOV_height//2*2, self.map_width+self.FOV_width//2*2),dtype=torch.float32,device=device)
              
        batch_indexs=torch.arange(self.B,device=device)  
        batch_indexs_all_agents = batch_indexs.reshape(self.B,1).repeat(1,self.A)
        
        # initialize the buffer
        padded_global_feats[batch_indexs_all_agents,0,offsetted_curr_positions[...,0],offsetted_curr_positions[...,1]]=1
        
        # padded_global_loc_feats = torch.zeros((self.B, 1, self.map_height+self.FOV_height//2*2, self.map_width+self.FOV_width//2*2),dtype=torch.float32,device=device)
        # padded_global_loc_feats[batch_indexs_all_agents, :, offsetted_curr_positions[...,0],offsetted_curr_positions[...,1]]=1
        
        self.curr_positions=curr_positions
        self.padded_global_feats=padded_global_feats
        # self.padded_global_loc_feats=padded_global_loc_feats
        self.offsetted_local_views=offsetted_local_views
        self.offsetted_curr_positions=offsetted_curr_positions
        
        # self.movements=torch.tensor([[0,1],[1,0],[0,-1],[-1,0],[0,0]],device=device,dtype=torch.int32)

    def encode(self, a, actions):
        '''
        a: [mini_B]
        actions: [B,mini_B]
        '''
        device=actions.device
        
        # B, mini_B 
        batch_indexs=torch.arange(self.B,device=a.device) 
        a_b = batch_indexs.reshape(self.B,1).repeat(1,len(a))
        
        # modify curr positions by actions
        # [B,mini_B,2]
        # movements=self.movements[actions]
        # offsetted_next_positions=self.offsetted_curr_positions[:,a]
        # offsetted_next_positions+=movements
        
        # padded_global_loc_feats = torch.zeros((self.B, 1, self.map_height+self.FOV_height//2*2, self.map_width+self.FOV_width//2*2),dtype=torch.float32,device=device)        
        # padded_global_loc_feats[a_b, :, offsetted_next_positions[...,0],offsetted_next_positions[...,1]]=1
        
        # B, mini_B 
        a_y = self.offsetted_curr_positions[:,a,0]
        a_x = self.offsetted_curr_positions[:,a,1]
        
        # NOTE: we cannot see its own action currently
        # update global_feats
        one_hot_actions=F.one_hot(actions.long(), num_classes=self.action_dim+1).float()
        self.padded_global_feats[a_b, :, a_y, a_x] = one_hot_actions
        # self.padded_global_loc_feats=padded_global_loc_feats
        
    def get(self, a):        
        # B, mini_B , FOV_height, FOV_weight, 2
        a_local_view = self.offsetted_local_views[:,a,]
        
        # retrieve local_feats from global_feats
        batch_indexs=torch.arange(self.B,device=a.device)  
        batch_indexs_local_view = batch_indexs.reshape(self.B,1,1,1).repeat(1,len(a),self.FOV_height,self.FOV_width)
        
        local_feats_a = self.padded_global_feats[batch_indexs_local_view, :, a_local_view[...,0], a_local_view[...,1]]
        local_feats_a = local_feats_a.permute(0,1,4,2,3)
        
        # local_loc_feats_a = self.padded_global_loc_feats[batch_indexs_local_view, :, a_local_view[...,0], a_local_view[...,1]]
        # local_loc_feats_a = local_loc_feats_a.permute(0,1,4,2,3)

        # local_feats_a = torch.concat([local_feats_a,local_loc_feats_a],dim=2)

        return local_feats_a