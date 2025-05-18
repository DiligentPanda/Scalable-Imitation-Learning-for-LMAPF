import torch.nn as nn
import torch
import sys
import numpy as np
from light_malib.utils.episode import EpisodeKey

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
                nn.LayerNorm([self.in_dim,self.FOV_height,self.FOV_width]),
                # 11,11 -> 9,9
                nn.Conv2d(self.in_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-2,self.FOV_width-2]),
                # 9,9 -> 7,7
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-4,self.FOV_width-4]),
                # 7,7- > 5,5
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-6,self.FOV_width-6]),
                # 5,5 -> 3,3
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-8,self.FOV_width-8]),
                # 3,3 -> 1,1
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-10,self.FOV_width-10]),
            )
        # FOV 21x21 -> 1x1
        elif self.FOV_height==21:
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
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-20,self.FOV_width-20]),
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
                
        # perm_indices=observations[...,-7].long()
        reverse_perm_indices=observations[...,-6].long()
        observations=observations[...,:-7-A*2]
        
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
        
        self.in_dim=4
        self.hidden_dim=32
        self.FOV_height=11
        self.FOV_width=11
         
        # TODO: residual?
        if self.FOV_height==11:
            self.backbone=nn.Sequential(
                nn.LayerNorm([self.in_dim,self.FOV_height,self.FOV_width]),
                # 11,11 -> 9,9
                nn.Conv2d(self.in_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-2,self.FOV_width-2]),
                # 9,9 -> 7,7
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-4,self.FOV_width-4]),
                # 7,7- > 5,5
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-6,self.FOV_width-6]),
                # 5,5 -> 3,3
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-8,self.FOV_width-8]),
                # 3,3 -> 1,1
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-10,self.FOV_width-10]),
            )
        # FOV 21x21 -> 1x1
        elif self.FOV_height==21:
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
                nn.ReLU(),
                nn.LayerNorm([self.hidden_dim,self.FOV_height-20,self.FOV_width-20]),
            )
        else:
            raise NotImplementedError
        
        self.out=nn.Linear(self.hidden_dim,5)
        
        self.rnn_layer_num=1
        self.rnn_state_size=1
        
        # self.num_robots=200
        # self.map_size=[32,32]
        # self.action_choices=np.array([[0,1],[1,0],[0,-1],[-1,0],[0,0]],dtype=np.int32)
        # self.action_choices=self.action_choices.reshape(-1).tolist()
        
        # sys.path.insert(0,"lmapf_lib/MAPFCompetition2023/build")
            
    def forward(self, **kwargs):

        global_observations=kwargs.get(EpisodeKey.CUR_GLOBAL_OBS,None)
        observations=kwargs.get(EpisodeKey.CUR_OBS,None)
        actor_rnn_states=kwargs.get(EpisodeKey.ACTOR_RNN_STATE,None)
        rnn_masks=kwargs.get(EpisodeKey.DONE,None)
        action_masks=kwargs.get(EpisodeKey.ACTION_MASK,None)
        actions=kwargs.get(EpisodeKey.ACTION,None)
        explore=kwargs.get("explore")

        # target_positions=observations[...,-2:].long()
        # curr_positions=observations[...,-4:-2].long()
        # priorities=observations[...,-5]
        
        B=global_observations.shape[0]
        A=observations.shape[0]//B
                
        # perm_indices=observations[...,-7].long()
        reverse_perm_indices=observations[...,-6].long()
        observations=observations[...,:-7-A*2]
        
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
        # B*A,5
        logits=self.out(h)
        # logits=logits.reshape(-1,5)
        
        illegal_action_mask = 1-action_masks
        logits=logits-1e10*illegal_action_mask
        
        # NOTE the action outside is permuted
        # logits=logits[reverse_perm_indices]
        
        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            # print(actions.shape)
            
            # # load PIBT solver here
            # import py_PIBT
            # seed=0
            # # TODO: we need to pickle this 
            # PIBTSolver=py_PIBT.PIBTSolver(seed)
            
            # # TODO: load PIBT here, and sample from it
            # probs=torch.softmax(logits,dim=-1)
            # probs=probs.reshape(batch_size,-1)
            # probs=probs.detach().cpu().numpy()
            
            # priorities=priorities.reshape(batch_size,-1)
            # priorities=priorities.detach().cpu().numpy()
            
            # locations=curr_positions.reshape(batch_size,-1)
            # locations=locations.detach().cpu().numpy()
            
            # actions=[]
            # for i in range(batch_size):
            #     _probs=probs[i].tolist()
            #     _priorities=priorities[i].tolist()
            #     _locations=locations[i].tolist()
            #     # TODO: non-explore mode
            #     _actions=PIBTSolver.solve(_priorities,_locations,_probs,self.action_choices,self.map_size,True)
            #     actions.append(_actions)
                
            # actions=torch.tensor(actions).to(logits.device).reshape(-1)
            
            # print(actions.shape)
            
            dist_entropy = None
        else:
            dist_entropy = dist.entropy()
        
        action_log_probs = dist.log_prob(actions)
        
        actions=actions[reverse_perm_indices]
        action_log_probs=action_log_probs[reverse_perm_indices]
        if dist_entropy is not None:
            dist_entropy=dist_entropy[reverse_perm_indices]
        logits=logits[reverse_perm_indices]
        
        return actions, actor_rnn_states, action_log_probs, dist_entropy, logits
