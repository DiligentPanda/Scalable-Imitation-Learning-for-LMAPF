import torch.nn as nn
from light_malib.utils.episode import EpisodeKey
import torch

from .utils.util import check, init
from .re_embed import ReEmbed

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Critic(nn.Module):
    def __init__(
        self,
        model_config,
        action_space,
        custom_config,
        initialization,
        backbone
    ):
        super().__init__()
        
        self.in_dim=4
        self.hidden_dim=32
        self.FOV_height=11
        self.FOV_width=11
        
        self.global_in_dim=3
        self.map_height=32
        self.map_width=32
        
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
        )
        
        
        # self.global_backbone=nn.Sequential(
        #     nn.LayerNorm([self.global_in_dim,self.map_height,self.map_width]),
        #     nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=3,padding=1),
        #     nn.ReLU(),
        #     nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
        #     nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=2,dilation=2),
        #     nn.ReLU(),
        #     nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
        #     nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=4,dilation=4),
        #     nn.ReLU(),
        #     nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
        #     nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=8,dilation=8),
        #     nn.ReLU(),
        #     nn.LayerNorm(self.hidden_dim)
        # )     
        
        # self.re_embed=ReEmbed()
        
        self.final_ln = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        
        self.head = nn.Linear(self.hidden_dim, 1)
        
        self.rnn_layer_num=1
        self.rnn_state_size=1
        
    def forward(self, **kwargs):
        
        critic_rnn_states=kwargs.get(EpisodeKey.CRITIC_RNN_STATE,None)
        global_observations=kwargs.get(EpisodeKey.CUR_GLOBAL_STATE,None)
        observations=kwargs.get(EpisodeKey.CUR_STATE,None)
        actor_rnn_states=kwargs.get(EpisodeKey.ACTOR_RNN_STATE,None)
        rnn_masks=kwargs.get(EpisodeKey.DONE,None)
        action_masks=kwargs.get(EpisodeKey.ACTION_MASK,None)
        actions=kwargs.get(EpisodeKey.ACTION,None)
        explore=kwargs.get("explore")
        
        B = global_observations.shape[0]
        A = observations.shape[0]//B
        
        # obs_neighboring_masks=observations[...,-7-A*2:-7-A]
        # act_neighboring_masks=original_observations[...,-7-A:-7]
        # perm_indices=observations[...,-7].long()
        reverse_perm_indices=observations[...,-6].long()
        
        # priorities=observations[...,-5]
        curr_positions=observations[...,-4:-2]
        
        batch_indices=torch.arange(B,dtype=torch.long,device=observations.device)*A
        batch_indices=batch_indices.reshape(B,1).repeat(1,A).reshape(-1)
        
        observations=observations[...,:-7-A*2]
        
        #perm_indices+=batch_indices
        reverse_perm_indices+=batch_indices
        
        # permute
        # observations=observations[perm_indices]
        # NOTE: the columns are already permuted in the environment
        # obs_neighboring_masks=obs_neighboring_masks[perm_indices]
        
        # B*A,F -> B*A,C,H,W
        observations=observations.view(-1,self.in_dim,self.FOV_height,self.FOV_width)
        # B*A,h,1,1
        h = self.backbone(observations)
        # B*A, -1
        observations = h.reshape(h.shape[0],-1)
        
        
        # B, H, W
        # global_observations = self.global_backbone(global_observations)
        
        # # B*A, -1
        # ctx_observations = self.re_embed(global_observations, curr_positions, observations)
        
        # observations = observations + ctx_observations
        
        observations = self.final_ln(observations)        
        
        # states = observations
        
        # Transformer
        # assert len(observations)%A==0
        # batch_size=len(observations)//A
        
        # states=states.reshape(batch_size,A,-1)
        # observations=observations.reshape(batch_size,A,-1)
        # obs_neighboring_masks=obs_neighboring_masks.reshape(batch_size,A,A)
        
        #obs_rep=self.encoder(states, observations, obs_neighboring_masks)
        
        values=self.head(observations)
        
        # permute back
        values=values[reverse_perm_indices]
        
        return values, critic_rnn_states