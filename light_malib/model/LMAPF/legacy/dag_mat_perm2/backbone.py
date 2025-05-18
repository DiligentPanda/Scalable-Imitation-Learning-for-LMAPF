import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from .ma_transformer import Encoder
from light_malib.utils.episode import EpisodeKey

class Backbone(nn.Module):
    def __init__(
        self,
        model_config,
        global_observation_space,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        
        # assert isinstance(global_observation_space, Box) and len(global_observation_space.shape)==1,global_observation_space
        # assert isinstance(observation_space,Box) and len(observation_space.shape)==1,observation_space
        
        # self.global_observation_space=global_observation_space
        # self.observation_space=observation_space
        # self.state_dim=global_observation_space.shape[0]
        # self.obs_dim=observation_space.shape[0]
                
        self.in_dim=4
        self.hidden_dim=32
        self.FOV_height=11
        self.FOV_width=11
        
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
        
        self.state_dim=self.hidden_dim
        self.obs_dim=self.hidden_dim
        
        self.num_agents=model_config["num_agents"]
        self.embed_dim=model_config["embed_dim"]
        self.num_blocks=model_config["num_blocks"]
        self.num_heads=model_config["num_heads"]
        self.encode_state=model_config["encode_state"]
        
        # TODO: only maintain encoder here
        self.encoder=Encoder(
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
            n_block=self.num_blocks,
            n_embd=self.embed_dim,
            n_head=self.num_heads,
            n_agent=self.num_agents,
            encode_state=self.encode_state
        )
    
    def forward(
            self,           
            **kwargs     
        ):
        '''
        backbone should return observations, but it could be a data structure containing anything.
        '''
        
        # CNN
        original_observations=kwargs.get(EpisodeKey.CUR_OBS,None)
        
        A = self.num_agents
        B = original_observations.shape[0] // A
        
        observations=original_observations[...,:-7-self.num_agents*2]
        obs_neighboring_masks=original_observations[...,-7-self.num_agents*2:-7-self.num_agents]
        # act_neighboring_masks=original_observations[...,-7-self.num_agents:-7]
        # perm_indices=original_observations[...,-7].long()
        # reverse_perm_indices=original_observations[...,-6].long()
        
        batch_indices=torch.arange(B,dtype=torch.long,device=observations.device)*A
        batch_indices=batch_indices.reshape(B,1).repeat(1,A).reshape(-1)
        
        # perm_indices+=batch_indices
        # reverse_perm_indices+=batch_indices
        
        # permute
        # observations=observations[perm_indices]
        # NOTE: the columns are already permuted in the environment
        # obs_neighboring_masks=obs_neighboring_masks[perm_indices]
        
        # B*A,F -> B*A,C,H,W
        observations=observations.view(-1,self.in_dim,self.FOV_height,self.FOV_width)
        # B*A,h,1,1
        h = self.backbone(observations)
        observations = h.reshape(h.shape[0],-1)
        states = observations
        
        # Transformer
        assert len(observations)%self.num_agents==0
        batch_size=len(observations)//self.num_agents
        
        states=states.reshape(batch_size,self.num_agents,-1)
        observations=observations.reshape(batch_size,self.num_agents,-1)
        obs_neighboring_masks=obs_neighboring_masks.reshape(batch_size,self.num_agents,self.num_agents)
        
        obs_rep=self.encoder(states, observations, obs_neighboring_masks)
        
        #values=values.reshape(batch_size*self.num_agents,-1)
        obs_rep=obs_rep.reshape(batch_size*self.num_agents,-1)
        # observations=observations.reshape(batch_size*self.num_agents,-1)
        
        return {"obs_rep": obs_rep, "obs": original_observations}
        
        