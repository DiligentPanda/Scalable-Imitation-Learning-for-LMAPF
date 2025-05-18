import torch.nn as nn
from light_malib.utils.episode import EpisodeKey
import torch

from .utils.util import check, init

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
        # TODO(jh): remove. legacy.
        self.rnn_layer_num=1
        self.rnn_state_size=1
        
        self.num_agents=backbone.num_agents
        self.embed_dim=backbone.embed_dim
        
        n_embd = self.embed_dim
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                            init_(nn.Linear(n_embd, 1)))
        
    def forward(self, **kwargs):
        
        global_observations=kwargs.get(EpisodeKey.CUR_GLOBAL_STATE,None)
        states=kwargs.get(EpisodeKey.CUR_STATE,None)
        critic_rnn_states=kwargs.get(EpisodeKey.CRITIC_RNN_STATE,None)
        rnn_masks=kwargs.get(EpisodeKey.DONE,None)
        
        observations = states["obs"]
        
        A = self.num_agents
        B = observations.shape[0] // A
        
        # observations=original_observations[...,:-7-self.num_agents*2]
        # obs_neighboring_masks=original_observations[...,-7-self.num_agents*2:-7-self.num_agents]
        # act_neighboring_masks=original_observations[...,-7-self.num_agents:-7]
        # perm_indices=observations[...,-7].long()
        reverse_perm_indices=observations[...,-6].long()
        
        batch_indices=torch.arange(B,dtype=torch.long,device=observations.device)*A
        batch_indices=batch_indices.reshape(B,1).repeat(1,A).reshape(-1)
        
        # perm_indices+=batch_indices
        reverse_perm_indices+=batch_indices
        
        rep = states["obs_rep"]
        values = self.head(rep)
        
        # permute back
        values=values[reverse_perm_indices]
        
        return values, critic_rnn_states