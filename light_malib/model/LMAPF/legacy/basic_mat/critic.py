import torch.nn as nn
from light_malib.utils.episode import EpisodeKey

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
        
    def forward(self, **kwargs):
        
        global_observations=kwargs.get(EpisodeKey.CUR_GLOBAL_STATE,None)
        states=kwargs.get(EpisodeKey.CUR_STATE,None)
        critic_rnn_states=kwargs.get(EpisodeKey.CRITIC_RNN_STATE,None)
        rnn_masks=kwargs.get(EpisodeKey.DONE,None)
        
        values = states["values"]
        return values, critic_rnn_states