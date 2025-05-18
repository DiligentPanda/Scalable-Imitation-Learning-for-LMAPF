from light_malib.utils.logger import Logger
from gym.spaces import Box, Discrete

class FeatureEncoder:
    def __init__(self,**kwargs):
        self.FOV_height=11
        self.FOV_weight=11
        self.num_channels=3
    
    @property
    def global_observation_space(self):
        return self.observation_space

    @property
    def observation_space(self):
        # TODO: this is a full observation of all robots
        return Discrete(self.FOV_height*self.FOV_weight*self.num_channels)
    
    @property
    def action_space(self):
        return Discrete(5)
    
    def encode(self):
        return 
