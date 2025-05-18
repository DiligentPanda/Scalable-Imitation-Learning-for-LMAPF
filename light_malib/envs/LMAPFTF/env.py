from ..base_env import BaseEnv
import torch
import torch.nn.functional as F
import numpy as np

# This env is just a wrapper of the underlying environment.
class LMPAFTFEnv(BaseEnv):
    def __init__(
        self,
        id,
        seed,  
        cfg
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        super().__init__(id, seed)
        self.cfg=cfg
    
    # accept a policy and return data for training
    def rollout(self, policy, eval=True):
        pass