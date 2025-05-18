import cma
import numpy as np
from light_malib.algorithm.mappo.policy import MAPPO
import torch.nn as nn
import torch

# ? do we need to explicitly add current params into samples?
class Optimizer:
    def __init__(self, policy, sigma=0.5, n_samples=32):
        self.policy = policy
        self.sigma = sigma
        self.n_samples = n_samples
        params = self._get_curr_params()
        self.es = cma.CMAEvolutionStrategy(params, self.sigma)
        
    def _get_curr_params(self):
        # this is implementation dependable! 
        last_layer: nn.Linear = self.policy.actor.out
        
        weight = last_layer.weight.detach().cpu().numpy()
        bias = last_layer.bias.data.detach().cpu().numpy()

        params=np.concatenate([weight.flatten(),bias.flatten()])
        return params
    
    def get_samples(self):
        samples = self.es.ask(self.n_samples)
        return samples
    
    def update(self, samples, unfitness):
        self.es.tell(samples, unfitness)
        self.es.logger.add()
        self.es.disp()       
        
    def get_best_sample(self):
        params = self.es.result.xbest 
        f = self.es.result.fbest
        return params, f
    
    @staticmethod
    def set_policy_params(policy, params):
        
        last_layer: nn.Linear = policy.actor.out

        w_shape = last_layer.weight.data.shape
        b_shape = last_layer.bias.data.shape
        
        w_n_params = last_layer.weight.data.numel()
        b_n_params = last_layer.bias.data.numel()
        
        w_data = params[:w_n_params]
        b_data = params[-b_n_params:]
        
        w_data = torch.from_numpy(w_data).to(device=last_layer.weight.data.device, dtype=last_layer.bias.data.dtype)
        b_data =torch.from_numpy(b_data).to(device=last_layer.bias.data.device, dtype=last_layer.bias.data.dtype)
        
        w_data = w_data.reshape(w_shape)
        b_data = b_data.reshape(b_shape)
        
        last_layer.weight.data = w_data
        last_layer.bias.data = b_data