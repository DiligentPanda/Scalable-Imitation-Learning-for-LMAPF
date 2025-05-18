import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from light_malib.utils.logger import Logger

class MeanField(nn.Module):
    def __init__(self, iterations):
        super().__init__()
        self.iterations=iterations
        self.action_dims=5
    
        self.FoV_h=11
        self.FoV_w=self.FoV_h    
        self.max_dist=self.FoV_h//2
        
        # FoV_h*FoV_w, #actions, #actions
        correlation_params=torch.zeros((self.FoV_h*self.FoV_w,self.action_dims,self.action_dims),dtype=torch.float32)
        self.correlation_params=nn.Parameter(correlation_params,requires_grad=True)
    
        
    def get_neighbors(self, curr_positions):
        # l_inf-dist
        # B,A,A
        dists_y=torch.abs(curr_positions[:,:,None,0]-curr_positions[:,None,:,0])
        dists_x=torch.abs(curr_positions[:,:,None,1]-curr_positions[:,None,:,1])
        dists=torch.maximum(dists_y,dists_x)
        
        selected=dists<=self.max_dist
        
        # [K,3]: b,a1,a2
        batch_indices, head_indices, tail_indices = torch.nonzero(selected, as_tuple=True)
        
        y_indices = dists_y[batch_indices, head_indices, tail_indices]+self.max_dist
        x_indices = dists_x[batch_indices, head_indices, tail_indices]+self.max_dist
        
        param_indices = y_indices*self.FoV_w+x_indices
        
        return batch_indices, head_indices, tail_indices, param_indices

    def forward(self, logits, illegal_action_masks, curr_positions):
        '''
        logits: [B,A,#actions]
        curr_positions: [B,A,2]
        '''
        q_logits=self.inference(logits, illegal_action_masks, curr_positions)
        return q_logits
    
    def expectation_propagation_inference(self,logits, illegal_action_masks, curr_positions):
        
        
        
        
        
        pass
    
    def inference(self, logits, illegal_action_masks, curr_positions):
        '''
        logits: [B,A,#actions]
        curr_positions: [B,A,2]
        '''
        
        # [K,],[K,],[K,],[K,]
        batch_indices, head_indices, tail_indices, param_indices = self.get_neighbors(curr_positions)
        
        # K, #actions, #actions
        params = self.correlation_params[param_indices]
        
        B,A,_=logits.shape
        q_logits=torch.rand_like(logits)
        q_logits=q_logits-1e10*illegal_action_masks
        for iteration in range(self.iterations):
            # B, A, #actions
            q_probs=torch.softmax(q_logits,dim=-1)
            
            # K, #actions, 1
            tail_q_probs=q_probs[batch_indices, tail_indices][:,:,None]
            
            # K, #actions, 1 
            correlations = params @ tail_q_probs
            # K, #actions
            correlations = correlations.squeeze(-1)
            
            # we need to use scatter here
            indices=batch_indices*A+head_indices
            # B*A, #actions
            correlations = scatter(correlations, indices, dim=0, dim_size=B*A, reduce="sum")
            
            # B, A, #actions
            correlations = correlations.reshape(B,A,-1)
            
            # B, A, #actions
            q_logits = logits+correlations
            
            q_logits=q_logits-1e10*illegal_action_masks
            
        return q_logits
    
    def compute_logits(self, logits, illegal_action_masks, curr_positions, actions):
        '''
        logits: [B,A,#actions]
        curr_positions: [B,A,2]
        actions: [B,A]
        '''
        
        logits=logits-1e10*illegal_action_masks
        
        # [K,],[K,],[K,],[K,]
        batch_indices, head_indices, tail_indices, param_indices = self.get_neighbors(curr_positions)
        
        # K, #actions, #actions
        params = self.correlation_params[param_indices]
        K = len(params)
        
        B,A,_=logits.shape
       
        # B,A,1
        _actions = actions.unsqueeze(-1)
        # B,A,1
        _individual_logits = torch.gather(logits, dim=-1, index=_actions)
        
        # B
        _individual_logits = _individual_logits.squeeze(-1).sum(-1)

        
        # K,
        head_actions = actions[batch_indices, head_indices]
        tail_actions = actions[batch_indices, tail_indices]
        
        # K,
        _pair_logits = params[torch.arange(K,dtype=torch.int32,device=params.device),head_actions,tail_actions]
        # B,
        _pair_logits = scatter(_pair_logits, batch_indices, dim=0, dim_size=B, reduce="sum")
        
        _logits = _individual_logits+_pair_logits
        
        Logger.error("{} {}".format(_individual_logits.mean(),_pair_logits.mean()))
        
        return _logits
    
    def forward_training(self, logits, illegal_action_masks, curr_positions, actions):
        B,A,_=logits.shape
        
        # B,A,#actions
        q_logits = self.inference(logits, illegal_action_masks, curr_positions)
        
        # sample
        q_dist = torch.distributions.Categorical(logits=q_logits)
        sampled_actions = q_dist.sample() 

        # B,         
        negative_logits = self.compute_logits(logits, illegal_action_masks, curr_positions, sampled_actions.detach()) /A
        # B, 
        positive_logits = self.compute_logits(logits, illegal_action_masks, curr_positions, actions)/A

        training_logits = negative_logits - positive_logits
        
        Logger.error("{} {}".format(negative_logits.mean(),positive_logits.mean()))
        
        # minimize it
        return q_logits, training_logits
        
        