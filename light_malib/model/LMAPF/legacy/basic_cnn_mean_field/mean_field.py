import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class MeanFieldLayer(nn.Module):
    def __init__(self, iterations):
        super().__init__()
        self.iterations=iterations
        self.punish_coef=-100
        
    def forward(self, logits, illegal_action_masks, conflict_pairs):
        if len(conflict_pairs)==0:
            return logits
        
        conflict_pairs=conflict_pairs.long()
        
        N,A=logits.shape
        
        q_logits=torch.zeros_like(logits)
        q_logits=q_logits-1e10*illegal_action_masks
        for iteration in range(self.iterations):
            q_probs=torch.softmax(q_logits,dim=-1)
            agent1=conflict_pairs[:,0]
            act1=conflict_pairs[:,1]
            agent2=conflict_pairs[:,2]
            act2=conflict_pairs[:,3]    
            masks=conflict_pairs[:,4].float()
                    
            index1=agent1*A+act1
            index2=agent2*A+act2
        
            q_probs=q_probs.reshape(-1) 
            _q_probs=q_probs[index2]
            
            # the punishment maybe should be normalize the max/min of logits
            punishments=scatter(_q_probs*masks,index1,dim=-1,dim_size=q_probs.shape[0],reduce="sum")*self.punish_coef
            
            q_logits=logits+punishments.reshape(N,A)
            q_logits=q_logits-1e10*illegal_action_masks
            
        return q_logits