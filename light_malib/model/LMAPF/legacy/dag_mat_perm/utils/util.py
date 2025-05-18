import copy
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

class SelfAttEmbedFunction(Function):
    @staticmethod
    def forward(ctx, att_weights, self_atten_weights, idxs1, idxs2):
        '''
        att_weights: B, nh, L0, L1
        self_atten_weights: B, nh, L0
        idx1: L0
        idx2: L0
        '''
        ctx.save_for_backward(idxs1, idxs2)
        output=att_weights.clone()
        output[:,:,idxs1,idxs2] = self_atten_weights
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        idxs1, idxs2 = ctx.saved_tensors
        grad_self_atten_weights = grad_output[:,:,idxs1,idxs2]
        grad_att_weights = grad_output.clone()
        grad_att_weights[:,:,idxs1,idxs2]=0
        
        return grad_att_weights, grad_self_atten_weights, None, None

class SelfAttSplitFunction(Function):
    @staticmethod
    def forward(ctx, att_weights, idxs1, idxs2):
        '''
        att_weights: B, nh, L0, L1
        self_atten_weights: B, nh, L0
        idx1: L0
        idx2: L0
        '''
        ctx.save_for_backward(idxs1, idxs2)
        
        self_att_weights = att_weights[:,:,idxs1,idxs2]
        att_weights = att_weights.clone()
        att_weights[:,:,idxs1,idxs2] = 0
        
        return att_weights, self_att_weights
    
    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        idxs1, idxs2 = ctx.saved_tensors
        grad_att_weights = grad_output1.clone()
        grad_att_weights[:,:,idxs1,idxs2]=grad_output2
        
        return grad_att_weights, None, None
    
self_atten_embed=SelfAttEmbedFunction.apply
self_atten_split=SelfAttSplitFunction.apply