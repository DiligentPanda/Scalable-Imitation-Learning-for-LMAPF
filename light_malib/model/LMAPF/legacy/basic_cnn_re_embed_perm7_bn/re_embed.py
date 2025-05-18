import torch
import torch.nn as nn
import torch.nn.functional as F
from light_malib.utils.logger import Logger
from torch.autograd import Function
from torchvision.ops import roi_align

class Scatter(Function):
    @staticmethod
    def forward(ctx, global_feats, idx0, idx2, idx3, local_feats):
        '''
        att_weights: B, nh, L0, L1
        self_atten_weights: B, nh, L0
        idx1: L0
        idx2: L0
        '''
        ctx.save_for_backward(idx0, idx2, idx3)
        output=global_feats.detach().clone()
        output[idx0, : , idx2, idx3] = output[idx0, : , idx2, idx3] + local_feats
                
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        idx0, idx2, idx3 = ctx.saved_tensors
        grad_local_feats = grad_output[idx0,:,idx2,idx3]
        return grad_output, None, None, None, grad_local_feats
    
scatter=Scatter.apply

class ReEmbed(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        
        self.in_dim=hidden_dim
        self.hidden_dim=hidden_dim
        self.FOV_height=11
        self.FOV_width=11
                
        self.global_in_dim=32
        
        self.global_proj=nn.Sequential(
            nn.BatchNorm2d(self.global_in_dim),
            nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=1)
        )
            
        self.local_proj=nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim+action_dim),
            nn.Linear(self.hidden_dim+action_dim, self.hidden_dim),
        )
        
        self.backbone=nn.Sequential(
                nn.BatchNorm2d(self.in_dim),
                # 11,11 -> 9,9
                nn.Conv2d(self.in_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_dim),
                # 9,9 -> 7,7
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_dim),
                # 7,7- > 5,5
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_dim),
                # 5,5 -> 3,3
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_dim),
                # 3,3 -> 1,1
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            )
        
    def forward(self, global_features, curr_positions, local_features, observations):
        '''
        global_features: B, Fg, H, W
        curr_positions: B*A, 2
        local_features: B*A, Fl
        '''
        B = global_features.shape[0]
        A = local_features.shape[0] // global_features.shape[0]
        curr_positions=curr_positions.long()
                
        global_features = self.global_proj(global_features)
        local_features = self.local_proj(local_features)
        
        device=global_features.device
        # TODO: move to the __init__, we add FOV_height//2 because of the padding         
        offsets_y=torch.arange(-(self.FOV_height//2),(self.FOV_height+1)//2,dtype=torch.int32,device=device)
        offsets_x=torch.arange(-(self.FOV_width//2),(self.FOV_width+1)//2,dtype=torch.int32,device=device)
        local_view_offsets=torch.stack(torch.meshgrid([offsets_y,offsets_x],indexing="ij"),dim=-1)
        padded_graph_offsets=torch.tensor([self.FOV_height//2,self.FOV_width//2],dtype=torch.int32,device=device)
        
        # B*A, FOV_height, FOV_weight, 2
        local_views=curr_positions[:,None,None,:]+local_view_offsets
        offsetted_local_views=local_views+padded_graph_offsets
        # B*A, 2
        offsetted_positions=curr_positions+padded_graph_offsets
        
        # TODO: we can scatter first, then pad
        # B, C, map_height, map_weight
        padded_global_features=F.pad(global_features,(self.FOV_height//2,self.FOV_height//2,self.FOV_width//2,self.FOV_width//2),mode="constant",value=-1)
        # B*A, FOV_h, FOV_w
        batch_indices=torch.arange(B,device=device).reshape(-1,1,1,1).repeat(1,A,self.FOV_height,self.FOV_width).reshape(-1,self.FOV_height,self.FOV_width)
        
        # B*A, Fg 
        # Logger.error("{} {} {} {} {}".format(batch_indices.shape, offsetted_positions.shape, local_features.shape, padded_global_features.shape, local_view_offsets.shape))
        # import time
        # time.sleep(10)
        padded_global_features=scatter(padded_global_features,batch_indices[:,0,0],offsetted_positions[...,0],offsetted_positions[...,1],local_features)
        
        # B*A, FOV_h, FOV_w, Fg
        _local_features = padded_global_features[batch_indices,:,offsetted_local_views[...,0],offsetted_local_views[...,1]]
        
        # B*A, Fl, FOV_h, FOV_w
        _local_features = _local_features.permute(0,3,1,2).view(B*A,self.hidden_dim,self.FOV_height,self.FOV_width)

        # B*A, Fl, 1, 1
        local_features = self.backbone(_local_features).view(B*A,-1)
        
        return local_features
        



class RoIReEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.in_dim=32
        self.hidden_dim=32
        self.FOV_height=11
        self.FOV_width=11
        
        self.global_in_dim=32
        
        self.global_proj=nn.Sequential(
            nn.BatchNorm2d(self.global_in_dim),
            nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=1)
        )
            
        self.local_proj=nn.Sequential(
            nn.BatchNorm1d(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
        )
        
        self.backbone=nn.Sequential(
                nn.BatchNorm2d(self.in_dim),
                # 11,11 -> 9,9
                nn.Conv2d(self.in_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_dim),
                # 9,9 -> 7,7
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_dim),
                # 7,7- > 5,5
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_dim),
                # 5,5 -> 3,3
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
                nn.ReLU(),
                nn.BatchNorm2d(self.hidden_dim),
                # 3,3 -> 1,1
                nn.Conv2d(self.hidden_dim,self.hidden_dim,3),
            )
        
        self.global_backbone=nn.Sequential(
            nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
            nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
            nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
            nn.Conv2d(self.global_in_dim,self.hidden_dim,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=2,dilation=2),
            nn.ReLU(),
            nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=4,dilation=4),
            nn.ReLU(),
            nn.LayerNorm([self.hidden_dim,self.map_height,self.map_width]),
            nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=8,dilation=8),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )     
        
    def forward(self, global_features, curr_positions, target_positions, local_features):
        '''
        global_features: B, Fg, H, W
        curr_positions: B*A, 2
        target_positions: B*A, 2
        local_features: B*A, Fl
        '''
        B = global_features.shape[0]
        A = local_features.shape[0] // global_features.shape[0]
        curr_positions=curr_positions.long()
        
        global_features = self.global_proj(global_features)
        local_features = self.local_proj(local_features)
        
        # B*A, FOV_h, FOV_w
        batch_indices=torch.arange(B,device=device).reshape(-1,1,1,1).repeat(1,A,self.FOV_height,self.FOV_width).reshape(-1,self.FOV_height,self.FOV_width)
        
        # re-embed features
        global_features = scatter(global_features, batch_indices[:,0,0], curr_positions[...,0], curr_positions[...,1], local_features)
        
        # we can actually use some pooling here  
        global_features = self.global_backbone(global_features)
        
        # RoI sampling
        # B*A, 1
        # TODO: we cannot use roi align now, because it requires ordered coordinates
        # TODO: but we can sort and flip later by gather operation!
        # batch_indices_for_RoI=torch.arange(B,device=device).reshape(-1,1).repeat(1,A).reshape(-1,1)
        # boxes=torch.stack([batch_indices_for_RoI, curr_positions, target_positions], dim=-1)        
        # roi_align(global_features, boxes, output_size=(3,3), )
        
        
        # we simple get the curr and goal position features
        
        
        device=global_features.device
        # TODO: move to the __init__, we add FOV_height//2 because of the padding         
        offsets_y=torch.arange(-(self.FOV_height//2),(self.FOV_height+1)//2,dtype=torch.int32,device=device)
        offsets_x=torch.arange(-(self.FOV_width//2),(self.FOV_width+1)//2,dtype=torch.int32,device=device)
        local_view_offsets=torch.stack(torch.meshgrid([offsets_y,offsets_x],indexing="ij"),dim=-1)
        padded_graph_offsets=torch.tensor([self.FOV_height//2,self.FOV_width//2],dtype=torch.int32,device=device)
        
        # B*A, FOV_height, FOV_weight, 2
        local_views=curr_positions[:,None,None,:]+local_view_offsets
        offsetted_local_views=local_views+padded_graph_offsets
        
        # B, C, map_height, map_weight
        padded_global_features=F.pad(global_features,(self.FOV_height//2,self.FOV_height//2,self.FOV_width//2,self.FOV_width//2),mode="constant",value=-1)
        
        # B*A, FOV_h, FOV_w, Fg
        _local_features = padded_global_features[batch_indices,:,offsetted_local_views[...,0],offsetted_local_views[...,1]]
        
        # B*A, Fl, FOV_h, FOV_w
        _local_features = _local_features.permute(0,3,1,2).view(B*A,self.hidden_dim,self.FOV_height,self.FOV_width)

        # B*A, Fl, 1, 1
        local_features = self.backbone(_local_features).view(B*A,-1)
        
        
        
        
        
        return local_features