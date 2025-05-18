import torch
import torch.nn as nn
import torchvision.ops as ops
from light_malib.utils.logger import Logger

class OrderedRoISampling(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        
        self.h = h
        self.w = w
        
        ys = torch.arange(self.h, dtype=torch.long).reshape(self.h,1).repeat(1,self.w).reshape(-1)
        xs = torch.arange(self.w, dtype=torch.long).reshape(1,self.w).repeat(self.h,1).reshape(-1)
        reverse_ys = torch.arange(self.h-1,-1,-1,dtype=torch.long).reshape(self.h,1).repeat(1,self.w).reshape(-1)  
        reverse_xs = torch.arange(self.w-1,-1,-1,dtype=torch.long).reshape(self.h,1).repeat(1,self.w).reshape(-1)
        
        # ys,xs
        idxs00=ys*self.w+xs 
        idxs01=ys*self.w+reverse_xs
        idxs10=reverse_ys*self.w+xs
        idxs11=reverse_ys*self.w+reverse_xs
        
        idxs=torch.stack([idxs00,idxs01,idxs01,idxs10,idxs11],dim=0)
        self.register_buffer("idxs", idxs)
    
    def forward(self, feats, batch_idxs, starts, goals):
        '''
        feats: B,C,H,W
        batch_idxs: B*A, 
        starts: B*A,2
        goals: B*A,2
        returns:
            rois: B*A, C, h, w
        '''
        
        starts=starts.float()
        goals=goals.float()
        
        y_mins = torch.min(starts[...,0],goals[...,0])
        y_maxs = torch.max(starts[...,0],goals[...,0])
        x_mins = torch.min(starts[...,1],goals[...,1])
        x_maxs = torch.max(starts[...,1],goals[...,1])
        
        # B*A
        y_reversed = starts[...,0]>goals[...,0]
        # B*A
        x_reversed = starts[...,1]>goals[...,1]
        
        # B*A
        idx_idxs = y_reversed*2+x_reversed
        # B*A, h*w
        idxs = self.idxs[idx_idxs]
        # B*A, C, h*w
        idxs = idxs.unsqueeze(1).repeat(1,feats.shape[1],1)
        
        # B*A, 5
        boxes=torch.stack([batch_idxs, x_mins, y_mins, x_maxs, y_maxs], dim=-1)

        # B*A, C, h, w
        rois = ops.roi_align(feats, boxes, output_size=(self.h,self.w),aligned=True)
        # B*A, C, h*w
        rois = rois.reshape(*rois.shape[:2],self.h*self.w)
        # B*A, C, h*w
        rois = torch.gather(rois, -1, idxs)
        # B*A, C, h, w
        rois = rois.reshape(*rois.shape[:2],self.h,self.w)
        
        return rois
        