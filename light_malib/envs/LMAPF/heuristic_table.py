import torch
from .map import Map

class HeuristicTable:
    def __init__(self,map: Map,padded_graph,empty_locs,main_heuristics,device):
        self.map=map
        self.padded_graph=padded_graph
        self.device=device
        # loc_size
        self.empty_locs=torch.tensor(empty_locs,dtype=torch.int32,device=device)
        self.loc_size=len(self.empty_locs)
        # loc_size*loc_size
        self.main_heuristics=torch.tensor(main_heuristics,dtype=torch.float32,device=device).reshape(self.loc_size,self.loc_size)
        self.loc_idxs=torch.full((self.map.height*self.map.width,),fill_value=-1,dtype=torch.int32,device=device)
        self.loc_idxs[self.empty_locs]=torch.arange(len(self.empty_locs),dtype=torch.int32,device=device)
        
    def get_heuristics(self, local_views, offsetted_local_views, target_positions):
        '''
        local_views [num_robots,...,2]
        target_positions [num_robots,2]
        '''
        
        # check bound and static obstacles
        # [num_robots, FOV_height, FOV_width]
        # masks 1:valid, 0:invalid
        masks=self.padded_graph[offsetted_local_views[...,0],offsetted_local_views[...,1]]==0
        
        num_out_of_bound=torch.numel(masks)-torch.count_nonzero(masks)
        
        local_views_locs=local_views[...,0]*self.map.width+local_views[...,1]
        local_views_locs[~masks]=self.empty_locs[torch.arange(num_out_of_bound,dtype=torch.int32)%len(self.empty_locs)]
        # num_robots, FOV_height, FOV_width
        local_view_idxs=self.loc_idxs[local_views_locs]
        
        target_position_locs=target_positions[...,0]*self.map.width+target_positions[...,1]
        # num_robots
        target_position_idxs=self.loc_idxs[target_position_locs]
        target_position_idxs=target_position_idxs.reshape(-1,*([1]*(local_views.dim()-2))).repeat(1,*local_views.shape[1:-1])
        
        heuristics=self.main_heuristics[local_view_idxs,target_position_idxs]
        
        # apply masks
        heuristics[~masks]=-1
        
        return heuristics, masks
    
    def get_distances(self, curr_positions, target_positions):
        curr_position_locs=curr_positions[...,0]*self.map.width+curr_positions[...,1]
        # num_robots
        curr_position_idxs=self.loc_idxs[curr_position_locs]

        target_position_locs=target_positions[...,0]*self.map.width+target_positions[...,1]
        # num_robots
        target_position_idxs=self.loc_idxs[target_position_locs]
        
        heuristics=self.main_heuristics[curr_position_idxs,target_position_idxs]
        
        return heuristics
    
    def to_device(self, device):
        self.device=device
        self.padded_graph=self.padded_graph.to(device)
        self.empty_locs=self.empty_locs.to(device)
        self.main_heuristics=self.main_heuristics.to(device)
        self.loc_idxs=self.loc_idxs.to(device)