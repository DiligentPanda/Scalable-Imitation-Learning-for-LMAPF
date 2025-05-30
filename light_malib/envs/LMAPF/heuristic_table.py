import torch
from .map import Map
from light_malib.utils.timer import global_timer
import numpy as np

class HeuristicTable:
    def __init__(self,map:Map,padded_graph,padded_graph_offsets,empty_locs,main_heuristics,device):
        self.map=map
        self.padded_graph=padded_graph
        self.padded_graph_offsets=padded_graph_offsets
        self.device=device
        # loc_size
        self.empty_locs=torch.tensor(empty_locs,dtype=torch.int32,device=device)
        self.loc_size=len(self.empty_locs)
        # loc_size*loc_size
        self.main_heuristics=torch.tensor(main_heuristics,dtype=torch.float32,device=device).reshape(self.loc_size,self.loc_size)
        self.loc_idxs=torch.full((self.map.height*self.map.width,),fill_value=-1,dtype=torch.int32,device=device)
        self.loc_idxs[self.empty_locs]=torch.arange(len(self.empty_locs),dtype=torch.int32,device=device)
        
    def get_heuristics(self, curr_positions, views, target_positions):
        '''
        local_views [num_robots,...,2]
        target_positions [num_robots,2]
        '''
        
        local_views=curr_positions[:,None,None,:]+views
        offsetted_local_views=local_views+self.padded_graph_offsets
        
        # check bound and static obstacles
        # [num_robots, FOV_height, FOV_width]
        # masks 1:valid, 0:invalid
        masks=self.get_masks(offsetted_local_views)
        
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
        
        heuristics = heuristics.reshape(len(curr_positions), views.shape[0], views.shape[1])
        
        return heuristics, masks, local_views, offsetted_local_views
    
    def get_masks(self, offsetted_local_views):
        masks=self.padded_graph[offsetted_local_views[...,0],offsetted_local_views[...,1]]==0
        return masks
    
    def get_distances(self, curr_positions, target_positions):
        curr_position_locs=curr_positions[...,0]*self.map.width+curr_positions[...,1]
        # num_robots
        curr_position_idxs=self.loc_idxs[curr_position_locs]

        target_position_locs=target_positions[...,0]*self.map.width+target_positions[...,1]
        # num_robots
        target_position_idxs=self.loc_idxs[target_position_locs]
        
        heuristics=self.main_heuristics[curr_position_idxs,target_position_idxs]
        
        return heuristics
    
    
class GuidedHeuristicTable:
    def __init__(self, map:Map, padded_graph, padded_graph_offsets, PyShadowSystem, device):
        self.map=map
        self.PyShadowSystem=PyShadowSystem
        self.padded_graph=padded_graph
        self.padded_graph_offsets=padded_graph_offsets
        self.device=device
        
    def get_heuristics(self, curr_positions, views, target_positions):
        '''
        curr_positions [num_robots,2]
        views [vh,vw,2]
        target_positions [num_robots,2]
        '''
        global_timer.record("func_get_h_s")
        
        global_timer.record("h_get_mask_s")
        local_views=curr_positions[:,None,None,:]+views
        offsetted_local_views=local_views+self.padded_graph_offsets
        
        # check bound and static obstacles
        # [num_robots, FOV_height, FOV_width]
        # masks 1:valid, 0:invalid
 
        masks=self.get_masks(offsetted_local_views)
        global_timer.time("h_get_mask_s","h_get_mask_e","h_get_mask") 

        global_timer.record("query_h_s")
        _locations = curr_positions[...,0]*self.map.width+curr_positions[...,1]
        _views_y = views[...,0].flatten()
        _views_x = views[...,1].flatten()

        heuristics=self.PyShadowSystem.query_heuristics(
            _locations.cpu().numpy().tolist(),
            _views_y.cpu().numpy().tolist(),
            _views_x.cpu().numpy().tolist()
        )
        global_timer.time("query_h_s","query_h_e","query_h")
        global_timer.record("h_to_gpu_s")
        heuristics = torch.from_numpy(np.array(heuristics,dtype=np.float32)).to(device=self.device)
        heuristics = heuristics.reshape(len(_locations), views.shape[0], views.shape[1])
        global_timer.time("h_to_gpu_s","h_to_gpu_e","h_to_gpu")
        
        # apply masks
        heuristics[~masks]=-1
        
        global_timer.time("func_get_h_s","func_get_h_e","func_get_h")
        
        return heuristics, masks, local_views, offsetted_local_views
    
    def get_masks(self, offsetted_local_views):
        masks=self.padded_graph[offsetted_local_views[...,0],offsetted_local_views[...,1]]==0
        return masks
    
    def get_distances(self, curr_positions, target_positions):        
        _locations = curr_positions[...,0]*self.map.width+curr_positions[...,1]
        heuristics=self.PyShadowSystem.query_heuristics(
            _locations.cpu().numpy().tolist(),
            [0],
            [0]
        )
        
        heuristics = torch.from_numpy(np.array(heuristics,dtype=np.float32)).to(device=self.device,dtype=torch.float32)
        heuristics = heuristics.reshape(len(_locations))
        
        return heuristics