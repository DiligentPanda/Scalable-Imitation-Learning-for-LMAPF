from .map import Map, MapManager, ProgressManager, Progress
import torch

from light_malib.utils.episode import EpisodeKey
from ..base_env import BaseEnv
from light_malib.registry import registry
import ray
import numpy as np
from light_malib.utils.distributed import get_actor
from light_malib.utils.logger import Logger
import sys 
import torch.nn.functional as F
import queue
from light_malib.utils.timer import global_timer
from collections import defaultdict

class EpisodeLog:
    def __init__(self, num_robots, enabled=True):
        self.log={
            "actionModel": "MAPF",
            # "AllValid": None,
            "teamSize": num_robots,
            "start": [],
            "numTaskFinished": -1,
            # "sumOfCost": None,
            "makespan": -1,
            "actualPaths": [[] for i in range(num_robots)],
            # "plannerTimes": None,
            # "errors": None,
            "events": [[] for i in range(num_robots)],
            "tasks": [],
        }
        
        self.tasks=[]
        self.agent_tasks=[None]*num_robots
        
        # NOTE this need to follow the same order as movements in the env
        self.actions=["R","D","L","U","W"]
        
        self.enabled=enabled
        
    def add_starts(self, locs):
        if not self.enabled:
            return
        self.log["start"]=[loc+["E"] for loc in locs.cpu().numpy().tolist()]
        
    def add_actions(self, actions):
        if not self.enabled:
            return
        actions=actions.cpu().numpy().tolist()
        for agent_idx in range(len(actions)):
            self.log["actualPaths"][agent_idx].append(self.actions[actions[agent_idx]])
            
    def add_completed_tasks(self, step, reached):
        if not self.enabled:
            return
        reached=reached.cpu().numpy().tolist()
        for agent_idx in range(len(reached)):
            if reached[agent_idx]:
                self.log["events"][agent_idx].append(
                    [step, self.agent_tasks[agent_idx][0], "finished"]
                )
    
    def add_new_tasks(self, step, reached, target_locs):
        if not self.enabled:
            return
        if reached is not None:
            reached=reached.cpu().numpy().tolist()
        target_locs=target_locs.cpu().numpy().tolist() 
        for agent_idx in range(len(target_locs)):
            if reached is None or reached[agent_idx]:
                task=[len(self.log["tasks"]), *target_locs[agent_idx]]
                self.agent_tasks[agent_idx]=task
                self.log["tasks"].append(task)
                self.log["events"][agent_idx].append(
                    [step, self.agent_tasks[agent_idx][0], "assigned"]
                )
    
    def summarize(self):
        numTaskFinsihed=0
        for agent_events in self.log["events"]:
            for event in agent_events:
                if event[-1]=="finished":
                    numTaskFinsihed+=1
        
        self.log["numTaskFinished"]=numTaskFinsihed
        self.log["makespan"]=len(self.log["actualPaths"][0])
    
    def dump(self, path):
        self.summarize()
        
        for i in range(len(self.log["actualPaths"])):
            self.log["actualPaths"][i]=",".join(self.log["actualPaths"][i])

        import json
        with open(path,"w") as f:
            json.dump(self.log,f)
            
    def __str__(self):
        self.summarize()
        return str(self.log)

class HeuristicTable:
    def __init__(self,map,padded_graph,empty_locs,main_heuristics,device):
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

class LMAPFEnv(BaseEnv):
    '''
    location used in this class are [r,c] or [y,x]
    '''
    # TODO: id and random seed
    def __init__(
        self, 
        id, 
        seed,
        map,
        num_robots, 
        device,     
        cfg,
        precompute_HT         
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        super().__init__(id, seed)
        self.cfg=cfg
        
        self.map = map
        self.num_robots = num_robots
        self.device = device
            
        self.rollout_length=self.cfg["rollout_length"]
        self.gae_gamma=self.cfg["gae_gamma"]
        self.gae_lambda=self.cfg["gae_lambda"]
        self.mappo_reward=self.cfg["mappo_reward"]
        self.WPPL_cfg=self.cfg["WPPL"]
        self.use_rank_feats=self.cfg["use_rank_feats"]
        
        self.agent_id="agent_0"
        self.agent_ids=[self.agent_id]
    
        # right, down, left, up, stay
        self.movements=torch.tensor([[0,1],[1,0],[0,-1],[-1,0],[0,0]],device=self.device,dtype=torch.int32)
        self.movements_py=self.movements.cpu().numpy()
        self.action_names=["R","D","L","U","W"]
        self.action_dim=len(self.movements)
        
        self.FOV_height=11
        self.FOV_width=11
        self.num_channels=4
        self.num_global_channels=3
        
        assert self.FOV_height%2==1
        assert self.FOV_width%2==1
        
        # TODO: move to the __init__, we add FOV_height//2 because of the padding
        offsets_y=torch.arange(-(self.FOV_height//2),(self.FOV_height+1)//2,dtype=torch.int32,device=self.device)
        offsets_x=torch.arange(-(self.FOV_width//2),(self.FOV_width+1)//2,dtype=torch.int32,device=self.device)
        # # FOV_height, FOV_width, 2
        self.local_view_offsets=torch.stack(torch.meshgrid([offsets_y,offsets_x],indexing="ij"),dim=-1)
        self.padded_graph_offsets=torch.tensor([self.FOV_height//2,self.FOV_width//2],dtype=torch.int32,device=self.device)
    
        # legacy code?
        self.team_sizes = {          
            self.agent_id: self.num_robots
        }
        
        # TODO(rivers): maybe we can cache some results later.
        
         # 1 - obstacle, 0 - empty location
        self.graph=torch.tensor(self.map.graph,device=self.device,dtype=torch.int32)
        self.empty_locations=torch.nonzero(self.graph==0).type(torch.int32)
        self.padded_graph=torch.nn.functional.pad(self.graph,(self.FOV_height//2,self.FOV_height//2,self.FOV_width//2,self.FOV_width//2),mode='constant',value=1)
        
        # self.padded_graph_location_ys=torch.zeros_like(self.padded_graph)
        # self.padded_graph_location_xs=torch.zeros_like(self.padded_graph)
        
        # self.padded_graph_location_ys=torch.arange(-self.FOV_height//2+1,self.map.height+self.FOV_height//2,dtype=torch.float32,device=self.device).reshape(-1,1).repeat(1,self.map.width+self.FOV_width-1)/(self.map.height*2)
        # self.padded_graph_location_xs=torch.arange(-self.FOV_width//2+1,self.map.width+self.FOV_width//2,dtype=torch.float32,device=self.device).repeat(self.map.height+self.FOV_height-1,1)/(self.map.width*2)
        
        # assert self.padded_graph_location_ys.shape[0]==self.map.height+self.FOV_height-1, "{} vs {}".format(self.padded_graph_location_ys.shape[0],self.map.height+self.FOV_height-1)
        # assert self.padded_graph_location_ys.shape[1]==self.map.width+self.FOV_width-1
        # assert self.padded_graph_location_xs.shape[0]==self.map.height+self.FOV_height-1
        # assert self.padded_graph_location_xs.shape[1]==self.map.width+self.FOV_width-1
        
        corner_graph=(1-self.graph).float()
        kernel=torch.tensor([[0,1,0],[1,0,1],[0,1,0]],dtype=torch.float32,device=self.device)
        self.corner_graph = F.conv2d(corner_graph.reshape(1,1,*corner_graph.shape), kernel.reshape(1,1,*kernel.shape), padding=kernel.shape[0]//2)
        self.corner_graph = (self.corner_graph<=1.0).reshape(*corner_graph.shape)
        
        
        if precompute_HT:
            sys.path.insert(0,"lmapf_lib/MAPFCompetition2023/build")
            import py_compute_heuristics
            # TODO
            ret=py_compute_heuristics.compute_heuristics(self.map.height, self.map.width, self.map.graph.flatten().tolist(), self.cfg["map_weights_path"])
            loc_size,empty_locs,main_heuristics=ret
            assert empty_locs.size==loc_size and main_heuristics.size==loc_size*loc_size

            self.heuristic_table=HeuristicTable(self.map,self.padded_graph,empty_locs,main_heuristics,self.device)
            
            
            # TODO(rivers): py_PLNS will compute the heuristic table internally           
            sys.path.insert(0,"lmapf_lib/MAPFCompetition2023/build")
            import py_PLNS
            # TODO: we should not recompute heuristic table here
            self.PLNSSolver=py_PLNS.PLNSSolver(
                self.map.height,
                self.map.width, 
                self.map.graph.flatten().tolist(),
                self.cfg["map_weights_path"],
                self.num_robots,
                self.WPPL_cfg.window_size,
                self.WPPL_cfg.num_threads,
                self.WPPL_cfg.max_iterations,
                self.WPPL_cfg.verbose
            )
            
        else:
            self.heuristic_table=None
            self.py_PLNSSolver=None

        # ret=py_compute_heuristics.compute_heuristics(cfg["map_path"],"")
        # loc_size,empty_locs,main_heuristics=ret
        # assert empty_locs.size==loc_size and main_heuristics.size==loc_size*loc_size
        # self.uniform_heuristic_table=HeuristicTable(self.map,self.padded_graph,empty_locs,main_heuristics,self.device)
    
        self.starts=[]
        self.tasks=[]
    
        # _num_conflict_actions=29
        # self.conflict_action_table={}
        # for dy in range(-2,2+1):
        #     for dx in range(-2,2+1):
        #         loc1=np.array([0,0])
        #         loc2=np.array([dy,dx])
        #         for act1, mov1 in enumerate(self.movements_py):
        #             for act2, mov2 in enumerate(self.movements_py):
        #                 next_loc1=loc1+mov1
        #                 next_loc2=loc2+mov2
        #                 if np.all(next_loc1==next_loc2) or (np.all(next_loc1==loc2) and np.all(next_loc2==loc1)):
        #                     if (dy,dx) not in self.conflict_action_table:
        #                         self.conflict_action_table[(dy,dx)]=[]
        #                     self.conflict_action_table[(dy,dx)].append((act1,act2))
        # self.num_conflict_actions=sum([len(l) for l in self.conflict_action_table.values()])
        # assert _num_conflict_actions==self.num_conflict_actions, "{} {} {}".format(_num_conflict_actions,self.num_conflict_actions,self.conflict_action_table)
         
        # TODO: preload heuristic graph
        # self.data_server=get_actor("Env_{}".format(self.id),"DataServer")
        # empty_locs=ray.get(self.data_server.get.remote("empty_locs"))
        # main_heuristics=ray.get(self.data_server.get.remote("main_heuristics"))
    
        sys.path.insert(0,"lmapf_lib/MAPFCompetition2023/build")
        import py_PIBT
        self.PIBTSolver=py_PIBT.PIBTSolver(seed)
        
        self._initialized=False
        
        self.prev_step_data = None
        
        # NOTE: this is not the real one-shot MAPF, just use to fix targets
        self._one_shot=False
        self._pibt_func="guard"
        self._eval=True
        
        # self.num_edges=10
        
        # self.history_len=3
        
        self.use_permutation=True
        # assert not self.use_permutation, "it is not supported anymore, if we don't want to use sequential models"
        
        self._enable_log=False
        self._check_valid=False
        
    def get_HT(self):
        return self.heuristic_table
    
    def get_PLNSSolver(self):
        return self.PLNSSolver
    
    def set_HT(self, HT:HeuristicTable):
        self.heuristic_table=HT
        self.heuristic_table.to_device(self.device)
        
    def set_PLNSSolver(self, PLNSSolver):
        self.PLNSSolver=PLNSSolver
        
    def set_seed(self,seed):
        self.seed=seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def check_valid(self,flag):
        self._check_valid=flag
        
    def enable_log(self,flag):
        self._enable_log=flag

    def load_starts(self, start_fp):
        with open(start_fp) as f:
            self.starts=[]
            num_agents = int(f.readline().strip())
            assert num_agents==self.num_robots, "{} vs {}".format(num_agents,self.num_robots)
            for i in range(num_agents):
                loc = int(f.readline().strip())
                self.starts.append((loc//self.map.width,loc%self.map.width))
        self.starts=torch.from_numpy(np.array(self.starts,dtype=np.int32)).to(self.device)
        
    def load_tasks(self, tasks_fp):
        with open(tasks_fp) as f:
            self.tasks=[]
            num_tasks = int(f.readline().strip())
            for i in range(num_tasks):
                loc = int(f.readline().strip())
                self.tasks.append((loc//self.map.width,loc%self.map.width))
        self.tasks=torch.from_numpy(np.array(self.tasks,dtype=np.int32)).to(self.device)
        self.tasks_completed=torch.zeros(self.num_robots,dtype=torch.int32,device=self.device)

    def set_eval(self, eval):
        self._eval=eval
    
    def set_rollout_length(self, rollout_length):
        self.rollout_length=rollout_length
        
    def set_one_shot(self, one_shot):
        self._one_shot=one_shot
        
    def set_pibt_func(self, pibt_func):
        assert pibt_func in ["guard","none","solve"]
        self._pibt_func=pibt_func
        
    def get_pibt_func(self):
        return self._pibt_func
        
    def PIBT_solve(self, guiding_actions=None, heuristics=None):
        # TODO: we need input actions, otherwise it is just a PIBT
        # TODO: we can add PIBT prior to the actor
        
        priorities=self.priorities.flatten().cpu().numpy().tolist()
        locations=self.curr_positions.flatten().cpu().numpy().tolist()
        action_choices=self.movements.flatten().cpu().numpy().tolist()
        map_size=[self.map.height,self.map.width]
        
        # get heuristics for PIBT
        local_views=self.curr_positions[:,None,:]+self.movements
        offseted_local_views=local_views+self.padded_graph_offsets
        # num_robots, num_actions
        
        assert guiding_actions is None or heuristics is None
                         
        if heuristics is None:
            heuristics,_=self.heuristic_table.get_heuristics(local_views,offseted_local_views,self.target_positions)
            # print(heuristics)
            # temp=heuristics.copy()
            # temp[temp<0]=1000000
            # best_actions=np.argmin(temp,axis=1)
            # # print(best_actions)
            # heuristics[np.arange(len(best_actions)),best_actions]=0
            # print(heuristics)
            # TODO: we actually need to check this action is valid.
            sampling=False
        else:
            sampling=True
            masks=self.padded_graph[offseted_local_views[...,0],offseted_local_views[...,1]]==0
            # TODO: sanity check: the sum of probs should be 1.
            heuristics=torch.softmax(heuristics,dim=-1)
            # heuristics[heuristics<1e-9]=0
            heuristics[~masks]=-1
                
        if guiding_actions is not None:
            if torch.any(heuristics[torch.arange(len(guiding_actions),dtype=torch.int32,device=self.device),guiding_actions]<0):
                Logger.error("err! this may sometimes happen if the probability is not safely clipped above 1e-9, for example.")
            # TODO: set to 0 has the potential bug when agent will arrive the goal
            heuristics[torch.arange(len(guiding_actions),dtype=torch.int32,device=self.device),guiding_actions]=torch.minimum(
                heuristics[torch.arange(len(guiding_actions),dtype=torch.int32,device=self.device),guiding_actions],
                torch.tensor(0,device=self.device,dtype=torch.float32)
            )
            
        heuristics=heuristics.cpu().numpy()
        heuristics=heuristics.flatten().tolist()
        
        actions=self.PIBTSolver.solve(priorities,locations,heuristics,action_choices,map_size,sampling)
        
        actions=torch.tensor(actions,dtype=torch.int32,device=self.device)
        
        return actions
    
    def initialized(self):
        return self._initialized
    
    def disable_agents(self):
        mask=self.corner_graph[self._target_positions[...,0],self._target_positions[...,1]]
        disabled=torch.zeros_like(self.priorities,dtype=torch.bool,device=self.device)
        disabled[mask]=True
        self.target_positions=self._target_positions.clone()
        # NOTE(rivers): don't use the following line! too agressive.
        # self.target_positions[mask]=self.curr_positions[mask]
        # NOTE(rivers): set the priorities to the lowest
        self.priorities[mask]=self.priorities[mask]-torch.floor(self.priorities[mask])
        
    def update_permutation(self):
        if not self.use_permutation:
            self.perm_indices=torch.arange(len(self.priorities),device=self.device)
        else:
            # we will keep the high-priority agent first
            # if self._eval:
            #     self.perm_indices=torch.argsort(self.priorities,descending=True)
            # else:
            self.perm_indices=torch.randperm(len(self.priorities),device=self.device)
        # self.reverse_perm_indices=torch.argsort(self.perm_indices)
        self.reverse_perm_indices=torch.arange(len(self.perm_indices),device=self.device,dtype=self.perm_indices.dtype)
        self.reverse_perm_indices[self.perm_indices]=self.reverse_perm_indices.clone()
    
    def record_progress(self):
        progress = None
        if not self.is_terminated():
            progress=Progress(
                # state
                self.step_ctr,
                self.curr_positions,
                self.target_positions,
                self.priorities,
                # statistics
                
            )
        
        self.progress_manager.record_progress(self.map.name, self.num_robots, progress)
        

    def reset(self, custom_reset_config):
        self._initialized=True
        # self.feature_encoders = custom_reset_config["feature_encoders"]
        # self.main_agent_id = custom_reset_config["main_agent_id"]
        # self.rollout_length = custom_reset_config["rollout_length"]
        
        self.step_ctr=0
        self.episode_log=EpisodeLog(self.num_robots, self._enable_log)

        for key in ["curr_positions","target_positions","priorities"]:
            if key in custom_reset_config:
                data=custom_reset_config[key]
                if key=="priorities":
                    dtype=torch.float32
                else:
                    dtype=torch.int32
                if isinstance(data,np.ndarray):
                    data=torch.tensor(data,dtype=dtype,device=self.device)
                else:
                    data=data.to(dtype=dtype,device=self.device)
                if key=="target_positions":
                    key="_"+key
                self.__setattr__(key,data)
            else:
                if key=="curr_positions":
                    self.curr_positions=self.sample_starts()
                elif key=="target_positions":
                    self._target_positions=self.sample_targets(self.num_robots)
                elif key=="priorities":
                    self.priorities=self.sample_priorities(self.num_robots)
                

        
        self.episode_log.add_starts(self.curr_positions)
        self.episode_log.add_new_tasks(0,None,self._target_positions)
                
        #self.disabled=torch.zeros_like(self.priorities,dtype=torch.bool,device=self.device)
    
        # statistics
        self.valid_ctr=0  
        self.action_consistent_rate=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        self.total_rewards=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        self.total_individual_rewards=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        self.total_team_rewards=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        # self.total_guiding_rewards=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        self.total_completed_tasks=torch.zeros_like(self.total_rewards)
        self.action_cnts=torch.zeros(size=(self.num_robots,self.action_dim),dtype=torch.float32,device=self.device)
        # self.vertex_usages=torch.zeros_like(self.graph,dtype=torch.float32,device=self.device)
        # self.edge_usages=torch.zeros((*self.graph.shape,self.action_dim),dtype=torch.float32,device=self.device)
        
        # self.reward_map_for_GGO=torch.zeros((*self.graph.shape,),dtype=torch.float32,device=self.device)
    
        # self.history=[]

        self.update_permutation()
        self.disable_agents()

        if self.use_permutation:
            self.curr_positions=self.curr_positions[self.perm_indices]
            self._target_positions=self._target_positions[self.perm_indices]
            self.target_positions=self.target_positions[self.perm_indices]
            self.priorities=self.priorities[self.perm_indices]

        observations, global_observations=self.get_observations()
        action_masks=self.get_action_masks()
        
        if self.use_permutation:
            self.curr_positions=self.curr_positions[self.reverse_perm_indices]
            self._target_positions=self._target_positions[self.reverse_perm_indices]
            self.target_positions=self.target_positions[self.reverse_perm_indices]
            self.priorities=self.priorities[self.reverse_perm_indices]
        
        # TODO: let set dones when agent finish tasks?
        dones=torch.zeros((self.num_robots,1),dtype=torch.bool,device=self.device)
    
        rets = {
            self.agent_id : {
                EpisodeKey.NEXT_OBS: observations,
                EpisodeKey.NEXT_GLOBAL_OBS: global_observations,
                EpisodeKey.ACTION_MASK: action_masks,
                EpisodeKey.DONE: dones
            }
        }
        return rets
    
    def sample_priorities(self, num):
        priorities=torch.rand((num,),dtype=torch.float32,device=self.device)
        return priorities
    
    def step(self, actions):
        '''
        actions should be int array
        shape: [num_robots,]
        value range: [0,4]
        return:
            reward:
            action_mask:
            ...
        '''
        global_timer.record("step_inner_s")
        self.step_ctr+=1
        
        global_timer.record("pibt_s")
        original_actions=actions[self.agent_id][EpisodeKey.ACTION]
        if isinstance(original_actions,np.ndarray):
            original_actions=torch.from_numpy(original_actions)
        if isinstance(original_actions,torch.Tensor):
            original_actions=original_actions.to(self.device,dtype=torch.int32)

        action_logits=actions[self.agent_id].get(EpisodeKey.ACTION_LOGITS,None)        
        
        if self._pibt_func=="guard":            
            if action_logits is not None:
                actions=self.PIBT_solve(heuristics=action_logits)
            else:
                actions=self.PIBT_solve(original_actions)
        elif self._pibt_func=="none":
            actions=original_actions
        elif self._pibt_func=="solve":          
            actions=self.PIBT_solve()
            original_actions=actions
        else:
            raise NotImplementedError("Unknown PIBT function: {}".format(self._pibt_func))


        self.actions=actions
        
        self.action_cnts[torch.arange(self.num_robots,device=self.device),actions]+=1
        self.episode_log.add_actions(actions)
        
        # assert len(actions)==self.num_robots
        # assert actions.max()<=4 and actions.min()>=0
        movements=self.movements[actions]
        next_positions=self.curr_positions+movements
        global_timer.time("pibt_s","pibt_e","pibt")
        
        global_timer.record("check_s")
        prev_positions=self.curr_positions
        valid_flag=True
        
        if self._check_valid:
            # TODO: make this part c++
            # check if out of bound
            if torch.any(next_positions[:,0]<0) or torch.any(next_positions[:,0]>=self.map.height) or torch.any(next_positions[:,1]<0) or torch.any(next_positions[:,1]>=self.map.width):
                print("out of bound")
                print(torch.nonzero(next_positions[:,0]<0))
                valid_flag=False
                
            if valid_flag:
                # check if collide with static obstacles
                if torch.any(self.graph[next_positions[:,0],next_positions[:,1]]==1):
                    print("collide with static obstacles")
                    valid_flag=False
                
                # check if collide with dynamic obstacles
                ## vertex collision
                # (num_robots,num_robots,2)
                vertex_collision=(next_positions[:,None,:]==next_positions[None,:,:]).all(dim=-1)
                vertex_collision[torch.arange(len(next_positions),device=self.device),torch.arange(len(next_positions),device=self.device)]=False
                if torch.any(vertex_collision):
                    print("vertex collision")
                    valid_flag=False
                
                ## edge collision
                # agent i's next location is the same as agent j's current location
                following_collision=(next_positions[:,None,:]==self.curr_positions[None,:,:]).all(dim=-1)
                edge_collision=torch.logical_and(following_collision,following_collision.T)
                edge_collision[torch.arange(len(next_positions),device=self.device),torch.arange(len(next_positions),device=self.device)]=False
                if torch.any(edge_collision):
                    print("edge collision")
                    valid_flag=False
            
        if valid_flag:
            # update current positions
            self.curr_positions=next_positions
            self.valid_ctr+=1
        else:
            pass
            # Logger.warning("Invalid action")
        global_timer.time("check_s","check_e","check")
        
        global_timer.record("reward_s")
        # check if reach targets
        reached=(self.curr_positions==self._target_positions).all(dim=-1)
        
        # compute rewards

        prev_costs=self.heuristic_table.get_distances(prev_positions,self._target_positions)
        curr_costs=self.heuristic_table.get_distances(self.curr_positions,self._target_positions)
        rewards, individual_rewards, team_rewards =self.get_rewards(prev_costs,curr_costs,reached)
        global_timer.time("reward_s","reward_e","reward")
        
        # consistent rewards
        # rewards+=(actions==original_actions).float()*0.1
        
        # update statistics
        global_timer.record("stats_s")
        self.total_rewards+=rewards
        self.total_individual_rewards+=individual_rewards
        self.total_team_rewards+=team_rewards
        self.total_completed_tasks+=reached.float()
        # self.vertex_usages[self.curr_positions[:,0],self.curr_positions[:,1]]+=1
        # self.edge_usages[self.curr_positions[:,0],self.curr_positions[:,1],actions]+=1
        self.action_consistent_rate+=(actions==original_actions).float()
        global_timer.time("stats_s","stats_e","stats")
        
        # TODO: we should use unweighted heuristic to compute reward here, so we need another heuristic table
        # global_timer.record("reward2_s")
        # prev_dists=self.heuristic_table.get_distances(prev_positions,self._target_positions)
        # curr_dists=self.heuristic_table.get_distances(self.curr_positions,self._target_positions)
        # individual_rewards=(prev_dists-curr_dists)-1
        # global_timer.time("reward2_s","reward2_e","reward2")
        #self.reward_map_for_GGO[prev_positions[:,0],prev_positions[:,1]]+=individual_rewards
        
        global_timer.record("reach_s")
        # if reached need to resample targets
        if torch.any(reached):
            self.episode_log.add_completed_tasks(self.step_ctr,reached)
            num_reached=torch.sum(reached.type(torch.int32)).item()
            # TODO(rivers): we should also set active mask
            # raise NotImplementedError
            if not self._one_shot:
                self._target_positions[reached]=self.sample_targets(num_reached, reached)
            self.episode_log.add_new_tasks(self.step_ctr,reached,self._target_positions)
            self.priorities[reached]=self.sample_priorities(num_reached)

        self.priorities[~reached]+=1
        
        global_timer.time("reach_s","reach_e","reach")
        
        global_timer.record("update_perm_s")
        self.update_permutation()

       
        self.disable_agents()
        global_timer.time("update_perm_s","update_perm_e","update_perm")
            
        if self.use_permutation:
            self.curr_positions=self.curr_positions[self.perm_indices]
            self._target_positions=self._target_positions[self.perm_indices]
            self.target_positions=self.target_positions[self.perm_indices]
            self.priorities=self.priorities[self.perm_indices]
        
        # update observations, etc.

        # global_timer.record("obs_s")
        observations, global_observations=self.get_observations()
        action_masks=self.get_action_masks()
        # global_timer.time("obs_s","obs_e","obs")
        
        if self.use_permutation:
            self.curr_positions=self.curr_positions[self.reverse_perm_indices]
            self._target_positions=self._target_positions[self.reverse_perm_indices]
            self.target_positions=self.target_positions[self.reverse_perm_indices]
            self.priorities=self.priorities[self.reverse_perm_indices]
        
                
        global_timer.record("to_cpu_s")
        # add an extra dim: some legacy settings in the framework
        rewards=rewards[...,None]
        
        if self.mappo_reward:
            rewards[:]=rewards.mean()
        
        # TODO: why use set reached to done doesn't work
        # dones=torch.zeros_like(reached[...,None])
        dones=reached[...,None]

        rets = {
            self.agent_id: {
                EpisodeKey.NEXT_OBS: observations,
                EpisodeKey.NEXT_GLOBAL_OBS: global_observations,
                EpisodeKey.ACTION_MASK: action_masks,
                EpisodeKey.REWARD: rewards,
                EpisodeKey.DONE: dones
            }
        }
        global_timer.time("to_cpu_s","to_cpu_e","to_cpu")
        
        global_timer.time("step_inner_s","step_inner_e","step_inner")
        
        # print(global_timer.mean_elapses)
        
        return rets
    
    def get_prev_step_data(self):
        return self.prev_step_data
    
    def set_prev_step_data(self, data):
        self.prev_step_data=data
        
    # def update_guiding_rewards(self, guiding_rewards):
    #     self.total_rewards+=guiding_rewards
    #     self.total_guiding_rewards+=guiding_rewards
        
    def get_episode_log(self):
        return self.episode_log
    
    def get_episode_stats(self):
        # TODO: add more statistics
        mean_reward=torch.mean(self.total_rewards)
        mean_individual_reward=torch.mean(self.total_individual_rewards)
        mean_team_reward=torch.mean(self.total_team_rewards)
        # mean_guiding_reward=torch.mean(self.total_guiding_rewards)
        mean_throughput=torch.sum(self.total_completed_tasks)/self.step_ctr
        valid_rate=self.valid_ctr/self.step_ctr
        mean_action_cnts=torch.mean(self.action_cnts,dim=0)
        mean_action_consistent_rate=torch.mean(self.action_consistent_rate,dim=0)/self.step_ctr
        
        stats={
            "reward":mean_reward,
            "throughput": mean_throughput,
            "valid_rate":valid_rate,
            "2_individual_reward":mean_individual_reward,
            "3_team_reward":mean_team_reward,
            "4_guiding_reward": 0
            }
        for i in range(len(self.action_names)):
            stats["{}_".format(len(stats)+i)+self.action_names[i]]=mean_action_cnts[i].item()
        
        # my stupid idx...
        offset=10
        stats["{}_consistent_rate".format(len(stats)+offset)]=mean_action_consistent_rate.item()
        
        return {self.agent_id: stats}
    
    def is_terminated(self):
        return self.step_ctr>=self.rollout_length
    
    def sample_starts(self):
        assert self.num_robots<=len(self.empty_locations),"#robots {} should be smaller than #empty locs {}".format(self.num_robots,len(self.empty_locations))
        if len(self.starts)==0:
            return self.empty_locations[torch.randperm(len(self.empty_locations))[:self.num_robots]]
        else:
            return self.starts
    
    def sample_targets(self, num, reached=None):
        if len(self.tasks)==0:
            idxs=torch.randint(0,len(self.empty_locations),(num,))
            return self.empty_locations[idxs]
        else:
            if reached is not None:
                self.tasks_completed+=reached
                agent_idxs=torch.nonzero(reached,as_tuple=True)[0]
                # print(agent_idxs)
                # roundrobin in LRR implementation
                task_idxs=(self.tasks_completed[agent_idxs]*self.num_robots+agent_idxs)%len(self.tasks)
                return self.tasks[task_idxs]
            else:
                # init targets
                return self.tasks[:self.num_robots]
    
    def get_observations(self):

        # TODO(rivers)：we will just implement several simple features. please refer to yutong's code for better designs. 
        # also refer to the paper: maybe we can add hints for the cost reduce of each action, the angle between the direction of goals, etc.
        # I believe they are also important.
        
        # # num_robots, FOV_height, FOV_width, 2
        local_views=self.curr_positions[:,None,None,:]+self.local_view_offsets
        offsetted_local_views=local_views+self.padded_graph_offsets
        
        # Feature 1: other static obstacles
        # num_robots, FOV_height, FOV_width
        # pad with 1 = obstacle
        
        local_static_obstacle_map=self.padded_graph[offsetted_local_views[...,0],offsetted_local_views[...,1]]
        
        # # Feature 2: other agents
        # # TODO: we can pre-allocate this
        global_agents_map=torch.zeros_like(self.padded_graph,dtype=torch.float32)
        global_agents_map[self.curr_positions[:,0]+self.FOV_height//2,self.curr_positions[:,1]+self.FOV_width//2]=self.priorities
        # num_robots, FOV_height, FOV_width
        local_agents_map=global_agents_map[offsetted_local_views[...,0],offsetted_local_views[...,1]]
        # # set its own position to 0
        # local_agents_map[:,self.FOV_height//2,self.FOV_width//2]=0
        # NOTE(rivers): >0 is a bug before. we fix it.
        _masks = local_agents_map<=0
        local_agents_map=torch.sign(local_agents_map-local_agents_map[:,self.FOV_height//2:self.FOV_height//2+1,self.FOV_width//2:self.FOV_width//2+1])
        local_agents_map[_masks]=0
        
        # NOTE(rivers): currently agents and targets have no association at all. this is something interesting to me.
        # global_targets_map=torch.zeros_like(self.padded_graph)
        # global_targets_map[self.target_positions[:,0]+self.FOV_height//2,self.target_positions[:,1]+self.FOV_width//2]=1
        # local_targets_map=global_targets_map[offsetted_local_views[...,0],offsetted_local_views[...,1]]
        
        # TODO: Feature 3: Heauristic Distance with cliping or maybe sigmoid? or normalized by the shortest distance?
        # num_robots, FOV_height, FOV_width
        heuristics_map, masks=self.heuristic_table.get_heuristics(local_views,offsetted_local_views,self.target_positions)
        
        if not self.use_rank_feats:
            # BUG(rivers): we should not use -1 in heuristics_map_1! because it can be negative and would cause confusion!
            # normalize by the local view
            heuristics_map_1=(heuristics_map-heuristics_map[:,self.FOV_height//2:self.FOV_height//2+1,self.FOV_width//2:self.FOV_width//2+1])/(self.FOV_height//2+self.FOV_width//2)*0.5
            # heuristics_map_1=torch.arctan(heuristics_map_1)*2/torch.pi
            heuristics_map_1[~masks]=-1
            heuristics_map_2=heuristics_map/(self.map.height+self.map.width)*0.5
            heuristics_map_2[~masks]=-1
        else:
            raise NotImplementedError
            # heuristics_map_2=heuristics_map/(self.map.height+self.map.width)*0.5
            # heuristics_map_2[~masks]=-1
            
            # # B*A, h*w
            # heuristics_map = heuristics_map.reshape(self.num_robots,-1)
            # # B*A, h*w
            # ranks = heuristics_map.argsort(dim=1).argsort(dim=1).float()
            # ranks = ranks.reshape(self.num_robots,self.FOV_height,self.FOV_width)
            # heuristics_map_1 = ranks/(self.FOV_height*self.FOV_width)
            # heuristics_map_1[~masks]=-1
            
        
        # TODO: we need a global view to provide information

        # normalize
        # num_robots, 1, 1
        # shortest_heuristics=self.heuristic_table.get(self.curr_positions[:,None,None,:],self.target_positions)
        # normalized_heuristic_map=heuristics_map/shortest_heuristics*0.5
        
        # Feature 4: location embedding
        
        # we can use stack
        observations=torch.zeros(size=(self.num_robots,self.num_channels,self.FOV_height,self.FOV_width),dtype=torch.float32,device=self.device)
        observations[:,0]=local_static_obstacle_map
        observations[:,1]=local_agents_map
        observations[:,2]=heuristics_map_1
        # TODO: we may use or may not use this feature
        observations[:,3]=heuristics_map_2
        # observations[:,4]=local_targets_map
        
        # TODO: whether this location is the target of another agent
        
        # location embedding
        # observations[:,4]=self.padded_graph_location_ys[offsetted_local_views[...,0],offsetted_local_views[...,1]]
        # observations[:,5]=self.padded_graph_location_xs[offsetted_local_views[...,0],offsetted_local_views[...,1]]
        
        # if len(self.history)==0:
        #     for i in range(self.history_len):
        #         self.history.append(observations[:,1].reshape(self.num_robots,-1))
                
        # self.history.pop(0)
        # self.history.append(observations[:,1].reshape(self.num_robots,-1))
        
        observations=observations.reshape(self.num_robots,-1)
        
        # edge_idices, edge_masks=self.get_neighbors(max_dist=self.FOV_height,num_closest=self.num_edges)
        # edge_idices=edge_idices.reshape(self.num_robots, self.num_edges*2)
        
        # steps=torch.full_like(self.priorities,fill_value=self.step_ctr/self.rollout_length,dtype=torch.float32,device=self.device).reshape(-1,1)
        
        # conflict_pairs=self.get_conflict_pairs(self.curr_positions)
        # conflict_pairs=conflict_pairs.reshape(self.num_robots,-1)
        
        # obs_neighboring_masks, act_neighboring_masks=self.get_neighboring_obs_act_masks(max_dist=self.FOV_height, num_closest=self.num_edges)
        # observations=torch.cat([observations, obs_neighboring_masks, act_neighboring_masks, self.priorities.unsqueeze(-1),self.curr_positions,self.target_positions],dim=-1)
        
        # observations=torch.cat([observations, self.priorities.unsqueeze(-1),self.curr_positions,self.target_positions],dim=-1)
        
        # observations=torch.cat(self.history[:-1]+[observations], dim=-1)

        observations=torch.cat([observations, self.perm_indices.unsqueeze(-1), self.reverse_perm_indices.unsqueeze(-1), self.priorities.unsqueeze(-1),self.curr_positions,self.target_positions],dim=-1)
        
        #observations=torch.cat([observations, edge_idices, edge_masks, self.priorities.unsqueeze(-1),self.curr_positions,self.target_positions],dim=-1)
        
        # we further build global observation here
        # for now, it will have the following 3 channels
        # 1. static obstacles
        # 2. agent start locations 0-1 encoded
        # 3. agent target locations 0-1 encoded
        # namely there is no association between agent starts and targets
        global_observations=torch.zeros(size=(1,self.num_global_channels,self.map.height,self.map.width),dtype=torch.float32,device=self.device)
        
        global_observations[0,0]=self.graph.float()
        global_observations[0,1,self.curr_positions[:,0],self.curr_positions[:,1]]=1
        global_observations[0,2,self.target_positions[:,0],self.target_positions[:,1]]=1        
        
        return observations, global_observations
    
    # Maybe we should have a BFS?
    # Maybe we want a best cover?
    # we may keep edge features
    def get_neighbors(self, max_dist, num_closest):
        # stategy 1: only those in the dist, closest 
        
        # l1-dist
        dists=torch.abs(self.curr_positions[:,None,:]-self.curr_positions[None,:,:]).sum(dim=-1)
        edge_dists, second_indices = torch.topk(dists,num_closest,largest=False)
        
        first_indices=torch.arange(len(self.curr_positions),dtype=torch.long,device=self.device).reshape(-1,1).repeat(1,num_closest)
        
        # num_robots, K
        edge_masks = edge_dists<=max_dist #torch.logical_and(edge_dists<=max_dist,edge_dists>0)
        
        # num_robots, K, 2
        edge_indices=torch.stack([first_indices, second_indices], dim=-1)
        
        return edge_indices, edge_masks
    
    def get_neighboring_obs_act_masks(self, max_dist, num_closest):
        obs_edge_indices, obs_edge_masks=self.get_neighbors(max_dist,num_closest)
        
        # num_robots*K, 2
        obs_edge_indices = obs_edge_indices.reshape(-1,2)
        # num_robots*K
        obs_edge_masks = obs_edge_masks.reshape(-1)
        obs_edge_indices = obs_edge_indices[obs_edge_masks]
        
        reverse_obs_edge_indices=obs_edge_indices[...,[1,0]]
        
        # observarion neighboring edges
        obs_edge_indices = torch.concat([obs_edge_indices,reverse_obs_edge_indices],dim=0)

        # False is not connected
        obs_neighboring_masks=torch.zeros([self.num_robots,self.num_robots], dtype=torch.bool, device=self.device)
        obs_neighboring_masks[obs_edge_indices[...,0],obs_edge_indices[...,1]]=True

        # only keep casual edges
        act_edge_masks = obs_edge_indices[...,0]>=obs_edge_indices[...,1]
        act_edge_indices = obs_edge_indices[act_edge_masks]
        
        act_neighboring_masks=torch.zeros([self.num_robots,self.num_robots], dtype=torch.bool, device=self.device)
        act_neighboring_masks[act_edge_indices[...,0],act_edge_indices[...,1]]=True
        
        # act_neighboring_masks = torch.tril(torch.ones(self.num_robots,self.num_robots, dtype=torch.bool, device=self.device))
        
        # assert self.permute
        # if self.permute:
        #     # NOTE: we only permute columns here, rows will be permuted with other observations later
        #     obs_neighboring_masks=obs_neighboring_masks[...,self.perm_indices]
        #     act_neighboring_masks=act_neighboring_masks[...,self.perm_indices]
            
        return obs_neighboring_masks, act_neighboring_masks
    
    def get_conflict_pairs(self, positions):
        
        # NOTE: other agents' positions-this agent's position
        diff=(positions[None,:,:]-positions[:,None,:]).cpu().numpy()
        
        # TODO: need to double check
        conflict_pairs=[]
        for a2 in range(self.num_robots):
            for a1 in range(self.num_robots):
                dy,dx=diff[a2,a1]
                if (dy,dx) in self.conflict_action_table:
                    conflict_actions=self.conflict_action_table[(dy,dx)]
                    # TODO: we could remove invalid actions here
                    for act1,act2 in conflict_actions:
                        conflict_pairs.append((a1,act1,a2,act2))


        max_conflict_pairs=self.num_conflict_actions*self.num_robots
        conflict_pair_masks=torch.zeros(size=(max_conflict_pairs,1),dtype=torch.int32,device=self.device)
        
        conflict_pairs=torch.tensor(conflict_pairs,device=self.device,dtype=torch.int32) 
        conflict_pair_masks[:len(conflict_pairs)]=1
        padding_pairs=torch.zeros(size=(max_conflict_pairs-len(conflict_pairs),4),dtype=torch.int32,device=self.device)
        conflict_pairs=torch.concat([conflict_pairs,padding_pairs],dim=0)
        
        conflict_pairs=torch.concat([conflict_pairs,conflict_pair_masks],dim=-1)
        
        return conflict_pairs
    
    def get_action_masks(self):
        '''
        1 - valid action
        0 - invalid action
        '''
        # shape: [num_robots,action_dim,2]
        next_positions=self.curr_positions[:,None,:]+self.movements[None,:,:]+self.padded_graph_offsets
        
        # check if out of bound and if collide with static obstacles
        # 0-valid, 1-invalid
        action_masks=self.padded_graph[next_positions[...,0],next_positions[...,1]]
        
        # 1-valid, 0-invalid
        action_masks=1-action_masks.float()

        return action_masks

    def get_rewards(self, prev_distances, curr_distances, reached):
        
        # TODO depends on heuristics
        dist_reward=1.0
        reached_reward=0.0
        
        individual_rewards=(prev_distances-curr_distances)*dist_reward-1
        
        team_rewards_coef=1.0
        individual_rewards_coef=1.0
        
        team_rewards=self.get_team_rewards(individual_rewards)
        
        #print(team_rewards.std(),team_rewards.min(),team_rewards.max(),team_rewards.size())
        
        rewards=individual_rewards*individual_rewards_coef+team_rewards_coef*team_rewards+reached.float()*reached_reward
        
        return rewards, individual_rewards, team_rewards
    
    def connected_componenets_bfs_search(self, positions):
        agent_map=torch.full_like(self.graph,fill_value=-1,dtype=torch.float32,device=self.device)
        agent_map[positions[:,0],positions[:,1]]=torch.arange(len(positions),device=self.device)
        visited=torch.zeros((self.num_robots,),dtype=bool,device=self.device)
        components=[]
        for i in range(self.num_robots):
            if not visited[i]:
                component=self.bfs_search(i,visited)
                components.append(component)
        return components
    
    def bfs_search(self, i, visited, positions, agent_map):
        component=[]
        q=queue.Queue()
        q.put(i)
        visited[i]=True
        component.append(i)
        while not q.empty():
            agent_idx=q.get()
            # for each neighbor
            for movement in self.movements:
                next_position=positions[agent_idx]+movement
                next_agent_idx=agent_map[next_position[0],next_position[1]]
                if next_agent_idx>=0 and not visited[next_agent_idx]:
                    visited[next_agent_idx]=True
                    component.append(next_agent_idx)
                    q.put(next_agent_idx)
                       
        return component
    
    def get_team_rewards(self, rewards):
        
        reward_map=torch.zeros_like(self.padded_graph,dtype=torch.float32,device=self.device)
        agent_map=torch.zeros_like(self.padded_graph,dtype=torch.float32,device=self.device)
        
        offseted_curr_positions=self.curr_positions+self.padded_graph_offsets
        
        reward_map[offseted_curr_positions[:,0],offseted_curr_positions[:,1]]=rewards
        agent_map[offseted_curr_positions[:,0],offseted_curr_positions[:,1]]=1
        
        kernel=torch.ones((3,3),dtype=torch.float32,device=self.device)
        
        assert kernel.shape[0]==kernel.shape[1] and kernel.shape[0]%2==1 and kernel.shape[1]%2==1
        kernel[kernel.shape[0]//2,kernel.shape[1]//2]=0
        
        with torch.no_grad():
            team_rewards = F.conv2d(reward_map.reshape(1,1,*reward_map.shape), kernel.reshape(1,1,*kernel.shape), padding=kernel.shape[0]//2)
            num_agents = F.conv2d(agent_map.reshape(1,1,*reward_map.shape), kernel.reshape(1,1,*kernel.shape), padding=kernel.shape[0]//2)
        
        team_rewards = team_rewards.reshape(*reward_map.shape)
        num_agents = num_agents.reshape(*agent_map.shape)
        
        team_rewards = team_rewards[offseted_curr_positions[:,0],offseted_curr_positions[:,1]]
        num_agents = num_agents[offseted_curr_positions[:,0],offseted_curr_positions[:,1]]
        
        # average moving speed
        team_rewards = (team_rewards) / (num_agents+1e-9)
                
        return team_rewards
    
    def render(self):
        pass
    
    def get_observations_and_rewards_for_GGO(self):
        '''
        This function will only be called after the evironment ends and used for GGO optimization
        '''
        
        # algorithm
        # MAPPO
        # TODO: should we consider DDPG?
        
        # features
        # 1. map
        # 2. vertex usages
        # 3. edges usages
        
        # action spaces
        # predict delta or the eventual value?
        
        # rewards
        # 1. mean total individual rewards
        # 2. TODO: mean local region rewards
        
        # normalize vertex usages
        self.vertex_usages=self.vertex_usages/self.step_ctr
        self.edge_usages=self.edge_usages/self.step_ctr
        
        observations=torch.stack([map.graph.float(),self.vertex_usages,self.edge_usages],dim=0)
        
        return {   
        }

@registry.registered(registry.ENV, "LMAPF")
class MultiLMAPFEnv(BaseEnv):
    def __init__(self, id, seed, cfg , device=None, map_filter_keep=None, map_filter_remove=None, precompute_HT=True):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        super().__init__(id, seed)
        self.cfg=cfg
        self.seed=seed
        
        if device is None:
            self.device=self.cfg["device"]
        else:
            self.device=device
        self.rollout_length=self.cfg["rollout_length"]
        self.gae_gamma=self.cfg["gae_gamma"]
        self.gae_lambda=self.cfg["gae_lambda"]
        self.mappo_reward=self.cfg["mappo_reward"]
        self.precompute_HT=precompute_HT
        
        self.map_manager = MapManager(map_filter_keep, map_filter_remove)
        
        instances = self.cfg["instances"]
        if instances:
            for instance in instances:
                map_path=instance["map_path"]
                agent_bins=instance["agent_bins"]
                m=Map(map_path,agent_bins)
                self.map_manager.add_map(m)
        else:
            learn_to_follow_maps_path = self.cfg["learn_to_follow_maps_path"]
            agent_bins = self.cfg["agent_bins"]
            if learn_to_follow_maps_path:
                Logger.warning("load maps from {}".format(learn_to_follow_maps_path))
                self.map_manager.load_learn_to_follow_maps(learn_to_follow_maps_path, agent_bins)
            else:
                map_path = self.cfg["map_path"]
                num_robots=self.cfg["num_robots"]
                m=Map(map_path,[num_robots])
                self.map_manager.add_map(m)
            
        # (map_name, num_agents): env
        self.envs=defaultdict(None)
        
        # (num_agents, map_h, map_w): [map_name]
        self.instance_groups=defaultdict(list)
        
        for name, num_robots in self.map_manager.instances_list:
            m: Map = self.map_manager[name]
            self.instance_groups[(num_robots, m.height, m.width)].append(name)

        self.group_infos=list(self.instance_groups.keys())
        
        self.set_curr_env(0,verbose=False)
        
    def set_curr_env2(self, map_name, num_robots, verbose=False):
        if (map_name,num_robots) not in self.envs:
            env=LMAPFEnv(
                id = "{}_{}_{}".format(self.id, map_name, num_robots),
                seed = self.seed,
                map = self.map_manager[map_name],
                num_robots = num_robots,
                device = self.device,
                cfg = {
                    "rollout_length": self.rollout_length,
                    "device": self.device,
                    "gae_gamma": self.gae_gamma,
                    "gae_lambda": self.gae_lambda,
                    "mappo_reward": self.mappo_reward,
                    "WPPL": self.cfg["WPPL"],
                    "map_weights_path": self.cfg["map_weights_path"],
                    "use_rank_feats": self.cfg.get("use_rank_feats",False)
                },
                precompute_HT=self.precompute_HT
            )
            self.envs[(map_name,num_robots)]=env

        if verbose:
            Logger.info("Env {} set curr env to map {} with {} agents".format(self.id, map_name,num_robots))
        self.curr_env=self.envs[(map_name,num_robots)]

    def set_curr_env(self, idx=None, verbose=False):
        if idx is None:
            idx = np.random.randint(0, np.iinfo(np.int32).max)
        
        # TODO(rivers): should not use constant here
        group_idx = idx
        instance_idx = np.random.randint(0, np.iinfo(np.int32).max)
        
        group_info = self.group_infos[group_idx%len(self.group_infos)]
        group = self.instance_groups[group_info]
        map_name = group[instance_idx%len(group)]
        num_robots = group_info[0]
        self.set_curr_env2(map_name, num_robots, verbose)
    
    def reset(self, *args, **kwargs):
        return self.curr_env.reset(*args, **kwargs)
        
    def step(self, *args, **kwargs):
        return self.curr_env.step(*args, **kwargs)
        
    def is_terminated(self):
        return self.curr_env.is_terminated()
    
    def get_episode_stats(self):
        return self.curr_env.get_episode_stats()
    
    # TODO: set_xxx must be implemented to take effects for all envs
    
    def __getattr__(self, key: str):
        try:
            return self.curr_env.__getattribute__(key)
        except AttributeError:
            return self.curr_env.__getattr__(key)
