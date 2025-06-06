from .map import Map, MapManager, ProgressManager, Progress
import torch

from light_malib.utils.episode import EpisodeKey
from ..base_env import BaseEnv
import numpy as np
from light_malib.utils.distributed import get_actor
from light_malib.utils.logger import Logger
import sys 
import torch.nn.functional as F
import queue
from light_malib.utils.timer import global_timer
from .heuristic_table import HeuristicTable
from .episode_log import EpisodeLog

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
        
        # we add FOV_height//2 because of the padding
        offsets_y=torch.arange(-(self.FOV_height//2),(self.FOV_height+1)//2,dtype=torch.int32,device=self.device)
        offsets_x=torch.arange(-(self.FOV_width//2),(self.FOV_width+1)//2,dtype=torch.int32,device=self.device)
        # FOV_height, FOV_width, 2
        self.local_view_offsets=torch.stack(torch.meshgrid([offsets_y,offsets_x],indexing="ij"),dim=-1)
        self.padded_graph_offsets=torch.tensor([self.FOV_height//2,self.FOV_width//2],dtype=torch.int32,device=self.device)
    
        # legacy code?
        self.team_sizes = {          
            self.agent_id: self.num_robots
        }
        
        # 1 - obstacle, 0 - empty location
        self.graph=torch.tensor(self.map.graph,device=self.device,dtype=torch.int32)
        self.empty_locations=torch.nonzero(self.graph==0).type(torch.int32)
        self.padded_graph=torch.nn.functional.pad(self.graph,(self.FOV_height//2,self.FOV_height//2,self.FOV_width//2,self.FOV_width//2),mode='constant',value=1)
        
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
    
        self.starts=[]
        self.tasks=[]

    
        sys.path.insert(0,"lmapf_lib/MAPFCompetition2023/build")
        import py_PIBT
        self.PIBTSolver=py_PIBT.PIBTSolver(seed)
        
        self._initialized=False
        
        self.prev_step_data = None
        
        # NOTE: if one_shot is set to True, agents' goals will be unchanged.
        self._one_shot=False
        self._pibt_func="guard"
        self._eval=True
        
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
            sampling=False
        else:
            sampling=True
            masks=self.padded_graph[offseted_local_views[...,0],offseted_local_views[...,1]]==0
            # heuristics is logits here.
            heuristics=torch.softmax(heuristics,dim=-1)
            heuristics[~masks]=-1
                
        if guiding_actions is not None:
            if torch.any(heuristics[torch.arange(len(guiding_actions),dtype=torch.int32,device=self.device),guiding_actions]<0):
                Logger.error("err! this may sometimes happen if the probability is not safely clipped above 1e-9, for example.")
            # TODO: set to 0 has the potential bug when agent will arrive the goal?
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
        # legacy code
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
    
        # statistics
        self.valid_ctr=0  
        self.action_consistent_rate=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        self.total_rewards=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        self.total_individual_rewards=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        self.total_team_rewards=torch.zeros(size=(self.num_robots,),dtype=torch.float32,device=self.device)
        self.total_completed_tasks=torch.zeros_like(self.total_rewards)
        self.action_cnts=torch.zeros(size=(self.num_robots,self.action_dim),dtype=torch.float32,device=self.device)

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
        
        # TODO: why set reached to done doesn't work
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
        # pad with 1, meaining obstacle
        local_static_obstacle_map=self.padded_graph[offsetted_local_views[...,0],offsetted_local_views[...,1]]
        
        # Feature 2: other agents
        global_agents_map=torch.zeros_like(self.padded_graph,dtype=torch.float32)
        global_agents_map[self.curr_positions[:,0]+self.FOV_height//2,self.curr_positions[:,1]+self.FOV_width//2]=self.priorities
        # num_robots, FOV_height, FOV_width
        local_agents_map=global_agents_map[offsetted_local_views[...,0],offsetted_local_views[...,1]]
        # set its own position to 0
        # local_agents_map[:,self.FOV_height//2,self.FOV_width//2]=0
        # NOTE(rivers): >0 is a bug before. we fix it.
        _masks = local_agents_map<=0
        local_agents_map=torch.sign(local_agents_map-local_agents_map[:,self.FOV_height//2:self.FOV_height//2+1,self.FOV_width//2:self.FOV_width//2+1])
        local_agents_map[_masks]=0
        
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
        
        # Feature 4: location embedding
        # we can use stacking rather than assigning here
        observations=torch.zeros(size=(self.num_robots,self.num_channels,self.FOV_height,self.FOV_width),dtype=torch.float32,device=self.device)
        observations[:,0]=local_static_obstacle_map
        observations[:,1]=local_agents_map
        observations[:,2]=heuristics_map_1
        # TODO: we may use or may not use this feature
        observations[:,3]=heuristics_map_2
        # observations[:,4]=local_targets_map
        
        observations=observations.reshape(self.num_robots,-1)
        observations=torch.cat([observations, self.perm_indices.unsqueeze(-1), self.reverse_perm_indices.unsqueeze(-1), self.priorities.unsqueeze(-1),self.curr_positions,self.target_positions],dim=-1)
                
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