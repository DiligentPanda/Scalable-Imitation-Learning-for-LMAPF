from light_malib.envs.LMAPF.env import MultiLMAPFEnv
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.algorithm.mappo.policy import shape_adjusting
import numpy as np
from light_malib.utils.episode import EpisodeKey
import copy
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer
import torch

# WPPL is policy wrapper implementing compute_action
# TODO: maybe we should insert WPPL into the model
class WPPL:
    def __init__(self, 
                 seed, 
                 real_env: MultiLMAPFEnv,
                 env_cfg, 
                 window_size, 
                 num_threads=1, 
                 max_iterations=-1, 
                 time_limit=1.0, 
                 time_limit_all=True, 
                 env_pibt_mode="guard",
                 device=None,
                 verbose=False
                 ):
        self.seed=seed
        self.window_size=window_size
        self.num_threads=num_threads
        self.max_iterations=max_iterations
        self.time_limit=time_limit
        self.time_limit_all=time_limit_all
        self.device=device
        self.verbose=verbose
        self.real_env=real_env
        
        # this env is for internal simulation
        self.env_cfg=copy.deepcopy(env_cfg)
        self.env=MultiLMAPFEnv("WPPL_{}".format(seed), seed, self.env_cfg, device=self.real_env.device, precompute_HT=False)
        self.env.check_valid(False)
        self.env.enable_log(False)
        # we don't want to sample new target positions, because we don't have ideas about the new target positions
        # TODO(rivers): we can do if ...
        self.env_pibt_mode=env_pibt_mode
        
        self._policy=None
        self._rollout_func=None
        self._pack_episode_func=None
        
    def set_policy(self, policy):
        self._policy=policy
        
    def set_rollout_func(self, rollout_func):
        self._rollout_func=rollout_func
    
    def set_pack_episode_func(self, pack_episode_func):
        self._pack_episode_func=pack_episode_func
    
    def __getattr__(self, key: str):
        try:
            return self.__getattribute__(key)
        except AttributeError:
            return self._policy.__getattribute__(key)
    
    def solve(
        self,
        curr_positions,    
        target_positions,
        priorities,
        eval
    ):
        
        assert len(curr_positions.shape)==2 and curr_positions.shape[1]==2 and target_positions.shape[0]==self.env.num_robots
        assert len(target_positions.shape)==2 and target_positions.shape[1]==2 and target_positions.shape[0]==self.env.num_robots
        
        custom_reset_config ={
            "curr_positions": curr_positions,
            "target_positions": target_positions,
            "priorities": priorities
        }
        
        agent = "agent_0"
        policy_id = "policy_0"
        behavior_policies = {
            agent : (policy_id, self._policy),
        }    
        
        global_timer.record("init_s")
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        rollout_desc = RolloutDesc(0, agent, None, None, None, None, None)
        rollout_results = self._rollout_func(
            eval=eval,
            rollout_worker=None,
            rollout_desc=rollout_desc,
            env=self.env,
            behavior_policies=behavior_policies,
            data_server=None,
            rollout_length=self.window_size,
            render=False,
            device=self.device,
            **custom_reset_config
            # rollout_epoch = 100,
        )
        global_timer.time("init_s","init_e","init")
        
        # num_robots, rollout_length
        init_paths=rollout_results["paths"]
        first_step_policy_output=rollout_results["first_step_policy_output"]
        step_data_list=rollout_results["episode"]

        # init_actions=[]
        # for i in range(self.env.num_robots):
        #     curr_loc=init_paths[i*(self.window_size+1)+0]
        #     next_loc=init_paths[i*(self.window_size+1)+1]
        #     curr_y=curr_loc//self.env.map.width
        #     curr_x=curr_loc%self.env.map.width
        #     next_y=next_loc//self.env.map.width
        #     next_x=next_loc%self.env.map.width
        #     delta_y=next_y-curr_y
        #     delta_x=next_x-curr_x
        #     for idx, movement in enumerate(self.env.movements):
        #         if delta_y==movement[0] and delta_x==movement[1]:
        #             init_actions.append(idx)
        #             break
            
        # init_actions=np.array(init_actions,dtype=int)
        
        #return init_actions
        
        # print(curr_positions)
        # print(target_positions)
        # print(init_paths)

        # call PLNSSolver
        start_locations=curr_positions[...,0]*self.env.map.width+curr_positions[...,1]
        start_locations=start_locations.cpu().numpy().tolist()
        goal_locations=target_positions[...,0]*self.env.map.width+target_positions[...,1]
        goal_locations=goal_locations.cpu().numpy().tolist()
        
        if self.time_limit_all:
            pass
        
        global_timer.record("lns_s")
        actions, total_delays=self.env.PLNSSolver.solve(
            start_locations,
            goal_locations,
            init_paths,
            self.action_choices,
            self.time_limit
        )       
        global_timer.time("lns_s","lns_e","lns")
        
        actions=np.array(actions,dtype=int)
        if not eval:
            last_step_data={
                rollout_desc.agent_id: step_data_list[-1]
            }
            step_data_list=step_data_list[:-1]
            
            for step_data in step_data_list:
                if EpisodeKey.REWARD in step_data:
                    step_data[EpisodeKey.REWARD][:]/=(self.window_size*10)
            
            # TODO: should add step as feature?
            step_data_list[-1][EpisodeKey.REWARD][:]=total_delays/self.window_size/self.env.num_robots
            step_data_list[-1][EpisodeKey.DONE][:]=False
            
            episode=self._pack_episode_func(step_data_list, last_step_data, rollout_desc, gae_gamma=self.env.gae_gamma, gae_lambda=self.env.gae_lambda)
        else:
            episode=None
            
        return actions, first_step_policy_output, episode
    
    @shape_adjusting
    def compute_action(self, **kwargs):
        
        map_name = kwargs["map_name"]
        num_robots = kwargs["num_robots"]
        
        self.real_env.set_curr_env2(map_name, num_robots, False)
        self.env.set_curr_env2(map_name, num_robots, False)
        self.env.set_HT(self.real_env.get_HT())
        self.env.set_PLNSSolver(self.real_env.get_PLNSSolver())
        self.env.set_one_shot(True)
        self.env.set_pibt_func(self.env_pibt_mode)
        self.env.set_rollout_length(self.window_size)
        self.action_choices=self.env.movements.reshape(-1).cpu().numpy().tolist()
        
        # weird but fine
        eval=not kwargs["explore"]
        
        # batch_size*num_robots, num_feats
        observations = kwargs[EpisodeKey.CUR_OBS]
        
        assert len(observations.shape)==2 and observations.shape[0]==self.env.num_robots, "only support batch_size=1"
        
        target_positions=observations[...,-2:].long()
        curr_positions=observations[...,-4:-2].long()
        # fix a bug, dtype should float for priorities
        priorities=observations[...,-5].float()
        # we need to add permutation here
        if self.env.use_permutation:
            reverse_perm_indices=observations[...,-6].long()
            target_positions=target_positions[reverse_perm_indices]
            curr_positions=curr_positions[reverse_perm_indices]
            priorities=priorities[reverse_perm_indices]        
        
        # batch_size*num_robots
        global_timer.record("solve_s")
        actions, first_step_policy_output, episode=self.solve(curr_positions, target_positions, priorities, eval)
        global_timer.time("solve_s","solve_e","solve")
        
        # print(global_timer.mean_elapses)
        # global_timer.clear()
        
        # print(actions)
        
        # print(ret.keys())
        # PIBT output
        first_step_policy_output[EpisodeKey.INIT_ACTION]=first_step_policy_output[EpisodeKey.ACTION]
        # LNS output
        first_step_policy_output[EpisodeKey.ACTION]=actions
        first_step_policy_output[EpisodeKey.EPISODE]=episode
        
        return first_step_policy_output
        
        