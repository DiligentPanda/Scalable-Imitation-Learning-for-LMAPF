import torch
import numpy as np
from ._env import LMAPFEnv
from light_malib.registry import registry
from ..base_env import BaseEnv
from .map import Map, MapManager, ProgressManager, Progress
from light_malib.utils.logger import Logger
from light_malib.utils.logger import Logger
from collections import defaultdict

@registry.registered(registry.ENV, "LMAPF")
class MultiLMAPFEnv(BaseEnv):
    '''
    LMAPFEnv only supports one map at a time, this class is used to manage multiple maps.
    '''
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
        self.curr_env:LMAPFEnv=self.envs[(map_name,num_robots)]

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
