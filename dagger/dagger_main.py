'''
We directly implement Dagger here using multiprocessing.
''' 
from light_malib.utils.cfg import load_cfg
from light_malib.envs.LMAPF.env import MultiLMAPFEnv
import time
from bc_trainer import BCTrainer
from light_malib.buffer.data_server import DataServer

config_path = "expr_configs/LMAPF/Dagger/random_32_32_400agents.yaml" 
cfg = load_cfg(config_path)

num_iters=cfg.framework.max_dagger_iterations
max_episodes_per_iter=cfg.framework.max_episodes_per_iter


data_server=DataServer("DataServer", )


seed=int(int(time.time()))
env=MultiLMAPFEnv(0, seed, cfg.rollout_manager.worker.envs[0], "cpu")
import copy
table_cfg=copy.deepcopy(cfg.data_server.table_cfg)
table_cfg.update({"capacity":2**20,"sampler_type":"uniform","sample_max_usage": 1e8, "rate_limiter_cfg":{}})
for map_name,num_agents in env.map_manager.instances_list:
    table_name="imitation_{}_{}".format(map_name,num_agents)
    # data_server.create_table.remote(table_name, table_cfg)


bc_trainer=BCTrainer()

for iter_idx in range(num_iters):
    # collect data
    
    # push to data server
    
    # bc
    bc_trainer.train()