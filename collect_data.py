from light_malib.envs.LMAPF.env import MultiLMAPFEnv
import numpy as np
from torch.distributions import Categorical
from light_malib.utils.cfg import load_cfg
from light_malib.rollout.rollout_func_LMAPF import rollout_func, rollout_func_for_WPPL
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.envs.LMAPF.WPPL import WPPL
from light_malib.utils.logger import Logger
from torch.multiprocessing import Pool
import torch.multiprocessing as multiprocessing
import time
import h5py
from torch.cuda.amp.autocast_mode import autocast
import os
import torch

database_fp = "datasets/paper_exps/room_a1000_r512_e128_PIBT_RL_LNS_BC_i0.h5"
config_path = "expr_configs/paper_exps/room/Bootstrap/bootstrap_iter1_room_a1000_r512.yaml"
model_path = "logs/LMAPF/paper_exps_basic_cnn_re_embed_perm5_bn_0block_room_a1000_RL_iter0/2024-06-19-09-07-52/agent_0/agent_0-default-1/best" # you need a dummy one now
WPPL_mode="PIBT-RL-LNS"
rollout_length=None
max_iterations=None
num_threads=None
collect_log=False
check_valid=False

num_devices=4
num_processes=64
num_episode_per_instance=128
chunk_size=32 # commit every chunk size
# TODO: we should compute num_episode_per_instance

# should not use the map filters in general. it is only for convienence
map_filter_keep=None#["mazes-s17_wc6_od55"]#["test-mazes-s41_wc5_od50","test-mazes-s45_wc4_od55"]#None
# interesting: not good at the following two maps
map_filter_remove=None #["test-mazes-s41_wc5_od50","test-mazes-s45_wc4_od55"]

x = input("are you sure to collect data and overwrite {}?".format(database_fp))
if x.lower() not in ['y',"yes"]:
    print("Cancelled")
    exit(0)

seed=int(int(time.time()))
cfg=load_cfg(config_path)
cfg.rollout_manager.worker.envs[0].WPPL.mode=WPPL_mode
if rollout_length is not None:
    cfg.rollout_manager.worker.envs[0].rollout_length=rollout_length
if max_iterations is not None:
    cfg.rollout_manager.worker.envs[0].WPPL.max_iterations=max_iterations
if num_threads is not None:
    cfg.rollout_manager.worker.envs[0].WPPL.num_threads=num_threads
env=MultiLMAPFEnv("global",seed,cfg.rollout_manager.worker.envs[0],"cpu",map_filter_keep,map_filter_remove)

def get_model_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize():
    global env
    global policy
    global device
    
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    
    current = multiprocessing.current_process()
    worker_id=current._identity[0]
    if num_devices!=1:
        device="cuda:{}".format(worker_id%num_devices)
    else:
        device="cuda"
        
    Logger.info("worker {} using device {}".format(worker_id,device))

    policy=MAPPO.load(model_path, env_agent_id="agent_0")
    policy=policy.to_device(device)
    Logger.warning("#params of actor is {}".format(get_model_num_params(policy.actor)))
    Logger.warning("#params of critic is {}".format(get_model_num_params(policy.critic)))
    Logger.info("policy generated")
    
    seed = np.random.randint(0,np.iinfo(np.int32).max)+worker_id
    env=MultiLMAPFEnv(worker_id, seed, cfg.rollout_manager.worker.envs[0], device, map_filter_keep, map_filter_remove)
    env.check_valid(check_valid)
    env.enable_log(collect_log)
    Logger.info("env generated")


def evaluate(exp_arg):
    global env
    global policy
    global device
    
    # current = multiprocessing.current_process()
    # worker_id=current._identity[0]
    
    # device="cuda:{}".format(worker_id%num_devices)
    
    # Logger.info("worker {} using device {}".format(worker_id,device))
    
    idx, config_path, model_path, seed, map_name, num_agents = exp_arg
    Logger.info("Start rollout {}".format(idx))
    # cfg = load_cfg(config_path)

    # policy_id = "policy_0"
    # policy=MAPPO.load(model_path, env_agent_id="agent_0")
    # policy=policy.to_device(device)
    
    s=time.time()
    rollout_desc = RolloutDesc(idx, "agent_0", None, None, None, None, None)
    # env=MultiLMAPFEnv(0, seed, cfg.rollout_manager.worker.envs[0], device)
    
    agent = "agent_0"
    policy_id = "policy_0"
    behavior_policies = {
        agent: (policy_id, policy),
    }
    
    with autocast():
        episode_length=cfg.rollout_manager.worker.envs[0].rollout_length
        rollout_results = rollout_func(
            eval=True,
            rollout_worker=None,
            rollout_desc=rollout_desc,
            env=env,
            behavior_policies=behavior_policies,
            data_server=None,
            rollout_length=episode_length,
            render=False,
            verbose=False,
            collect_data=True,
            device=device,
            instance=(map_name, num_agents)
            # rollout_epoch = 100,
        )
    Logger.info("exp {}'s results: {}".format(idx, rollout_results["results"]))
    elapse=time.time()-s
    rollout_results["elapse"]=elapse/episode_length
    return rollout_results

all_througputs=[]
all_mean_step_times=[]
try:
    f=h5py.File(database_fp,'w')
    group_names=["meta","curr_positions","target_positions","priorities","actions"]  
    for group_name in group_names:
        f.create_group(group_name)
    f["meta"].attrs["num_chunks"]=0
    f["meta"].attrs["num_episodes"]=0
    
    pool=Pool(num_processes,initializer=initialize)

    np.random.seed(seed)
    num_agents2map_names={}
    for map_name,num_agents in env.map_manager.instances_list:
        if num_agents not in num_agents2map_names:
            num_agents2map_names[num_agents]=[]
        num_agents2map_names[num_agents].append(map_name)
    
    for num_agents,map_names in num_agents2map_names.items():
        exp_args=[]
        for jdx in range(num_episode_per_instance):
            for map_name in map_names:
                seed=np.random.randint(0, np.iinfo(np.int32).max)
                exp_args.append((jdx, config_path, model_path, seed, map_name, num_agents))
        num_episodes=len(exp_args)
        Logger.warning("total exps: {} for {} instances".format(len(exp_args),len(map_names)))

        results=pool.imap_unordered(evaluate, exp_args)
        
        map_names=[]
        num_robots=[]
        curr_positions=[]
        target_positions=[]
        priorties=[]
        actions=[]
        throughputs=[]
        mean_step_times=[]

        for idx,result in enumerate(results):
            map_names.append(result["imitation_data"]["map_name"].encode("ascii", "ignore"))
            num_robots.append(result["imitation_data"]["num_robots"])
            throughputs.append(result["results"][0]["stats"]["agent_0"]["throughput"].cpu().numpy())
            mean_step_times.append(result["elapse"])
            curr_positions.append(result["imitation_data"]["curr_positions"])
            target_positions.append(result["imitation_data"]["target_positions"])
            priorties.append(result["imitation_data"]["priorities"])
            actions.append(result["imitation_data"]["actions"])

            all_througputs.append(result["results"][0]["stats"]["agent_0"]["throughput"].cpu().numpy())
            all_mean_step_times.append(result["elapse"])
            
            if (idx+1)%32==0:
                Logger.info("complete {}/{}".format(idx+1,num_episodes))
                Logger.info("mean throughput: {}".format(np.mean(all_througputs)))
                mean_step_time=np.mean(all_mean_step_times)
                Logger.info("mean step time: {}".format(mean_step_time))
            
            
            if (idx+1)%chunk_size==0 or idx==num_episodes-1:
                chunk_idx=idx//chunk_size
    
                dataset_name="chunk_{}".format(chunk_idx)
                f.create_dataset("map_names/{}".format(dataset_name), data=np.array(map_names))
                f.create_dataset("num_robots/{}".format(dataset_name), data=np.array(num_robots))
                f.create_dataset("throughput/{}".format(dataset_name),data=np.array(throughputs))
                f.create_dataset("mean_step_time/{}".format(dataset_name),data=np.array(mean_step_times))
                f.create_dataset("curr_positions/{}".format(dataset_name), data=np.array(curr_positions))
                f.create_dataset("target_positions/{}".format(dataset_name), data=np.array(target_positions))
                f.create_dataset("priorties/{}".format(dataset_name), data=np.array(priorties))
                f.create_dataset("actions/{}".format(dataset_name), data=np.array(actions))
                
                Logger.info("complete {}/{}".format(idx+1,num_episodes))
                Logger.info("mean throughput: {}".format(np.mean(all_througputs)))
                mean_step_time=np.mean(all_mean_step_times)
                Logger.info("mean step time: {}".format(mean_step_time))
                f["meta"].attrs["num_chunks"]+=1
                f["meta"].attrs["num_episodes"]+=len(priorties)
                f.flush()
                
                curr_positions=[]
                target_positions=[]
                priorties=[]
                actions=[]
                map_names=[]
                num_robots=[]
                throughputs=[]
                mean_step_times=[]
                
    f["meta"].attrs["mean_throughput"]=np.mean(all_througputs)
    f["meta"].attrs["mean_step_time"]=mean_step_time
except Exception as e:
    import traceback
    Logger.error("error: {}".format(e))
    Logger.error(traceback.format_exc())
finally:    
    f.close()
    Logger.info("mean throughput: {}".format(np.mean(all_througputs)))
    mean_step_time=np.mean(all_mean_step_times)
    Logger.info("mean step time: {}".format(mean_step_time))