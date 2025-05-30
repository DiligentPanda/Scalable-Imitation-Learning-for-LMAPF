from light_malib.envs.LMAPF.env import MultiLMAPFEnv
from light_malib.utils.logger import Logger
import numpy as np
from torch.distributions import Categorical
from light_malib.utils.cfg import load_cfg
from light_malib.rollout.rollout_func_LMAPF import rollout_func, rollout_func_for_WPPL
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.envs.LMAPF.WPPL import WPPL
from light_malib.utils.timer import global_timer
import torch
from torch.multiprocessing import Pool
import time
import torch.multiprocessing as multiprocessing
import os
# import threading
import argparse

import json
from default_configs import default_configs

import sys

# https://github.com/pytorch/pytorch/issues/82843
multiprocessing.set_start_method("spawn", force=True)

arg_parser = argparse.ArgumentParser(description="Evaluate SILLM performance on LMAPF tasks.")
arg_parser.add_argument("--output_folder", type=str, default="exp", help="Output folder to save the results.")
arg_parser.add_argument("--exp_name", type=str, help="Experiment name.")
arg_parser.add_argument("--model_path", type=str, help="Path to the trained model.")
arg_parser.add_argument("--WPPL_mode", type=str, help="WPPL mode to use.", choices=["PIBT","PIBT-RL","PIBT-LNS","PIBT-RL-LNS"])
arg_parser.add_argument("--num_agents", type=str, help="Number of agents in the environment.")
arg_parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use for evaluation.")
arg_parser.add_argument("--num_devices", type=int, default=1, help="Number of GPU devices to use for evaluation.")
arg_parser.add_argument("--num_episodes_per_instance", type=int, default=8, help="Number of episodes to run per instance.")
arg_parser.add_argument("--rollout_length", type=int, default=None, help="Simulation steps of each test. If None, use the default length from the config.")
arg_parser.add_argument("--LNS_max_iterations", type=int, default=5000*8, help="Maximum iterations for LNS.")
arg_parser.add_argument("--LNS_num_threads", type=int, default=8, help="Number of threads to use for LNS.")
arg_parser.add_argument("--collect_log", action="store_true", help="Whether to collect logs during evaluation.")
arg_parser.add_argument("--check_valid", action="store_true", help="Whether to check the validity of the movement each step.")

args = arg_parser.parse_args()

output_folder = args.output_folder

exp_name = args.exp_name
model_path = args.model_path
WPPL_mode = args.WPPL_mode
num_agents = args.num_agents

config_path = default_configs[exp_name]["config_path"].replace("NUM_AGENTS", num_agents)

max_iterations=args.LNS_max_iterations # 5000
num_threads=args.LNS_num_threads # 1

rollout_length=args.rollout_length

collect_log=False
check_valid=False

# if use predefined data, please be careful with the number of episodes
predefined_starts_indexed=True
predefined_starts_template= default_configs[exp_name]["predefined_starts_template"].replace("NUM_AGENTS", num_agents) #"/root/GRF_MARL/lmapf_lib/MAPFCompetition2023/our_problems/maze.domain/agents/maze_1_256_20000_0_rs19.agents"#"lmapf_lib/Guided-PIBT/guided-pibt/benchmark-lifelong/agents/sortation_small_{}_600.agents" # #"lmapf_lib/MAPFCompetition2023/example_problems/random.domain/agents/random_400.agents"
predefined_tasks_indexed=True
predefined_tasks_template= default_configs[exp_name]["predefined_tasks_template"].replace("NUM_AGENTS", num_agents) #"/root/GRF_MARL/lmapf_lib/MAPFCompetition2023/our_problems/maze.domain/tasks/maze_1_256_20000_0_rs19.tasks"#"lmapf_lib/Guided-PIBT/guided-pibt/benchmark-lifelong/tasks/sortation_small_{}.task" #"lmapf_lib/Guided-PIBT/guided-pibt/benchmark-lifelong/tasks/warehouse_large_{}.tasks" #"lmapf_lib/MAPFCompetition2023/example_problems/random.domain/tasks/random-32-32-20-400.tasks"

num_devices=args.num_devices
num_processes=args.num_processes
num_episodes_per_instance=args.num_episodes_per_instance

# should not use the map filters in general. it is only for convienence
map_filter_keep=None#["mazes-s17_wc6_od55"]#["test-mazes-s41_wc5_od50","test-mazes-s45_wc4_od55"]#None
# interesting: not good at the following two maps
map_filter_remove=None #["test-mazes-s41_wc5_od50","test-mazes-s45_wc4_od55"]


basic_info={
    "exp_name": exp_name,
    "model_path": model_path,
    "WPPL_mode": WPPL_mode,
    "num_agents": num_agents,
    "rollout_length": rollout_length,
    "max_iterations": max_iterations,
    "num_threads": num_threads,
    "num_devices": num_devices,
    "num_processes": num_processes,
    "num_episodes_per_instance": num_episodes_per_instance
}


timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
output_folder = os.path.join(output_folder, "{}_{}_{}_{}".format(timestamp,exp_name,WPPL_mode,num_agents))
log_folder = os.path.join(output_folder,"log")
os.makedirs(log_folder, exist_ok=True)

def get_model_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

for map_name, num_robots in env.map_manager.instances_list:
    # it will precompute HT for each instance
    env.set_curr_env2(map_name, num_robots, verbose=False)

def initialize():
    global env
    global policy
    global device
    
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # torch.set_num_threads(1)
    
    if num_processes!=1:
        current = multiprocessing.current_process()
        worker_id=current._identity[0]
    else:
        worker_id=0    
        
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
    _env=MultiLMAPFEnv(0, seed, cfg.rollout_manager.worker.envs[0], device, map_filter_keep, map_filter_remove, precompute_HT=False)
    for map_name, num_robots in _env.map_manager.instances_list:
        # it will precompute HT for each instance
        env.set_curr_env2(map_name, num_robots, verbose=False)
        _env.set_curr_env2(map_name, num_robots, verbose=False)
        _env.set_HT(env.get_HT())
        _env.set_PLNSSolver(env.get_PLNSSolver())
    env=_env
    env.check_valid(check_valid)
    env.enable_log(collect_log)
    Logger.info("env generated")

def evaluate(exp_arg):
    global_timer.record("evaluate_s")
    # current = multiprocessing.current_process()
    
    # if num_devices!=1:
    #     worker_id=current._identity[0]
    #     device="cuda:{}".format(worker_id%num_devices)
    # else:
    #     device="cuda"
    
    global env
    global policy
    global device
    
    idx, config_path, model_path, seed, map_name, num_agents = exp_arg
    Logger.info("Start rollout {}".format(idx))
    # env.set_seed(seed)
    # cfg = load_cfg(config_path)

    # cfg.rollout_manager.worker.envs[0].WPPL.mode=WPPL_mode
    # cfg.rollout_manager.worker.envs[0].rollout_length=rollout_length
    # cfg.rollout_manager.worker.envs[0].WPPL.max_iterations=max_iterations
    # cfg.rollout_manager.worker.envs[0].WPPL.num_threads=num_threads

    # policy_id = "policy_0"
    # policy=MAPPO.load(model_path, env_agent_id="agent_0")
    
    # Logger.warning("#params of actor is {}".format(get_model_num_params(policy.actor)))
    # Logger.warning("#params of critic is {}".format(get_model_num_params(policy.critic)))
    
    # policy=policy.to_device(device)
    
    # Logger.info("policy generated")    
    # env=MultiLMAPFEnv(0, seed, cfg.rollout_manager.worker.envs[0], device, map_filter_keep, map_filter_remove)
    
    # env.check_valid(check_valid)
    # env.enable_log(collect_log)
    
    # Logger.info("env generated")
    
    if predefined_starts_template:
        if predefined_starts_indexed:
            predefined_starts_path=predefined_starts_template.format(idx)
        else:
            predefined_starts_path=predefined_starts_template
        env.load_starts(predefined_starts_path)
        Logger.info("predefined starts loadded")
    
    if predefined_tasks_template:
        if predefined_tasks_indexed:
            predefined_tasks_path=predefined_tasks_template.format(idx)
        else:
            predefined_tasks_path=predefined_tasks_template
        env.load_tasks(predefined_tasks_path)
        Logger.info("predefined tasks loaded")
    
    agent = "agent_0"
    policy_id = "policy_0"
    behavior_policies = {
        agent: (policy_id, policy),
    }
    
    global_timer.time("evaluate_s","prepare")
    
    Logger.info("rollout started")
    episode_length=cfg.rollout_manager.worker.envs[0].rollout_length
    rollout_desc = RolloutDesc(idx, "agent_0", None, None, None, None, None)
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
        collect_log=collect_log,
        device=device,
        instance=(map_name, num_agents)
        # rollout_epoch = 100,
    )
    Logger.info("rollout finished")
    global_timer.time("prepare","rollout")
    mean_step_time=global_timer.elapse("rollout")/episode_length
    rollout_results["idx"]=idx
    rollout_results["seed"]=env.seed
    rollout_results["mean_step_time"]=mean_step_time
    rollout_results["prepare_time"]=global_timer.elapse("prepare")
    Logger.info("exp {}'s results: {}, mean step time: {}".format(idx, rollout_results["results"], mean_step_time))
    
    print(global_timer.mean_elapses)
    
    assert len(rollout_results["results"])==1
    result = rollout_results["results"][0]
    if collect_log:
        log = result["log"]       
        log_path = os.path.join(log_folder,"log_{}.json".format(idx))
        log.dump(log_path)
        result.pop("log")
    
    return rollout_results

if __name__=="__main__":
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    np.random.seed(seed)
    exp_args=[]
    idx=0
    for jdx in range(num_episodes_per_instance):
        for map_name,num_agents in env.map_manager.instances_list:
            seed=np.random.randint(0, np.iinfo(np.int32).max)
            exp_args.append((idx, config_path, model_path, seed, map_name, num_agents))
            idx+=1
            
    Logger.warning("total exps: {} for {} instances".format(len(exp_args),len(env.map_manager.instances_list)))

    if num_processes==1:
        initialize()
        results=[evaluate(exp_arg) for exp_arg in exp_args]
    else:
        pool=Pool(num_processes,initializer=initialize)
        results=pool.map(evaluate, exp_args)

    import tree
    import torch

    def convert_to_np(x):
        if isinstance(x, torch.Tensor):
            if len(x.shape)==0:
                return x.item()
            else:
                return x.cpu().numpy().tolist()
        else:
            return x
    
    results=tree.map_structure(convert_to_np,results)

    # TODO dump results
    results_json_path = os.path.join(output_folder,"results.json")
    with open(results_json_path,'w') as f:
        json.dump(results,f,indent=4)

    throughputs=[result["results"][0]["stats"]["agent_0"]["throughput"] for result in results]
    mean_step_times=[result["mean_step_time"] for result in results]
    prepare_times=[result["prepare_time"] for result in results]
        
    Logger.info("throughput: mean: {} std: {} min: {} max: {} 2.5\%: {} 97.5\%: {}".format(np.mean(throughputs), np.std(throughputs), np.min(throughputs), np.max(throughputs), np.percentile(throughputs,2.5), np.percentile(throughputs,97.5)))
    Logger.info("mean step time: {}".format(np.mean(mean_step_times)))
    Logger.info("mean prepare time: {}".format(np.mean(prepare_times)))


    data={}
    for _r in results:
        r=_r["results"][0]
        if (r["map_name"],r["num_robots"]) not in data:
            data[(r["map_name"],r["num_robots"])]={
                "throughput": [],
                "mean_step_time": [],
            }
        data[(r["map_name"],r["num_robots"])]["throughput"].append(r["stats"]["agent_0"]["throughput"])
        data[(r["map_name"],r["num_robots"])]["mean_step_time"].append(_r["mean_step_time"])

    # import pandas as pd
    # data=pd.DataFrame(data,columns=["map_name","num_robots","stats"])
    #ins_data=data.groupby(by=["map_name","num_robots"]).mean().reset_index()

    import numpy as np
    from scipy.stats import t as t_dist

    def get_stats(arr):
        sample_mean = np.mean(arr)
        sample_std = np.std(arr,ddof=1)
        sample_min = np.min(arr)
        sample_max = np.max(arr)
        sample_p025 = np.percentile(arr, 2.5)
        sample_p975 = np.percentile(arr, 97.5)
        n = len(arr)
        degree_of_freedom = n-1
        confidence_interval = t_dist.interval(
            0.95, 
            degree_of_freedom, 
            loc=sample_mean, 
            scale=sample_std / np.sqrt(n)
        )
        stats={
            "mean": sample_mean,
            "std": sample_std,
            "min": sample_min,
            "max": sample_max,
            "p025": sample_p025,
            "p975": sample_p975,
            "ci95": confidence_interval 
        }
        
        return stats

    summaries={}
    all_summary_throughput=get_stats(throughputs)
    all_summary_mean_step_time=get_stats(mean_step_times)
    summaries={
        "all": {
            "throughput": all_summary_throughput,
            "mean_step_time": all_summary_mean_step_time
        },
        "instances": {},
        "basic_info": basic_info
    }

    data=sorted([(k,v) for k,v in data.items()])
    for k,v in data:
        summary_throughput = get_stats(v["throughput"])
        summary_mean_step_time = get_stats(v["mean_step_time"])
        summaries["instances"]["{},{}".format(k[0],k[1])] = {
            "throughput": summary_throughput,
            "mean_step_time": summary_mean_step_time
        }

    summaries_json_path = os.path.join(output_folder,"summaries.json")
    with open(summaries_json_path,'w') as f:
        json.dump(summaries,f,indent=4)
        
    import pprint
    pprint.pprint(summaries)