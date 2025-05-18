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
from optimizer import Optimizer
from utils import get_stats
import pprint
import json

# TODO(rivers): we should save evaluation results rather than print them...

config_path = "expr_configs/LMAPF/BC/mappo_random_32_32_400agents.yaml" 
#root_folder = "logs/LMAPF/basic_cnn_re_embed_perm5_bn_1block_ltf_BC_iter1_256agents_4gpus/2024-05-26-23-55-06/"
root_folder = "logs/LMAPF/LMAPF_test_BC_1024samples_basic_cnn_re_embed_perm5_bn_1block_random_32_32_20_400agents_2iter_random_perm/2024-05-22-05-15-58/"
#root_folder = "logs/LMAPF/basic_cnn_re_embed_perm5_bn_0block_ltf_RL_iter0_map_room_1000agents/2024-05-29-07-10-07"
WPPL_mode="PIBT-RL"
rollout_length=200
max_iterations=5000
num_threads=1
collect_log=False
check_valid=False

# if use predefined data, please be careful with the number of episodes
predefined_starts_indexed=True
predefined_starts_template="" #"lmapf_lib/Guided-PIBT/guided-pibt/benchmark-lifelong/agents/warehouse_large_{}_10000.agents" #"lmapf_lib/MAPFCompetition2023/example_problems/random.domain/agents/random_400.agents"
predefined_tasks_indexed=True
predefined_tasks_template="" #"lmapf_lib/Guided-PIBT/guided-pibt/benchmark-lifelong/tasks/warehouse_large_{}.tasks" #"lmapf_lib/MAPFCompetition2023/example_problems/random.domain/tasks/random-32-32-20-400.tasks"
num_devices=1
num_processes=8
num_episode_per_instance=8
# CMA-ES
n_iters=50
n_samples=128
sigma=1
# should not use the map filters in general. it is only for convienence
map_filter_keep=None#["mazes-s17_wc6_od55"]#["test-mazes-s41_wc5_od50","test-mazes-s45_wc4_od55"]#None
# interesting: not good at the following two maps
map_filter_remove=None #["test-mazes-s41_wc5_od50","test-mazes-s45_wc4_od55"]

model_path = os.path.join(root_folder,"agent_0/agent_0-default-1/best")
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
output_folder = os.path.join(model_path, "cma_es/{}_mode_{}_roll_{}steps_lns_{}iters_{}threads".format(timestamp,WPPL_mode,rollout_length,max_iterations,num_threads))
log_folder = os.path.join(output_folder,"log")
model_folder = os.path.join(output_folder,"model")
os.makedirs(output_folder)
os.makedirs(log_folder)
os.makedirs(model_folder)

def evaluate(exp_arg):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    
    global_timer.record("evaluate_s")
    current = multiprocessing.current_process()
    
    if num_devices!=1:
        worker_id=current._identity[0]
        device="cuda:{}".format(worker_id%num_devices)
    else:
        device="cuda"
    
    idx, config_path, model_path, seed, map_name, num_agents, params_idx, params_sample = exp_arg
    cfg = load_cfg(config_path)

    cfg.rollout_manager.worker.envs[0].WPPL.mode=WPPL_mode
    cfg.rollout_manager.worker.envs[0].rollout_length=rollout_length
    cfg.rollout_manager.worker.envs[0].WPPL.max_iterations=max_iterations
    cfg.rollout_manager.worker.envs[0].WPPL.num_threads=num_threads

    policy_id = "policy_0"
    policy=MAPPO.load(model_path, env_agent_id="agent_0")
    
    Optimizer.set_policy_params(policy, params_sample)
    Logger.info("set sampled policy params")
    
    policy=policy.to_device(device)
    Logger.info("policy generated")
    

    
    episode_length=cfg.rollout_manager.worker.envs[0].rollout_length
    rollout_desc = RolloutDesc(idx, "agent_0", None, None, None, None, None)
    env=MultiLMAPFEnv(0, seed, cfg.rollout_manager.worker.envs[0], "cpu", map_filter_keep, map_filter_remove)
    
    env.check_valid(check_valid)
    env.enable_log(collect_log)
    
    Logger.info("env generated")
    
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
    behavior_policies = {
        agent: (policy_id, policy),
    }
    
    global_timer.time("evaluate_s","prepare")
    
    Logger.info("rollout started")
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
    rollout_results["params_idx"]=params_idx
    rollout_results["idx"]=idx
    rollout_results["seed"]=seed
    rollout_results["mean_step_time"]=mean_step_time
    rollout_results["prepare_time"]=global_timer.elapse("prepare")
    Logger.info("exp {}'s results: {}, mean step time: {}".format(idx, rollout_results["results"], mean_step_time))
    
    assert len(rollout_results["results"])==1
    result = rollout_results["results"][0]
    if collect_log:
        log = result["log"]       
        log_path = os.path.join(log_folder,"log_{}.json".format(idx))
        log.dump(log_path)
        result.pop("log")
    
    return rollout_results


policy_id = "policy_0"
policy=MAPPO.load(model_path, env_agent_id="agent_0")
optimizer = Optimizer(policy, sigma=sigma, n_samples = n_samples)

cfg=load_cfg(config_path)
env=MultiLMAPFEnv("global",0,cfg.rollout_manager.worker.envs[0],"cpu",map_filter_keep,map_filter_remove)

pool=Pool(num_processes)

for iter_idx in range(n_iters):
    params_samples = optimizer.get_samples()
    exp_args=[]
    idx=0
    for params_sample_idx in range(len(params_samples)):
        params_sample = params_samples[params_sample_idx]
        for jdx in range(num_episode_per_instance):
            for map_name,num_agents in env.map_manager.instances_list:
                seed=idx
                exp_args.append((idx, config_path, model_path, seed, map_name, num_agents, params_sample_idx, params_sample))
                idx+=1
            
    Logger.warning("[Iter {}] total exps: {} for {} samples and  {} instances with each {} exps".format(iter_idx, len(params_sample), len(exp_args), len(env.map_manager.instances_list), num_episode_per_instance))

    if num_processes==1:
        results=[evaluate(exp_arg) for exp_arg in exp_args]
    else:
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

    throughputs=[result["results"][0]["stats"]["agent_0"]["throughput"] for result in results]
    mean_step_times=[result["mean_step_time"] for result in results]
    prepare_times=[result["prepare_time"] for result in results]
    
    Logger.info("[Iter {}] throughput: mean: {} std: {} min: {} max: {} 2.5\%: {} 97.5\%: {}".format(iter_idx, np.mean(throughputs), np.std(throughputs), np.min(throughputs), np.max(throughputs), np.percentile(throughputs,2.5), np.percentile(throughputs,97.5)))
    Logger.info("[Iter {}] mean step time: {}".format(iter_idx, np.mean(mean_step_times)))
    Logger.info("[Iter {}] mean prepare time: {}".format(iter_idx, np.mean(prepare_times)))

    data={}
    for _r in results:
        r=_r["results"][0]
        if _r["params_idx"] not in data:
            data[_r["params_idx"]]=[]
        data[_r["params_idx"]].append(r["stats"]["agent_0"]["throughput"])

    summaries={}
    all_summary=get_stats(throughputs)
    summaries={
        "all": all_summary,
        "samples": {}
    }

    data=sorted([(k,v) for k,v in data.items()])
    for k,v in data:
        summary = get_stats(v)
        summaries["samples"][k] =summary
        
    Logger.info("[Iter {}]\n{}".format(iter_idx,pprint.pformat(summaries)))
    
    # cma-es will minimize the func
    unfitnesses = [-summaries["samples"][params_idx]["mean"] for params_idx in range(len(params_samples))]
    
    optimizer.update(params_samples, unfitnesses)
    
    best_params, fval = optimizer.get_best_sample()
    Logger.warn("[Iter {}] best throughput: {}".format(iter_idx,-fval))
    Optimizer.set_policy_params(optimizer.policy, best_params)
    _model_folder = os.path.join(model_folder,"iter_{}".format(iter_idx))
    policy.dump(_model_folder)

optimizer.es.result_pretty()


    