# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import OrderedDict
import numpy as np
from light_malib.utils.logger import Logger
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.LMAPF.env import MultiLMAPFEnv
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.timer import global_timer
from light_malib.utils.naming import default_table_name
from light_malib.envs.LMAPF.WPPL import WPPL
import ray
import torch.nn.functional as F
import torch
import time

def rename_fields(data, fields, new_fields):
    assert len(fields)==len(new_fields)
    for agent_id, agent_data in data.items():
        for field, new_field in zip(fields,new_fields):
            if field in agent_data:
                field_data = agent_data.pop(field)
                agent_data[new_field] = field_data
    return data


def select_fields(data, fields):
    rets = {
        agent_id: {field: agent_data[field] for field in fields if field in agent_data}
        for agent_id, agent_data in data.items()
    }
    return rets


def update_fields(data1, data2):
    def update_dict(dict1, dict2):
        d = {}
        d.update(dict1)
        d.update(dict2)
        return d

    rets = {
        agent_id: update_dict(data1[agent_id], data2[agent_id]) for agent_id in data1
    }
    return rets

def shallow_copy(data):
    return {agent_id: {k: v for k, v in agent_data.items()} for agent_id, agent_data in data.items()}

def stack_step_data(step_data_list, bootstrap_data, fields=None):
    if fields is None:
        fields=step_data_list[0].keys()
    episode_data = {}
    for field in fields:
        data_list = [step_data[field] for step_data in step_data_list]
        if field in bootstrap_data:
            data_list.append(bootstrap_data[field])
        try:
            episode_data[field] = torch.stack(data_list)
        except Exception as e:
            import traceback
            Logger.error("error stacking {}".format(field))
            Logger.error(traceback.format_exc())
            first_shape=data_list[0].shape
            for idx,data in enumerate(data_list):
                if data.shape!=first_shape:
                    Logger.error("field {}: first_shape: {}, mismatched_shape: {}, mismatched_idx: {}".format(field,first_shape,data.shape,idx))
                    break
            raise e
    return episode_data

def pull_policies(rollout_worker,policy_ids):
    rollout_worker.pull_policies(policy_ids)
    behavior_policies = rollout_worker.get_policies(policy_ids)
    return behavior_policies

def env_reset(env:MultiLMAPFEnv, behavior_policies, custom_reset_config):
    global_timer.record("env_step_start")
    env_rets = env.reset(custom_reset_config)
    global_timer.time("env_step_start", "env_step_end", "env_step")

    init_rnn_states = {
        agent_id: behavior_policies[agent_id][1].get_initial_state(
            batch_size=env.team_sizes[agent_id]
        )
        for agent_id in env.agent_ids
    }

    step_data = update_fields(env_rets, init_rnn_states)
    return step_data

def compute_advantages(episode, gamma=0.99, lamb=0.95):
    # only support gae
    
    # T, num_robots, 1
    rewards=episode[EpisodeKey.REWARD]
    # T+1, num_robots, 1
    dones=episode[EpisodeKey.DONE].float()
    # T+1, num_robots, 1
    values=episode[EpisodeKey.STATE_VALUE]
    
    # in LMAP 
    T=rewards.shape[0]
    assert dones.shape[0]==T+1 and values.shape[0]==T+1

    gae = 0
    advantages = torch.zeros_like(rewards)
    values = values.to(rewards.device)
    for t in reversed(range(T)):
        # TODO: double check dones[t] or dones[t+1]
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t+1]) - values[t]
        gae = delta + gamma * lamb * (1 - dones[t]) * gae
        advantages[t] = gae
        
    episode[EpisodeKey.ADVANTAGE] = advantages
    episode[EpisodeKey.RETURN] = advantages + values[:-1]
    episode[EpisodeKey.DONE] = dones[:-1]
    episode[EpisodeKey.STATE_VALUE] = values[:-1]
    
    return episode

def pack_episode(step_data_list, last_step_data, rollout_desc, gae_gamma, gae_lambda, s_idx=None, e_idx=None):
    bootstrap_data = select_fields(
        last_step_data,
        [
            EpisodeKey.STATE_VALUE,
            EpisodeKey.DONE
        ],
    )
    bootstrap_data = rename_fields(bootstrap_data, [EpisodeKey.NEXT_OBS,EpisodeKey.NEXT_STATE], [EpisodeKey.CUR_OBS,EpisodeKey.CUR_OBS])
    bootstrap_data = bootstrap_data[rollout_desc.agent_id]
    
    _episode = stack_step_data(
        step_data_list[s_idx:e_idx],
        # TODO CUR_STATE is not supported now
        bootstrap_data,
    )

    episode = _episode
    episode = compute_advantages(episode, gae_gamma, gae_lambda)

    return episode


def submit_episode(data_server, episode, rollout_desc):
    for k, v in episode.items():
        episode[k]=v.to("cpu")
    
    # submit data:
    if hasattr(data_server.save, 'remote'):
        data_server.save.remote(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            [episode],
        )
    else:
        data_server.save(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            [episode],
        )

def submit_episode_bc(data_server, table_name, imitation_data):
    # seed = imitation_data["seed"]
    # map_name = imitation_data["map_name"]
    # num_robots = imitation_data["num_robots"]
    # curr_positions=imitation_data["curr_positions"]
    # target_positions=imitation_data["target_positions"]
    # priorities=imitation_data["priorities"]
    # actions=imitation_data["actions"]
    
    # assert len(curr_positions)==len(actions), "{} vs {}".format(len(curr_positions),len(actions))
    
    # transitions = []
    # for step in range(len(actions)):
    #     transition = {
    #         "map_name": map_name,
    #         "num_robots": num_robots,
    #         "curr_positions": curr_positions[step],
    #         "target_positions": target_positions[step],
    #         "priorities": priorities[step],
    #         "actions": actions[step]
    #     }
    #     transitions.append(transition)    
    
    if hasattr(data_server.save, 'remote'):
        data_server.save.remote(
            table_name,
            [imitation_data]
        )
    else:
        data_server.save(
            table_name,
            [imitation_data]
        )

def submit_batches(data_server,episode, rollout_desc):
    for k, v in episode.items():
        episode[k]=v.to("cpu")
    
    transitions = []
    for step in range(len(episode) - 1):
        transition = {
            EpisodeKey.CUR_OBS: episode[step][EpisodeKey.CUR_OBS],  # [np.newaxis, ...],
            EpisodeKey.ACTION_MASK: episode[step][EpisodeKey.ACTION_MASK],  # [np.newaxis, ...],
            EpisodeKey.ACTION: episode[step][EpisodeKey.ACTION],  # [np.newaxis, ...],
            EpisodeKey.REWARD: episode[step][EpisodeKey.REWARD],  # [np.newaxis, ...],
            EpisodeKey.DONE: episode[step][EpisodeKey.DONE],  # [np.newaxis, ...],
            EpisodeKey.NEXT_OBS: episode[step + 1][EpisodeKey.CUR_OBS],  # [np.newaxis, ...],
            EpisodeKey.NEXT_ACTION_MASK: episode[step + 1][EpisodeKey.ACTION_MASK],  # [np.newaxis, ...]
            EpisodeKey.CRITIC_RNN_STATE: episode[step][EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.NEXT_CRITIC_RNN_STATE: episode[step + 1][EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.CUR_GLOBAL_STATE: episode[step][EpisodeKey.CUR_GLOBAL_STATE],
            EpisodeKey.NEXT_GLOBAL_STATE: episode[step + 1][EpisodeKey.CUR_GLOBAL_STATE]
        }
        transitions.append(transition)
    if hasattr(data_server.save, 'remote'):
        data_server.save.remote(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            transitions
        )
    else:
        data_server.save(
            default_table_name(
                rollout_desc.agent_id,
                rollout_desc.policy_id,
                rollout_desc.share_policies,
            ),
            transitions
        )

def rollout_func(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: MultiLMAPFEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs
):
    s=time.time()
    """
    TODO(jh): modify document

    Rollout in simultaneous mode, support environment vectorization.

    :param VectorEnv env: The environment instance.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A dict of rollout information.
    """
    device=kwargs.get("device",None)
    if device is None:
        policy_device="cuda"
    else:
        policy_device=device
    
    rollout_epoch=kwargs.get("rollout_epoch",None)
    
    sync_rnd_val=kwargs.get("sync_rnd_val",None)
    instance=kwargs.get("instance", None)
    
    if instance is None:
        if sync_rnd_val is not None and not eval:
            # TODO(rivers): it is a bad idea to use rollout_epoch to sync here?
            env.set_curr_env(sync_rnd_val)
        else:
            env.set_curr_env(None)
    else:
        map_name, num_robots=instance
        env.set_curr_env2(map_name, num_robots, verbose=False)
        
    collect_data=kwargs.get("collect_data",False)
    collect_log=kwargs.get("collect_log",False)
    
    if collect_log:
        env.enable_log(collect_log)

    sample_length = kwargs.get("sample_length", rollout_length)
    render = kwargs.get("render", False)
    if render:
        env.render()

    episode_mode = kwargs.get('episode_mode','traj')
    verbose = kwargs.get("verbose", False)

    policy_ids = OrderedDict()
    feature_encoders = OrderedDict()
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id
        policy.eval()
        # TODO: the number of devices should be configurable
        behavior_policies[agent_id]=(policy_id,policy.to_device(policy_device,in_place=True))
        
    WPPL_mode=kwargs.get("wppl_mode",None)
    if WPPL_mode is None:
        # really bad code design...
        WPPL_mode=env.cfg["WPPL"]["mode"]
    if collect_data:
        assert WPPL_mode in ["PIBT-RL-LNS-Guide","PIBT-RL-LNS","PIBT-LNS"]
    assert WPPL_mode in ["PIBT-RL", "PIBT", "PIBT-LNS","PIBT-RL-LNS","PIBT-RL-LNS-Guide"]
    old_pibt_func=env.get_pibt_func()
    if WPPL_mode not in ["PIBT-RL","PIBT"]:
        
        # TODO it should be set based on whether we want to evaluate or train
        if WPPL_mode=="PIBT-LNS":
            pibt_mode="solve"
        elif WPPL_mode in ["PIBT-RL-LNS","PIBT-RL-LNS-Guide"]:
            pibt_mode="guard"
            
        # we should move WPPL solver to env as well
        wppl = WPPL(
            env.seed+1000000, env, env.cfg,
            env.cfg["WPPL"]["window_size"], env.cfg["WPPL"]["num_threads"], env.cfg["WPPL"]["max_iterations"],
            env.cfg["WPPL"]["time_limit"], True, pibt_mode, policy_device, env.cfg["WPPL"]["verbose"]
        )
        wppl.set_rollout_func(rollout_func_for_WPPL)
        wppl.set_pack_episode_func(pack_episode)
        
        policy_id, policy=behavior_policies[rollout_desc.agent_id]
        policy=policy.to_device(policy_device)
        behavior_policies[rollout_desc.agent_id]=(policy_id,policy.to_device(policy_device,in_place=True))
        
        if WPPL_mode=="PIBT-RL-LNS-Guide":
            # PIBT-RL-LNS-Guide uses a guiding policy with wppl to provide expert advice
            # however, the transitions are still based on the current policy
            env.set_pibt_func("guard")
            if data_server is not None:
                guiding_policy=ray.get(data_server.get.remote("guiding_policy",False))
            else:
                guiding_policy=None
            if guiding_policy is not None:
                guiding_policy=guiding_policy.to_device(policy_device,in_place=True)
            else:
                # self-guiding
                guiding_policy=policy
            wppl.set_policy(guiding_policy)
        else:
            assert eval, "only support eval mode for PIBT-LNS and PIBT-RL-LNS, because their transisions are based on wppl-enhanced policies"
            env.set_pibt_func("none")
            if data_server is not None:
                guiding_policy=ray.get(data_server.get.remote("guiding_policy",False))
            else:
                guiding_policy=None
            if guiding_policy is not None:
                guiding_policy=guiding_policy.to_device(policy_device,in_place=True)
            else:
                # self-guiding
                guiding_policy=policy
            wppl.set_policy(guiding_policy)
    else:
        if WPPL_mode=="PIBT":
            env.set_pibt_func("solve")
        else:
            env.set_pibt_func("guard")
        #policy = policy.to_device(policy_device)
        wppl = None  

    custom_reset_config = {
        # "feature_encoders": feature_encoders,
        # "main_agent_id": rollout_desc.agent_id
    }
    
    if "curr_positions" in kwargs:
        custom_reset_config["curr_positions"]=kwargs["curr_positions"]
    if "target_positions" in kwargs:
        custom_reset_config["target_positions"]=kwargs["target_positions"]
    if "priorities" in kwargs:
        custom_reset_config["priorities"]=kwargs["priorities"]
    
    # assert rollout_length==env.rollout_length, "we assume the simplest case now!"
    
    if eval:
        assert rollout_length<=env.rollout_length,"rollout length {} should be less or equal to env rollout length {} during evaluation".format(rollout_length,env.rollout_length)
    env.set_eval(eval)
    
    if not env.initialized() or eval or env.is_terminated():
        # assert rollout_length<=env.rollout_length,"rollout length {} should be less than or equal to env rollout length {} during evaluation".format(rollout_length,env.rollout_length)
        # we add a shallow_copy to ensure correctness in case that step_data is modified by some in-place operation.
        step_data = env_reset(env,behavior_policies,custom_reset_config)
        env.set_prev_step_data(shallow_copy(step_data))
    else:
        step_data = env.get_prev_step_data()

    step = 0
    step_data_list = []
    results = []
    episodes = []
    
    if collect_data:
        seed = env.get_seed()
        
        curr_positions_list=[]
        target_positions_list=[]
        priorities_list=[]
        actions_list=[]

        curr_positions=env.curr_positions.clone()
        curr_positions=curr_positions[...,0]*env.map.width+curr_positions[...,1]
        curr_positions=curr_positions
        curr_positions_list.append(curr_positions)
        
        target_positions=env.target_positions.clone()
        target_positions=target_positions[...,0]*env.map.width+target_positions[...,1]
        target_positions=target_positions
        target_positions_list.append(target_positions)
    
        priorities_list.append(env.priorities.clone())
    # collect until rollout_length
    # TODO: we need to carefully deal with the termination of the environment
    # because we need to discard the last step data if the episode is terminated
    for step in range(rollout_length):
        if env.is_terminated():
            # it must be not eval
            if eval:
                assert False, "env cannot terminate here during evaluation"
            
            assert len(step_data_list)>=1, "step_data_list should not be empty here"

            # we have a step data in the list, and we also have the last step.
            policy_inputs = rename_fields(
                step_data, 
                [EpisodeKey.NEXT_OBS,EpisodeKey.NEXT_GLOBAL_OBS], 
                [EpisodeKey.CUR_OBS,EpisodeKey.CUR_GLOBAL_OBS]
            )
            global_timer.record("inference_start")
            for agent_id, (policy_id, policy) in behavior_policies.items():
                if WPPL_mode=="PIBT":
                    # just some dummy values if PIBT
                    policy_outputs[agent_id] = {
                        EpisodeKey.ACTION: None,
                        EpisodeKey.STATE_VALUE: None
                    }
                else:
                    policy_outputs[agent_id] = policy.compute_action(
                        inference=True, 
                        explore=not eval,
                        to_numpy=False,
                        step = kwargs.get('rollout_epoch', 0),
                        **policy_inputs[agent_id]
                    )
            global_timer.time("inference_start", "inference_end", "inference")
            step_data=update_fields(step_data,select_fields(policy_outputs,[EpisodeKey.STATE_VALUE]))
            episode = pack_episode(step_data_list, step_data, rollout_desc, gae_gamma=env.gae_gamma, gae_lambda=env.gae_lambda)
            episodes.append(episode)
            # clear step_data_list
            step_data_list=[]
        
            stats = env.get_episode_stats()
            result = {
                "main_agent_id": rollout_desc.agent_id,
                "policy_ids": policy_ids,
                "stats": stats,
            }
            if collect_log:
                result["log"]=env.get_episode_log()
            results.append(result)
            
            step_data = env_reset(env,behavior_policies,custom_reset_config)
            # we add a shallow_copy to ensure correctness in case that step_data is modified by some in-place operation.
            env.set_prev_step_data(shallow_copy(step_data))
        # prepare policy input
        
        # if step==0:
        #     print(env.curr_positions)
        #     print(env.target_positions)
        #     observation=step_data[rollout_desc.agent_id][EpisodeKey.NEXT_OBS]
        #     observation=observation.reshape(observation.shape[0],4,11,11)
        #     print(observation[:,2])
        
        policy_inputs = rename_fields(
            step_data, 
            [EpisodeKey.NEXT_OBS, EpisodeKey.NEXT_GLOBAL_OBS], 
            [EpisodeKey.CUR_OBS, EpisodeKey.CUR_GLOBAL_OBS]
        )
        policy_outputs = {}
        global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            if WPPL_mode=="PIBT":
                # just some dummy values if PIBT
                policy_outputs[agent_id] = {
                    EpisodeKey.ACTION: None,
                    EpisodeKey.STATE_VALUE: None
                }
            else:
                policy_outputs[agent_id] = policy.compute_action(
                    inference=True, 
                    explore=not eval,
                    to_numpy=False,
                    step = kwargs.get('rollout_epoch', 0),
                    **policy_inputs[agent_id]
                )

        global_timer.time("inference_start", "inference_end", "inference")

        # NOTE(rivers): there are actually two ways we can do,
        # first we can sample with the actions from PIBT
        # second we can sample with the actions from the WPPL
        # we will try the first one first, because it is close to the offline setting.
        
        # NOTE the second one need to recompute the log prob from PIBT-RL
        # replace the original policy with WPPL-wrapped one
        # behavior_policies[rollout_desc.agent_id]=(policy_id,wppl)
        
        # we assume WPPL returns 
        # the final actions (from WPPL) as EpisodeKey.ACTION
        # the init actions (from PIBT) as EpisodeKey.INIT_ACTION
        # TODO really bad coding
        # if not eval:
        #     env.set_pibt_func("guard")
        #     # rollout for training
        #     if wppl is not None:
        #         actions = select_fields(policy_outputs, [EpisodeKey.INIT_ACTION])
        #         actions = rename_fields(actions, [EpisodeKey.INIT_ACTION], [EpisodeKey.ACTION])
        #     else:    
        #         actions = select_fields(policy_outputs, [EpisodeKey.ACTION])
        # else:
        #     if wppl is not None:
        #         env.set_pibt_func("none")
        #     else:    
        #         env.set_pibt_func("guard")
        #     actions = select_fields(policy_outputs, [EpisodeKey.ACTION])
        
        
        if WPPL_mode in ["PIBT-LNS","PIBT-RL-LNS","PIBT-RL-LNS-Guide"]: # or (WPPL_mode=="PIBT-RL-LNS-Guide" and not eval):
            guiding_policy_outputs={}
            global_timer.record("inference_guidance_start")
            guiding_policy_outputs[rollout_desc.agent_id]=wppl.compute_action(
                inference=True, 
                explore=not eval,
                to_numpy=False,
                step = kwargs.get('rollout_epoch', 0),
                **policy_inputs[rollout_desc.agent_id],
                map_name=env.map.name,
                num_robots=env.num_robots
            )
            global_timer.time("inference_guidance_start", "inference_guidance_end", "inference_guidance")
                    
            if collect_data:
                actions_list.append(guiding_policy_outputs[rollout_desc.agent_id][EpisodeKey.ACTION])
            
        if eval:
            if WPPL_mode in ["PIBT-LNS","PIBT-RL-LNS"]:
                actions = select_fields(guiding_policy_outputs, [EpisodeKey.ACTION])                
            elif WPPL_mode in ["PIBT-RL-LNS-Guide","PIBT-RL","PIBT"]:
                actions = select_fields(policy_outputs, [EpisodeKey.ACTION])
            else:
                raise NotImplementedError("unknown wppl mode to obtain actions")
        else:
            # awalys use actions from the policy without wppl wrapper
            actions = select_fields(policy_outputs, [EpisodeKey.ACTION])

        global_timer.record("env_step_start")
        env_rets = env.step(actions)
        global_timer.time("env_step_start", "env_step_end", "env_step")
        
        # Modify the reward here
        # if WPPL_mode=="PIBT-RL-LNS-Guide" and not eval:
        #     ## evaluate guided action at current action logits
        #     action_logits=torch.from_numpy(policy_outputs[rollout_desc.agent_id][EpisodeKey.ACTION_LOGITS])
        #     guiding_actions=torch.from_numpy(guiding_policy_outputs[rollout_desc.agent_id][EpisodeKey.ACTION])
        #     rewards=env_rets[rollout_desc.agent_id][EpisodeKey.REWARD]
        #     with torch.no_grad():
        #         guiding_rewards = -F.cross_entropy(action_logits, guiding_actions, reduction='none').reshape(-1,1)
        #         assert len(guiding_rewards.shape)==2 and guiding_rewards.shape[0]==rewards.shape[0] and guiding_rewards.shape[1]==rewards.shape[1]
        #         # maybe use a double head network to predict the reward
        #         env_rets[rollout_desc.agent_id][EpisodeKey.REWARD] = rewards*env.cfg.rewards_coef+guiding_rewards
        #     env.update_guiding_rewards(guiding_rewards.reshape(-1))
        

        if collect_data:
            curr_positions=env.curr_positions.clone()
            curr_positions=curr_positions[...,0]*env.map.width+curr_positions[...,1]
            curr_positions_list.append(curr_positions)
            
            target_positions=env.target_positions.clone()
            target_positions=target_positions[...,0]*env.map.width+target_positions[...,1]
            target_positions_list.append(target_positions)
        
            priorities_list.append(env.priorities.clone())
                
        if verbose:
            Logger.info("env {} step {}'s stats: {}".format(env.id,step, env.get_episode_stats()))

        # record data after env step
        step_data = update_fields(
            step_data, select_fields(env_rets, [EpisodeKey.REWARD])
        )
        
        step_data = update_fields(
            step_data,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTION, EpisodeKey.ACTION_LOG_PROB, EpisodeKey.STATE_VALUE],
            ),
        )
        
        if WPPL_mode in ["PIBT-LNS","PIBT-RL-LNS"] or (WPPL_mode in ["PIBT-RL-LNS-Guide"] and not eval):
            guiding_policy_outputs = rename_fields(guiding_policy_outputs,[EpisodeKey.ACTION],[EpisodeKey.GUIDING_ACTION])
            step_data = update_fields(
                step_data,
                select_fields(
                    guiding_policy_outputs,
                    [EpisodeKey.GUIDING_ACTION],
                ),
            )

        if not eval:
            # save data of trained agent for training
            step_data_list.append(step_data[rollout_desc.agent_id])
                    
        step_data = update_fields(
            step_data, select_fields(env_rets, [EpisodeKey.DONE])
        )

        # record data for next step
        step_data = update_fields(
            env_rets,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTOR_RNN_STATE, EpisodeKey.CRITIC_RNN_STATE],
            ),
        )
        
        # we add a shallow_copy to ensure correctness in case that step_data is modified by some in-place operation.
        env.set_prev_step_data(shallow_copy(step_data))
    
    if not eval:            #collect after rollout done
        assert len(step_data_list)>=1, "step_data_list should not be empty here"
        
        # call policy.compuate_action again to collect the state value for the last step data.
        policy_inputs = rename_fields(
            step_data, 
            [EpisodeKey.NEXT_OBS, EpisodeKey.NEXT_GLOBAL_OBS], 
            [EpisodeKey.CUR_OBS, EpisodeKey.CUR_GLOBAL_OBS]
        )
        global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            if WPPL_mode=="PIBT":
                # just some dummy values if PIBT
                policy_outputs[agent_id] = {
                    EpisodeKey.ACTION: None,
                    EpisodeKey.STATE_VALUE: None
                }
            else:
                policy_outputs[agent_id] = policy.compute_action(
                    inference=True, 
                    explore=not eval,
                    to_numpy=False,
                    step = kwargs.get('rollout_epoch', 0),
                    **policy_inputs[agent_id]
                )

        global_timer.time("inference_start", "inference_end", "inference")
        step_data=update_fields(step_data,select_fields(policy_outputs,[EpisodeKey.STATE_VALUE]))
        
        # compute advatanges here
        episode = pack_episode(step_data_list, step_data, rollout_desc, gae_gamma=env.gae_gamma, gae_lambda=env.gae_lambda)
        episodes.append(episode)
        
        # submit all episodes
        concated_episodes={}
        for key in episodes[0].keys():
            # concate multiple episodes along the time dimension.
            concated_episodes[key]=torch.concatenate([episode[key] for episode in episodes],axis=0)
        submit_episode(data_server, concated_episodes, rollout_desc)
    
    # print(env.episode_log)
    # env.episode_log.dump("test_log.json")
    
    if eval or env.is_terminated():
        stats = env.get_episode_stats()
        result = {
            "main_agent_id": rollout_desc.agent_id,
            "policy_ids": policy_ids,
            "map_name": env.map.name,
            "num_robots": env.num_robots,
            "stats": stats,
        }
        if collect_log:
            result["log"]=env.get_episode_log()
        results.append(result)
        # step_data = env_reset(env,behavior_policies,custom_reset_config,rollout_epoch,eval)
        # env.set_prev_step_data(step_data)
        
    env.set_pibt_func(old_pibt_func)

    # return restuls
    results={"results":results}      
  
    if collect_data:        
        curr_positions_list=torch.stack(curr_positions_list[:-1],dim=0).to(device="cpu",dtype=torch.int32).numpy()
        target_positions_list=torch.stack(target_positions_list[:-1],dim=0).to(device="cpu",dtype=torch.int32).numpy()
        priorities_list=torch.stack(priorities_list[:-1],dim=0).to(device="cpu",dtype=torch.float32).numpy()
        actions_list=torch.stack(actions_list,dim=0).to(device="cpu",dtype=torch.int32).numpy()
  
        imitation_data = {
            "seed": seed,
            "map_name": env.map.name,
            "num_robots": env.num_robots,
            "curr_positions":curr_positions_list,
            "target_positions":target_positions_list,
            "priorities":priorities_list,
            "actions":actions_list
        }
  
        if data_server is None:
            results["imitation_data"]=imitation_data
        else:
            # submit data to data server
            Logger.error("{} try to submit bc data".format(env.id))
            table_name = "imitation_{}_{}".format(map_name,num_robots)
            submit_episode_bc(data_server, table_name, imitation_data)
            Logger.error("{} submitted bc data".format(env.id))
    
    
    elapse=time.time()-s
    # Logger.error("rollout elapse: {} {}".format(elapse,global_timer.elapses))
    # global_timer.clear()
    
    return results


def rollout_func_BC(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: MultiLMAPFEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs    
):
    sync_rnd_val=kwargs.get("sync_rnd_val",None)
    instance=kwargs.get("instance", None)
    
    env.set_eval(eval)
    
    if instance is None:
        if sync_rnd_val is not None and not eval:
            # TODO(rivers): it is a bad idea to use rollout_epoch to sync here?
            env.set_curr_env(sync_rnd_val)
        else:
            env.set_curr_env(None)
    else:
        map_name, num_robots=instance
        env.set_curr_env2(map_name, num_robots, verbose=False)
    
    map_name = env.curr_env.map.name
    num_robots = env.curr_env.num_robots
    
    while True:
        data,succ=ray.get(data_server.sample.remote("imitation_{}_{}".format(map_name,num_robots),rollout_length))
        if succ:
            break
        else:
            Logger.error("fail to sample data from data server in rollout_func_BC")
    
    step_data_list=[]
    
    for datum in data:
        if datum is None:
            Logger.error("datum is None")
            continue
        # _map_name=datum["map_name"]
        # _num_robots=datum["num_robots"]

        # assert map_name==_map_name and num_robots==_num_robots
        
        # env.set_curr_env2(map_name,num_robots,False)
        
        # num_robots
        curr_positions=datum["curr_positions"]
        curr_positions_y=curr_positions//env.map.width
        curr_positions_x=curr_positions%env.map.width
        # num_robots,2
        curr_positions=np.stack([curr_positions_y,curr_positions_x],axis=-1)
        target_positions=datum["target_positions"]
        target_positions_y=target_positions//env.map.width
        target_positions_x=target_positions%env.map.width
        target_positions=np.stack([target_positions_y,target_positions_x],axis=-1)
        priorities=datum["priorities"]
        actions=torch.tensor(datum["actions"], dtype=torch.int32, device=env.device)
        
        custom_reset_config = {
            "curr_positions":curr_positions,
            "target_positions":target_positions,
            "priorities":priorities
        }
        
        env_rets = env_reset(env,behavior_policies,custom_reset_config)

        step_data = {
            EpisodeKey.CUR_OBS: env_rets[rollout_desc.agent_id][EpisodeKey.NEXT_OBS],
            EpisodeKey.CUR_GLOBAL_OBS: env_rets[rollout_desc.agent_id][EpisodeKey.NEXT_GLOBAL_OBS],
            EpisodeKey.ACTION_MASK: env_rets[rollout_desc.agent_id][EpisodeKey.ACTION_MASK],
            EpisodeKey.ACTION: actions,
            EpisodeKey.DONE: env_rets[rollout_desc.agent_id][EpisodeKey.DONE],
            EpisodeKey.ACTOR_RNN_STATE: env_rets[rollout_desc.agent_id][EpisodeKey.ACTOR_RNN_STATE],
            EpisodeKey.CRITIC_RNN_STATE: env_rets[rollout_desc.agent_id][EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.GUIDING_ACTION: actions,
        }

        step_data_list.append(step_data)

        
    episode=stack_step_data(step_data_list,{})

    # for k,v in episode.items():
    #     print(k,v.shape)
    
    submit_episode(data_server, episode, rollout_desc)
    
    results={"results":[]}
    return results

def rollout_func_BC_resim(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: MultiLMAPFEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs    
):
    sync_rnd_val=kwargs.get("sync_rnd_val",None)
    instance=kwargs.get("instance", None)
    
    env.set_eval(eval)
    
    if instance is None:
        if sync_rnd_val is not None and not eval:
            # TODO(rivers): it is a bad idea to use rollout_epoch to sync here?
            env.set_curr_env(sync_rnd_val)
        else:
            env.set_curr_env(None)
    else:
        map_name, num_robots=instance
        env.set_curr_env2(map_name, num_robots, verbose=False)
    
    map_name = env.curr_env.map.name
    num_robots = env.curr_env.num_robots
    
    while True:
        data,succ=ray.get(data_server.sample.remote("imitation_{}_{}".format(map_name,num_robots),1))
        if succ:
            break
        else:
            Logger.error("fail to sample data from data server in rollout_func_BC")
    
    step_data_list=[]
    
    # datum={
    #     "map_name": _map_name,
    #     "num_robots": _num_robots,
    #     "curr_positions":_curr_positions[:200],
    #     "target_positions":_target_positions[:200],
    #     "priorities":_priorities[:200],
    #     "actions":_actions[:200]
    # }
    
    data=data[0]
    seed=data["seed"]
    
    sampled_steps=np.random.choice(len(data["curr_positions"]),size=rollout_length,replace=False)
    sampled_step_masks=np.zeros(len(data["curr_positions"]),dtype=bool)
    sampled_step_masks[sampled_steps]=True
    
    # NOT support now
    actor_rnn_states=None
    critic_rnn_states=None
    for step in range(len(data["curr_positions"])):
         # num_robots
        curr_positions=data["curr_positions"][step]
        curr_positions_y=curr_positions//env.map.width
        curr_positions_x=curr_positions%env.map.width
        # num_robots,2
        curr_positions=np.stack([curr_positions_y,curr_positions_x],axis=-1)
        target_positions=data["target_positions"][step]
        target_positions_y=target_positions//env.map.width
        target_positions_x=target_positions%env.map.width
        target_positions=np.stack([target_positions_y,target_positions_x],axis=-1)
        priorities=data["priorities"][step]
        actions=torch.tensor(data["actions"][step], dtype=torch.int32, device=env.device)
        
        custom_reset_config={
            "seed": seed,
            "curr_positions": curr_positions,
            "target_positions": target_positions,
            "priorities": priorities
        }
        
        if step==0:
            env_rets = env_reset(env,behavior_policies,custom_reset_config)
            actor_rnn_states=env_rets[rollout_desc.agent_id][EpisodeKey.ACTOR_RNN_STATE]
            critic_rnn_states=env_rets[rollout_desc.agent_id][EpisodeKey.CRITIC_RNN_STATE]
        else:
            env_rets = env.sync_step(custom_reset_config)
            
        step_data = {
            EpisodeKey.CUR_OBS: env_rets[rollout_desc.agent_id][EpisodeKey.NEXT_OBS],
            EpisodeKey.CUR_GLOBAL_OBS: env_rets[rollout_desc.agent_id][EpisodeKey.NEXT_GLOBAL_OBS],
            EpisodeKey.ACTION_MASK: env_rets[rollout_desc.agent_id][EpisodeKey.ACTION_MASK],
            EpisodeKey.ACTION: actions,
            EpisodeKey.DONE: env_rets[rollout_desc.agent_id][EpisodeKey.DONE],
            EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states,
            EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states,
            EpisodeKey.GUIDING_ACTION: actions,
        }

        if sampled_step_masks[step]:
            step_data_list.append(step_data)
        
    episode=stack_step_data(step_data_list,{})

    # for k,v in episode.items():
    #     print(k,v.shape)
    if data_server is not None:
        submit_episode(data_server, episode, rollout_desc)
                
    # stats = env.get_episode_stats()
    # result = {
    #     "main_agent_id": rollout_desc.agent_id,
    #     "policy_ids": "_imitation",
    #     "map_name": env.map.name,
    #     "num_robots": env.num_robots,
    #     "stats": stats,
    # }
    
    results={"results":[]}
    return results

def rollout_func_BC_RNN(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: MultiLMAPFEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs    
):
    sync_rnd_val=kwargs.get("sync_rnd_val",None)
    instance=kwargs.get("instance", None)
    
    device=kwargs.get("device",None)
    if device is None:
        policy_device="cuda"
    else:
        policy_device=device
    
    if instance is None:
        if sync_rnd_val is not None and not eval:
            # TODO(rivers): it is a bad idea to use rollout_epoch to sync here?
            env.set_curr_env(sync_rnd_val)
        else:
            env.set_curr_env(None)
    else:
        map_name, num_robots=instance
        env.set_curr_env2(map_name, num_robots, verbose=False)
    
    map_name = env.curr_env.map.name
    num_robots = env.curr_env.num_robots
    
    while True:
        data,succ=ray.get(data_server.sample.remote("imitation_{}_{}".format(map_name,num_robots),1))
        if succ:
            break
        else:
            Logger.error("fail to sample data from data server in rollout_func_BC")
            
            
    policy_ids = OrderedDict()
    feature_encoders = OrderedDict()
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id
        policy.eval()
        # TODO: the number of devices should be configurable
        behavior_policies[agent_id]=(policy_id,policy.to_device(policy_device,in_place=True))
    
    step_data_list=[]
    actor_rnn_states=None
    critic_rnn_states=None
    
    data=data[0]
    
    _map_name=data["map_name"]
    _num_robots=data["num_robots"]
    assert map_name==_map_name and num_robots==_num_robots
    
    
     # L, num_robots
    curr_positions=data["curr_positions"]
    curr_positions_y=curr_positions//env.map.width
    curr_positions_x=curr_positions%env.map.width
    # L, num_robots,2
    curr_positions=np.stack([curr_positions_y,curr_positions_x],axis=-1)
    target_positions=data["target_positions"]
    target_positions_y=target_positions//env.map.width
    target_positions_x=target_positions%env.map.width
    target_positions=np.stack([target_positions_y,target_positions_x],axis=-1)
    priorities=data["priorities"]
    actions=torch.tensor(data["actions"], dtype=torch.int32, device=env.device)
    
    for idx in range(curr_positions.shape[0]):  
        custom_reset_config = {
            "curr_positions":curr_positions[idx],
            "target_positions":target_positions[idx],
            "priorities":priorities[idx]
        }
        
        step_data = env_reset(env,behavior_policies,custom_reset_config)
        
        if actor_rnn_states is not None:
            step_data[rollout_desc.agent_id][EpisodeKey.ACTOR_RNN_STATE]=actor_rnn_states
            
        if critic_rnn_states is not None:
            step_data[rollout_desc.agent_id][EpisodeKey.CRITIC_RNN_STATE]=critic_rnn_states
        
        policy_inputs = rename_fields(
            step_data, 
            [EpisodeKey.NEXT_OBS, EpisodeKey.NEXT_GLOBAL_OBS], 
            [EpisodeKey.CUR_OBS, EpisodeKey.CUR_GLOBAL_OBS]
        )
        
        policy_outputs = {}
        global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            policy_outputs[agent_id] = policy.compute_action(
                inference=True, 
                explore=not eval,
                to_numpy=False,
                step = kwargs.get('rollout_epoch', 0),
                **policy_inputs[agent_id]
            )
        actor_rnn_states=policy_outputs[rollout_desc.agent_id][EpisodeKey.ACTOR_RNN_STATE]
        critic_rnn_states=policy_outputs[rollout_desc.agent_id][EpisodeKey.CRITIC_RNN_STATE]
        global_timer.time("inference_start", "inference_end", "inference")


        _step_data = {
            EpisodeKey.CUR_OBS: step_data[rollout_desc.agent_id][EpisodeKey.CUR_OBS],
            EpisodeKey.CUR_GLOBAL_OBS: step_data[rollout_desc.agent_id][EpisodeKey.CUR_GLOBAL_OBS],
            EpisodeKey.ACTION_MASK: step_data[rollout_desc.agent_id][EpisodeKey.ACTION_MASK],
            EpisodeKey.ACTION: policy_outputs[rollout_desc.agent_id][EpisodeKey.ACTION],
            EpisodeKey.DONE: step_data[rollout_desc.agent_id][EpisodeKey.DONE],
            EpisodeKey.ACTOR_RNN_STATE: policy_outputs[rollout_desc.agent_id][EpisodeKey.ACTOR_RNN_STATE],
            EpisodeKey.CRITIC_RNN_STATE: policy_outputs[rollout_desc.agent_id][EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.GUIDING_ACTION: actions[idx],
        }

        step_data_list.append(_step_data)

        
    episode=stack_step_data(step_data_list,{})

    # for k,v in episode.items():
    #     print(k,v.shape)
    
    submit_episode(data_server, episode, rollout_desc)
    
    results={"results":[]}
    return results


def rollout_func_for_Dagger(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: MultiLMAPFEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs    
):
    sync_rnd_val=kwargs.get("sync_rnd_val",None)
    
    ret_dagger=None
    # generate new data
    rollout_epoch=kwargs["rollout_epoch"]
    # TODO: make them configurable
    if (rollout_epoch-1)%20==0:
        Logger.warn("[Rollout Epoch {}] Dagger collects new data".format(rollout_epoch))
        envs=kwargs["envs"]
        data_gen_env=envs[1]
        # full simulation
        data_gen_rollout_length=envs[1].rollout_length
        kwargs["episode_mode"]="dagger"
        kwargs["collect_data"]=True
        ret_dagger=rollout_func(
            True,
            rollout_worker,
            rollout_desc,
            data_gen_env,
            behavior_policies,
            data_server,
            data_gen_rollout_length,
            **kwargs    
        )
        
        
        imitation_data=ret_dagger.pop("imitation_data")
        # TODO submit it to data server
        submit_episode_bc(data_server, imitation_data)
    
    # return ret_dagger

    # sample from imitation dataset and train the network
    ret_bc=rollout_func_BC(
        eval,
        rollout_worker,
        rollout_desc,
        env,
        behavior_policies,
        data_server,
        rollout_length,
        **kwargs    
    )

    # return ret_BC
    if ret_dagger is None:
        ret_dagger=ret_bc
    
    return ret_dagger

# TO make it clear, I separate the rollout_func for WPPL
def rollout_func_for_WPPL(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: MultiLMAPFEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs
):
    """
    TODO(jh): modify document

    Rollout in simultaneous mode, support environment vectorization.

    :param VectorEnv env: The environment instance.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A dict of rollout information.
    """
    
    env.set_eval(eval)
    
    device=kwargs.get("device",None)
    if device is None:
        policy_device="cuda"
    else:
        policy_device=device
    
    sync_rnd_val=kwargs.get("sync_rnd_val",None)  
    rollout_epoch = kwargs.get("rollout_epoch",None)

    sample_length = kwargs.get("sample_length", rollout_length)
    render = kwargs.get("render", False)
    assert render == False
    if render:
        env.render()
    
    # either use PIBT or PIBT-RL
    assert env.get_pibt_func() in ["guard","solve"]

    episode_mode = kwargs.get('episode_mode','traj')
    verbose = kwargs.get("verbose", False)
    assert episode_mode=='traj'

    policy_ids = OrderedDict()
    feature_encoders = OrderedDict()
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id
        policy.eval()
        # TODO: the number of devices should be configurable
        behavior_policies[agent_id]=(policy_id,policy.to_device(policy_device,in_place=True))

    custom_reset_config = {
        # "feature_encoders": feature_encoders,
        # "main_agent_id": rollout_desc.agent_id
    }
    
    assert "curr_positions" in kwargs and "target_positions" in kwargs and "priorities" in kwargs
    
    if "curr_positions" in kwargs:
        custom_reset_config["curr_positions"]=kwargs["curr_positions"]
    if "target_positions" in kwargs:
        custom_reset_config["target_positions"]=kwargs["target_positions"]
    if "priorities" in kwargs:
        custom_reset_config["priorities"]=kwargs["priorities"]
    
    # assert rollout_length<=env.rollout_length,"rollout length {} should be less or equal to env rollout length {} during evaluation".format(rollout_length,env.rollout_length)
    # we add a shallow_copy to ensure correctness in case that step_data is modified by some in-place operation.
    step_data = env_reset(env,behavior_policies,custom_reset_config)
    # env.set_prev_step_data(shallow_copy(step_data))

    step = 0
    step_data_list = []
    results = []
    paths=[]
    first_step_policy_output=None
    episode=None

    curr_positions=env.curr_positions.clone()
    curr_positions=curr_positions[...,0]*env.map.width+curr_positions[...,1]
    paths.append(curr_positions.cpu().numpy())

    # collect until rollout_length
    # TODO: we need to carefully deal with the termination of the environment
    # because we need to discard the last step data if the episode is terminated
    for step in range(rollout_length):
        policy_inputs = rename_fields(
            step_data, 
            [EpisodeKey.NEXT_OBS, EpisodeKey.NEXT_GLOBAL_OBS], 
            [EpisodeKey.CUR_OBS, EpisodeKey.CUR_GLOBAL_OBS]
        )
        policy_outputs = {}
        # global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            if env.get_pibt_func()=="solve":
                # just some dummy values if PIBT
                policy_outputs[agent_id] = {
                    EpisodeKey.ACTION: None,
                    EpisodeKey.STATE_VALUE: None
                }
            else:
                policy_outputs[agent_id] = policy.compute_action(
                    inference=True, 
                    explore=not eval,
                    to_numpy=False,
                    step = kwargs.get('rollout_epoch', 0),
                    **policy_inputs[agent_id]
                )
        
        if step==0:
            first_step_policy_output=policy_outputs[rollout_desc.agent_id]

        # global_timer.time("inference_start", "inference_end", "inference")

        actions = select_fields(policy_outputs, [EpisodeKey.ACTION])

        # global_timer.record("env_step_start")
        env_rets = env.step(actions)
        # global_timer.time("env_step_start", "env_step_end", "env_step")

        curr_positions=env.curr_positions.clone()
        curr_positions=curr_positions[...,0]*env.map.width+curr_positions[...,1]
        paths.append(curr_positions.cpu().numpy())

        if verbose:
            Logger.info("env {} step {}'s stats: {}".format(env.id,step, env.get_episode_stats()))

        # record data after env step
        step_data = update_fields(
            step_data, select_fields(env_rets, [EpisodeKey.REWARD])
        )
        step_data = update_fields(
            step_data,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTION, EpisodeKey.ACTION_LOG_PROB, EpisodeKey.STATE_VALUE],
            ),
        )

        if not eval:
            # save data of trained agent for training
            step_data_list.append(step_data[rollout_desc.agent_id])
                
        step_data = update_fields(
            step_data, select_fields(env_rets, [EpisodeKey.DONE])
        )

        # record data for next step
        step_data = update_fields(
            env_rets,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTOR_RNN_STATE, EpisodeKey.CRITIC_RNN_STATE],
            ),
        )
        
        # we add a shallow_copy to ensure correctness in case that step_data is modified by some in-place operation.
        # env.set_prev_step_data(shallow_copy(step_data))
    
    if not eval:            #collect after rollout done
        #TODO(rivers): should we set done forcely here? because we don't care about delay beyond window anymore.
        # step_data_list[-1][EpisodeKey.DONE][...]=True
        
        assert len(step_data_list)>=1, "step_data_list should not be empty here"
        
        # call policy.compuate_action again to collect the state value for the last step data.
        policy_inputs = rename_fields(
            step_data, 
            [EpisodeKey.NEXT_OBS, EpisodeKey.NEXT_GLOBAL_OBS], 
            [EpisodeKey.CUR_OBS, EpisodeKey.CUR_GLOBAL_OBS]
        )
        # global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            if env.get_pibt_func()=="solve":
                # just some dummy values if PIBT
                policy_outputs[agent_id] = {
                    EpisodeKey.ACTION: None,
                    EpisodeKey.STATE_VALUE: None
                }
            else:
                policy_outputs[agent_id] = policy.compute_action(
                    inference=True, 
                    explore=not eval,
                    to_numpy=False,
                    step = kwargs.get('rollout_epoch', 0),
                    **policy_inputs[agent_id]
                )

        # global_timer.time("inference_start", "inference_end", "inference")
        step_data=update_fields(step_data,select_fields(policy_outputs,[EpisodeKey.STATE_VALUE]))
        
        # compute advatanges here
        episode = step_data_list
        episode.append(step_data[rollout_desc.agent_id])
        
    # print(env.episode_log)
    # env.episode_log.dump("test_log.json")
    
    if eval or env.is_terminated():
        stats = env.get_episode_stats()
        result = {
            "main_agent_id": rollout_desc.agent_id,
            "policy_ids": policy_ids,
            "stats": stats,
        }
        results.append(result)
        # step_data = env_reset(env,behavior_policies,custom_reset_config)
        # env.set_prev_step_data(step_data)

    # return restuls
    results={"results":results}      
    # return paths

    paths=np.array(paths,dtype=int)
    paths=paths.transpose().reshape(-1).tolist()
    results["paths"]=paths
    
    results["first_step_policy_output"]=first_step_policy_output
    results["episode"]=episode
    
    return results


def rollout_func_for_WPPL_training_data(
    rollout_desc: RolloutDesc,
    env: MultiLMAPFEnv,
    rollout_length,
    step_data_list,
    wppl_actions,
    **kwargs
):
    """
    TODO(jh): modify document

    Rollout in simultaneous mode, support environment vectorization.

    :param VectorEnv env: The environment instance.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A dict of rollout information.
    """
    
    # either use PIBT or PIBT-RL
    old_pibt_func = env.get_pibt_func()
    
    env.set_pibt_func("none")
    
    custom_reset_config = {
        # "feature_encoders": feature_encoders,
        # "main_agent_id": rollout_desc.agent_id
    }
    
    assert "curr_positions" in kwargs and "target_positions" in kwargs and "priorities" in kwargs
    
    if "curr_positions" in kwargs:
        custom_reset_config["curr_positions"]=kwargs["curr_positions"]
    if "target_positions" in kwargs:
        custom_reset_config["target_positions"]=kwargs["target_positions"]
    if "priorities" in kwargs:
        custom_reset_config["priorities"]=kwargs["priorities"]
    
    assert rollout_length<=env.rollout_length,"rollout length {} should be less or equal to env rollout length {} during evaluation".format(rollout_length,env.rollout_length)

    # we add a shallow_copy to ensure correctness in case that step_data is modified by some in-place operation.
    env.reset(custom_reset_config)

    # collect until rollout_length
    # TODO: we need to carefully deal with the termination of the environment
    # because we need to discard the last step data if the episode is terminated
    # collecting rewards
    for step in range(rollout_length):
        actions={
            rollout_desc.agent_id: {
                EpisodeKey.ACTION: wppl_actions[step]
            }
        }
        global_timer.record("env_step_start")
        env.step(actions)
        global_timer.time("env_step_start", "env_step_end", "env_step")

    total_rewards=torch.mean(env.total_rewards)/rollout_length
    
    last_step_data={
        rollout_desc.agent_id: step_data_list[-1]
    }
    step_data_list=step_data_list[:-1]
    
    for step_data in step_data_list:
        if EpisodeKey.REWARD in step_data:
            step_data[EpisodeKey.REWARD][:]=0
            
    step_data_list[-1][EpisodeKey.REWARD][:]=total_rewards
    step_data_list[-1][EpisodeKey.DONE][:]=True
    
    episode=pack_episode(step_data_list, last_step_data, rollout_desc, gae_gamma=env.gae_gamma, gae_lambda=env.gae_lambda)

    env.set_pibt_func(old_pibt_func)
    
    return episode
