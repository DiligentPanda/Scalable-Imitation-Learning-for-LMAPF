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

from collections import OrderedDict

from light_malib import rollout, agent, training, agent, buffer
from light_malib.agent import AgentManager
from light_malib.agent.agent import Agent, Agents
from light_malib.evaluation.evaluation_manager_ppo import EvaluationManagerPPO as EvaluationManager
from light_malib.agent.policy_data.policy_data_manager import PolicyDataManager
from light_malib.framework.scheduler.psro_scheduler import PSROScheduler
from light_malib.framework.scheduler.seq_league_scheduler import SeqLeagueScheduler
from light_malib.framework.scheduler.stopper.common import max_step_stoppper
from light_malib.framework.scheduler.simple_scheduler import SimpleScheduler
from light_malib.utils.desc.task_desc import TrainingDesc, RolloutDesc
import ray
import numpy as np
from light_malib.utils.distributed import get_resources
from light_malib.utils.logger import Logger
from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.envs.LMAPF.env import MultiLMAPFEnv

class PPORunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.framework_cfg = self.cfg.framework
        self.id = self.framework_cfg.name

        ###### Initialize Components #####

        RolloutManager = ray.remote(
            **get_resources(cfg.rollout_manager.distributed.resources)
        )(rollout.RolloutManager)
        TrainingManager = ray.remote(
            **get_resources(cfg.training_manager.distributed.resources)
        )(training.TrainingManager)
        DataServer = ray.remote(**get_resources(cfg.data_server.distributed.resources))(
            buffer.DataServer
        )
        PolicyServer = ray.remote(
            **get_resources(cfg.policy_server.distributed.resources)
        )(buffer.PolicyServer)

        # the order of creation is important? cannot have circle reference
        # create agents
        agents = AgentManager.build_agents(self.cfg.agent_manager)
        
        self.data_server = DataServer.options(
            name="DataServer", max_concurrency=self.cfg.rollout_manager.num_workers+5
        ).remote("DataServer", self.cfg.data_server)

        self.policy_server = PolicyServer.options(
            name="PolicyServer", max_concurrency=self.cfg.rollout_manager.num_workers+5
        ).remote("PolicyServer", self.cfg.policy_server, agents)

        self.rollout_manager = RolloutManager.options(
            name="RolloutManager", max_concurrency=self.cfg.rollout_manager.num_workers+5
        ).remote("RolloutManager", self.cfg.rollout_manager, agents)

        self.training_manager = TrainingManager.options(
            name="TrainingManager", max_concurrency=5
        ).remote("TrainingManager", self.cfg.training_manager)

        # NOTE: self.agents is not shared with remote actors.
        self.agent_manager = AgentManager(self.cfg.agent_manager)
        self.policy_data_manager = PolicyDataManager(
            self.cfg.policy_data_manager, self.agent_manager
        )
        self.evaluation_manager = EvaluationManager(
            self.cfg.evaluation_manager, self.agent_manager, self.policy_data_manager
        )

        # TODO(jh): scheduler is designed for future distributed purposes.
        if self.id in ["simple","simple_imitation","simple_dagger","simple_dagger_pibt"]:
            self.scheduler = SimpleScheduler(
                self.cfg.framework, self.agent_manager, self.policy_data_manager
            )
        else:
            raise NotImplementedError

        Logger.info("PPORunner {} initialized".format(self.id))

    def run(self):
        if self.id in ["simple_dagger","simple_dagger_pibt"]:
            self.scheduler.initialize(self.cfg.populations)
            
            num_iters=self.cfg.framework.max_dagger_iterations
            num_episodes_per_iter=self.cfg.framework.num_episodes_per_iter
            
            env=MultiLMAPFEnv(0, 0, self.cfg.rollout_manager.worker.envs[0], "cpu")
            import copy
            table_cfg=copy.deepcopy(self.cfg.data_server.table_cfg)
            table_cfg.update({"capacity":2**16,"sampler_type":"uniform","sample_max_usage": 1e8, "rate_limiter_cfg":{}})
            for map_name,num_agents in env.map_manager.instances_list:
                table_name="imitation_{}_{}".format(map_name,num_agents)
                ray.get(self.data_server.create_table.remote(table_name, table_cfg))
                
            # load imitation data
            if self.cfg.data_server.imitation_dataset:
                ray.get(self.data_server.load_database.remote("imitation",self.cfg.data_server.imitation_dataset))
            
            # load guiding policy
            if self.cfg.data_server.guiding_policy:
                model_path = self.cfg.data_server.guiding_policy
                policy_id = "guiding_policy"
                policy=MAPPO.load(model_path, env_agent_id="agent_0")
                ray.get(self.data_server.put.remote(policy_id,policy))

            self.evaluation_manager.eval()
            for iter_idx in range(num_iters):
                training_desc = self.scheduler.get_task()
                
                if iter_idx==0:
                    if self.id=="simple_dagger":
                        wppl_mode="PIBT-RL-LNS"
                    elif self.id=="simple_dagger_pibt":
                        wppl_mode="PIBT-LNS"
                    else:
                        raise NotImplementedError
                else:
                    wppl_mode="PIBT-RL-LNS-Guide"
                
                if not (iter_idx==0 and self.cfg.data_server.imitation_dataset):
                    rollout_desc = RolloutDesc(
                        0,
                        training_desc.agent_id,
                        training_desc.policy_id,
                        training_desc.policy_distributions,
                        training_desc.share_policies,
                        stopper = None,
                        sync=False, # useless, we will collect data asyncly anyway
                        kwargs = {
                            "collect_data": True,
                            "wppl_mode": wppl_mode,
                            "num_episodes_per_iter": num_episodes_per_iter
                        } 
                    )                        
                
                    # collect new data: TODO: we should save the data
                    ray.get(self.rollout_manager.collect_data.remote(rollout_desc))
                
                # BC
                Logger.info("training_desc: {}".format(training_desc))
                training_task_ref = self.training_manager.train.remote(training_desc)
                ray.get(training_task_ref)
                self.scheduler.submit_result(None)
                
                # directly bootrap but not use dagger.
                if not self.cfg.data_server.guiding_policy:
                    for map_name,num_agents in env.map_manager.instances_list:
                        table_name="imitation_{}_{}".format(map_name,num_agents)
                        ray.get(self.data_server.remove_table.remote(table_name))
                    for map_name,num_agents in env.map_manager.instances_list:
                        table_name="imitation_{}_{}".format(map_name,num_agents)
                        ray.get(self.data_server.create_table.remote(table_name,table_cfg))
            
            # guiding policy will only be used in the first iteration, then we will boostrap
            if self.cfg.data_server.guiding_policy:
                policy_id = "guiding_policy"
                ray.get(self.data_server.pop.remote(policy_id))
        
        else:
            if self.id=="simple_imitation":
                # env=MultiLMAPFEnv(0, 0, self.cfg.rollout_manager.worker.envs[0], "cpu")
                # import copy
                # table_cfg=copy.deepcopy(self.cfg.data_server.table_cfg)
                # table_cfg.update({"capacity":2**20,"sampler_type":"uniform","sample_max_usage": 1e8, "rate_limiter_cfg":{}})
                # for map_name,num_agents in env.map_manager.instances_list:
                #     table_name="imitation_{}_{}".format(map_name,num_agents)
                #     ray.get(self.data_server.create_table.remote(table_name, table_cfg))
                
                # load imitation data
                ray.get(self.data_server.load_database.remote("imitation",self.cfg.data_server.imitation_dataset))
            
            if self.cfg.data_server.guiding_policy:
                model_path = self.cfg.data_server.guiding_policy
                policy_id = "guiding_policy"
                policy=MAPPO.load(model_path, env_agent_id="agent_0")
                ray.get(self.data_server.put.remote(policy_id,policy))
            
            self.scheduler.initialize(self.cfg.populations)
            if self.cfg.eval_only:
                self.evaluation_manager.eval(eval_more_metrics=True)
            else:
                self.evaluation_manager.eval()
                training_desc = self.scheduler.get_task()
                Logger.info("training_desc: {}".format(training_desc))
                training_task_ref = self.training_manager.train.remote(training_desc)
                ray.get(training_task_ref)
                self.scheduler.submit_result(None)
        
        Logger.info("PPORunner {} ended".format(self.id))

    def close(self):
        ray.get(self.training_manager.close.remote())
        ray.get(self.rollout_manager.close.remote())
