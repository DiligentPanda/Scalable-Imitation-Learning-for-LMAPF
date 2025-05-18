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
from light_malib.registry import registry
from light_malib.agent.agent_manager import AgentManager

from light_malib.agent.policy_data.policy_data_manager import PolicyDataManager
from light_malib.utils.logger import Logger
from light_malib.agent import Population
from light_malib.utils.desc.task_desc import TrainingDesc
import numpy as np
import importlib


class SimpleScheduler:
    """
    TODO(jh): abstract it later
    """

    def __init__(
        self, cfg, agent_manager: AgentManager, policy_data_manager: PolicyDataManager
    ):
        self.cfg = cfg
        self.agent_manager = agent_manager
        self.agents = self.agent_manager.agents
        assert len(self.agents)==1 and self.agent_manager.get_agent_ids()[0]=="agent_0"
        
        self.population_id = "default"
        self.policy_data_manager = policy_data_manager
        self.sync_training = self.cfg.get("sync_training", False)

        self._schedule = self._gen_schedule()

    def initialize(self, populations_cfg):
        # add populations
        for agent_id in self.agents.training_agent_ids:
            assert len(populations_cfg) == 1
            population_id = populations_cfg[0]["population_id"]
            assert population_id == self.population_id
            algorithm_cfg = populations_cfg[0]["algorithm"]
            self.agent_manager.add_new_population(
                agent_id, self.population_id, algorithm_cfg
            )

        for population_cfg in populations_cfg:
            population_id = population_cfg["population_id"]
            algorithm_cfg = population_cfg.algorithm
            policy_init_cfg = algorithm_cfg.get("policy_init_cfg", None)
            if policy_init_cfg is None:
                continue
            for agent_id, agent_policy_init_cfg in policy_init_cfg.items():
                agent_initial_policies = agent_policy_init_cfg.get(
                    "initial_policies", None
                )
                if agent_initial_policies is None:
                    continue
                for policy_cfg in agent_initial_policies:
                    policy_id = policy_cfg["policy_id"]
                    policy_dir = policy_cfg["policy_dir"]
                    self.agent_manager.load_policy(
                        agent_id, population_id, policy_id, policy_dir
                    )
                    Logger.info(f"Load initial policy {policy_id} from {policy_dir}")

        # generate the first policy for main agent
        for agent_id in self.agents.training_agent_ids:
            self.agent_manager.gen_new_policy(agent_id, self.population_id)

        # TODO(jh):Logger
        Logger.warning("after initialization:\n{}".format(self.agents))

    def _gen_schedule(self):
        generation_ctr=0
        while True:
            training_agent_id=self.agents.training_agent_ids[0]
            # gen new policy
            training_policy_id = self.agent_manager.gen_new_policy(
                training_agent_id, self.population_id
            )
            policy_distributions={}
            policy_distributions[training_agent_id] = {training_policy_id: 1.0}
            Logger.warning(
                "********** Generation[{}] Agent[{}] START **********".format(
                    generation_ctr, training_agent_id
                )
            )

            # TODO: add a stopper
            stopper = registry.get(registry.STOPPER, self.cfg.stopper.type)(
                policy_data_manager=self.policy_data_manager,
                **self.cfg.stopper.kwargs,
            )

            training_desc = TrainingDesc(
                training_agent_id,
                training_policy_id,
                policy_distributions,
                self.agents.share_policies,
                self.sync_training,
                stopper,
            )
            yield training_desc
            generation_ctr+=1

    def get_task(self):
        try:
            task = next(self._schedule)
            return task
        except StopIteration:
            return None

    def submit_result(self, result):
        pass
