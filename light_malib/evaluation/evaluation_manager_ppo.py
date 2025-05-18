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

from light_malib import rollout
from light_malib.agent.policy_data.policy_data_manager import PolicyDataManager
from light_malib.utils.desc.task_desc import RolloutEvalDesc
from light_malib.utils.distributed import get_actor
import ray
import numpy as np

from light_malib.utils.logger import Logger
from .elo import EloManager
from .melo import MEloManager

# from open_spiel.python.egt import alpharank, utils as alpharank_utils
import nashpy as nash


class EvaluationManagerPPO:
    def __init__(self, cfg, agent_manager, policy_data_manager):
        self.cfg = cfg
        self.agents = agent_manager.agents
        self.policy_data_manager = policy_data_manager
        self.rollout_manager = get_actor("EvaluationManager", "RolloutManager")

    def eval(self):
        # generate tasks from payoff matrix
        rollout_eval_desc = self.generate_rollout_tasks()

        # call rollout_eval remotely
        eval_results = ray.get(
            self.rollout_manager.rollout_eval.remote(rollout_eval_desc)
        )

        # update policy data
        self.policy_data_manager.update_policy_data(
            eval_results
        )

    def generate_rollout_tasks(self):
        policy_combs = []
        for index_comb in [[0]]:
            if not self.agents.share_policies or self._ordered(index_comb):
                assert len(index_comb) == len(self.agents)
                print(index_comb,self.agents["agent_0"].policy_ids)
                policy_comb = {
                    agent_id: {agent.policy_ids[index_comb[i]]: 1.0}
                    for i, (agent_id, agent) in enumerate(self.agents.items())
                }
                policy_combs.append(policy_comb)

        Logger.warning(
            "Evaluation rollouts (num: {}) for {} policy combinations: {}".format(
                self.cfg.num_eval_rollouts, len(policy_combs), policy_combs
            )
        )
        rollout_eval_desc = RolloutEvalDesc(
            policy_combs, self.cfg.num_eval_rollouts, self.agents.share_policies
        )
        return rollout_eval_desc
