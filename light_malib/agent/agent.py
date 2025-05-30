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
import pprint
from .policy_factory import PolicyFactory
from light_malib.utils.logger import Logger


class Agents(OrderedDict):
    def __init__(self, agents_dict, share_policies):
        self.share_policies = share_policies
        self.agent_ids = [agent_id for agent_id in agents_dict]
        self.agent_id_set = set(self.agent_ids)
        super().__init__(agents_dict)

    @property
    def training_agent_ids(self):
        return [self.agent_ids[0]] if self.share_policies else self.agent_ids

    def __reduce__(self):
        state = super().__reduce__()
        newstate = (
            self.__class__,
            ({}, None),
            {
                "share_policies": self.share_policies,
                "agent_ids": self.agent_ids,
            },  # state to __setstate__
            None,
            state[4],
        )  # items
        return newstate

    def __setstate__(self, state):
        self.share_policies = state["share_policies"]
        self.agent_ids = state["agent_ids"]
        self.agent_id_set = set(self.agent_ids)

    def __str__(self):
        s = "\n"
        for agent_id in self.training_agent_ids:
            s += "{}\n".format(self[agent_id])
        return s

    __repr__ = __str__


class Agent:
    def __init__(self, id):
        self.id = id
        self.policy_ids = []  # used in agent_manager
        self.policy_id2idx = {}
        self.policy_data = {}  # used in rollout_worker
        self.populations = OrderedDict()
        self.add_new_population("__all__", None, None)

    def gen_new_policy(self, population_id):
        population: Population = self.populations[population_id]
        policy_id, policy = population.gen_new_policy()
        return policy_id, policy

    def add_new_policy(self, population_id, policy_id):
        if policy_id in self.policy_id2idx:
            Logger.error(
                "Cannot add {}, which is already in the policy pool".format(policy_id)
            )
            raise Exception(
                "Cannot add {}, which is already in the policy pool".format(policy_id)
            )
        self.policy_ids.append(policy_id)
        self.policy_id2idx[policy_id] = len(self.policy_id2idx)
        self.policy_data[policy_id] = {}
        self.populations[population_id].add_new_policy(policy_id)
        self.populations["__all__"].add_new_policy(policy_id)

    def add_new_population(self, population_id, algorithm_cfg, policy_server):
        if population_id in self.populations:
            Logger.error(
                "Cannot add {}, which is already in the population pool {}".format(
                    population_id, self.populations
                )
            )
            raise Exception(
                "Cannot add {}, which is already in the population pool {}".format(
                    population_id, self.populations
                )
            )
        self.populations[population_id] = Population(
            population_id, self, algorithm_cfg, policy_server
        )

    def __str__(self):
        s = "<A {}>\npolicy_ids:\n{}\npopulations:\n".format(self.id, self.policy_ids)
        for population_id, population in self.populations.items():
            s += "{}".format(population)
        return s

    __repr__ = __str__


class Population:
    def __init__(self, id, agent, algorithm_cfg, policy_server):
        self.id = id
        self.agent = agent
        self.policy_ids = []
        if algorithm_cfg is not None and policy_server is not None:
            self.policy_factory = PolicyFactory(
                self.agent.id, self.id, algorithm_cfg, policy_server
            )
        else:
            self.policy_factory = None

    def add_new_policy(self, policy_id):
        """
        TODO(jh): maybe in the future we could save something like how this model is trained as policy meta-data.
        """
        self.policy_ids.append(policy_id)

    def gen_new_policy(self):
        policy_id, policy = self.policy_factory.gen_new_policy()
        return policy_id, policy

    def __str__(self):
        return "<P {}> policy_ids:{}\n".format(self.id, self.policy_ids)

    __repr__ = __str__
