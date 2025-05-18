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

import threading
from light_malib.agent.agent import Agents
from light_malib.utils.desc.policy_desc import PolicyDesc
from readerwriterlock import rwlock
from light_malib.utils.logger import Logger
import numpy as np

class PolicyServer:
    """
    TODO(jh) This implementation is still problematic. we should rewrite it in asyncio's way, e.g. should use asyncio.Lock.
    Because there is not yield here, and no resouce contention, no lock is still correct.
    """

    def __init__(self, id, cfg, agents: Agents):
        self.id = id
        self.cfg = cfg
        self.agents = agents
        locks = (
            [rwlock.RWLockWrite()] * len(self.agents)
            if self.agents.share_policies
            else [rwlock.RWLockWrite() for i in range(len(self.agents))]
        )
        self.locks = {
            agent_id: lock for agent_id, lock in zip(self.agents.agent_ids, locks)
        }

        # agent_id
        self.population={
            
        }
        self.population_max_capacity=4
        self.population_sampling_probs=[0.4,0.3,0.2,0.1]
        
        self.use_population=True

        Logger.info("{} initialized".format(self.id))
        
    async def clear_population(self):
        self.population={}
        
    async def update_population(self,agent_id,policy_id, eval_result):
        policy_desc=await self.pull(self.id,agent_id,policy_id)
    
        policy_id = policy_desc.policy_id
        version = policy_desc.version
        key = "{}:{}".format(policy_id, version)
    
        Logger.info("update population with policy {} and eval result: {}".format(key,eval_result))
        
        if key in self.population:
            old_eval_result, old_policy_desc = self.population[key]
            eval_result=(eval_result+old_eval_result)*0.5
            self.population[key]=(eval_result, policy_desc)
        else:    
            if len(self.population)==self.population_max_capacity:
                # maintain heap
                min_eval_result=1e8
                min_key=None
                max_eval_result=-1e8
                max_key=None
                for _key, (_eval_result,_policy_desc) in self.population.items():
                    if _eval_result<min_eval_result:
                        min_eval_result=_eval_result
                        min_key=_key
                    if _eval_result>max_eval_result:
                        max_eval_result=_eval_result
                        max_key=_key
                self.population.pop(min_key)
            
            self.population[key]=(eval_result, policy_desc)
                   
        policy_desc = await self.sample_from_population()
        await self.push(self.id,policy_desc)
                    
    async def sample_from_population(self):
        policies=[]
        probs=[]
        for idx,(_key, (_eval_result,_policy_desc)) in enumerate(self.population.items()):
            policies.append((_eval_result,_key))
            probs.append(self.population_sampling_probs[idx])
    
        probs=np.array(probs)
        probs=probs/np.sum(probs)
    
        policies=sorted(policies,key=lambda x: x[0], reverse=True)
        
        policy_idx = np.random.choice(len(policies), replace=False, p=probs)
        policy_key=policies[policy_idx][1]
        
        policy_desc=self.population[policy_key][1]
        
        Logger.info("sample policy {} with eval results {} from {}".format(policies[policy_idx][1],policies[policy_idx][0],policies))
        return policy_desc
        

    async def push(self, caller_id, policy_desc: PolicyDesc):
        # Logger.debug("{} try to push({}) to policy server".format(caller_id,str(policy_desc)))
        agent_id = policy_desc.agent_id
        policy_id = policy_desc.policy_id
        #lock = self.locks[agent_id]
        self.agents[agent_id].policy_data[policy_id] = policy_desc
      
        # with lock.gen_wlock():
        #     old_policy_desc = self.agents[agent_id].policy_data.get(policy_id, None)
        #     if (
        #         old_policy_desc is None
        #         or old_policy_desc.version is None
        #         or old_policy_desc.version < policy_desc.version
        #     ):
        #         self.agents[agent_id].policy_data[policy_id] = policy_desc
        #     else:
        #         Logger.debug("{}::push() discard order policy".format(self.id))
        # Logger.debug("{} try to push({}) to policy server ends".format(caller_id,str(policy_desc)))

    async def pull(self, caller_id, agent_id, policy_id, old_version=None):
        # Logger.debug("{} try to pull({},{},{}) from policy server".format(caller_id,agent_id,policy_id,old_version))
        return self.agents[agent_id].policy_data[policy_id]
        
        # lock = self.locks[agent_id]
        # with lock.gen_rlock():
        #     if policy_id not in self.agents[agent_id].policy_data:
        #         ret = None
        #     else:
        #         policy_desc: PolicyDesc = self.agents[agent_id].policy_data[policy_id]
        #         if old_version is None or old_version < policy_desc.version:
        #             ret = policy_desc
        #         else:
        #             ret = None
        # # Logger.debug("{} try to pull({},{},{}) from policy server ends".format(caller_id,agent_id,policy_id,old_version))
        # return ret

    def dump_policy(self):
        pass
