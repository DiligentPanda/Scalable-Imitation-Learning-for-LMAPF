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

import numpy as np
from collections import defaultdict
from light_malib.training.data_generator import (
    recurrent_generator,
    simple_data_generator,
)
from .loss import MAPPOLoss
import torch
import functools
from light_malib.utils.logger import Logger
from light_malib.utils.timer import global_timer
from ..return_compute import compute_return
from ..common.trainer import Trainer
from light_malib.registry import registry
from light_malib.utils.episode import EpisodeKey

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@registry.registered(registry.TRAINER)
class MAPPOTrainer(Trainer):
    def __init__(self, tid):
        super().__init__(tid)
        self.id = tid
        # TODO(jh)
        self._loss = MAPPOLoss()

    def optimize(self, batch, **kwargs):
        total_opt_result = defaultdict(list)
        policy = self.loss.policy
        
        ppo_epoch = policy.custom_config["ppo_epoch"]
        num_mini_batch = policy.custom_config["num_mini_batch"]  # num_mini_batch
        kl_early_stop = policy.custom_config.get("kl_early_stop", None)
        assert (
            kl_early_stop is None
        ), "TODO(jh): kl early stop is not supported is current distributed implmentation."

        # # move data to gpu
        # global_timer.record("move_to_gpu_start")
        # for key, value in batch.items():
        #     if isinstance(value, np.ndarray):
        #         value = torch.FloatTensor(value)
        #     batch[key] = value.to(policy.device)
        
        # if EpisodeKey.CUR_STATE not in batch:
        #     batch[EpisodeKey.CUR_STATE]=batch[EpisodeKey.CUR_OBS]
        # global_timer.time("move_to_gpu_start", "move_to_gpu_end", "move_to_gpu")
        
        with torch.no_grad():
            if EpisodeKey.ADVANTAGE in batch:
                for key in [EpisodeKey.ADVANTAGE, EpisodeKey.RETURN, EpisodeKey.STATE_VALUE]:
                    value = batch[key]
                    if isinstance(value, np.ndarray):
                        value = torch.FloatTensor(value)
                    batch[key] = value.to(policy.device)
                
                # normalize advantages
                advantages = batch[EpisodeKey.ADVANTAGE].to(policy.device)
                batch[EpisodeKey.ADVANTAGE] = (advantages - advantages.mean()) / (1e-9 + advantages.std())

                # normalize returns
                if policy.custom_config["use_popart"]:
                    returns = batch[EpisodeKey.RETURN].to(policy.device)
                    shape = returns.shape
                    batch[EpisodeKey.RETURN] = policy.value_normalizer(returns.reshape(-1,shape[-1])).reshape(shape)
                    values = batch[EpisodeKey.STATE_VALUE].to(policy.device)
                    shape = values.shape
                    batch[EpisodeKey.STATE_VALUE] = policy.value_normalizer(values.reshape(-1,shape[-1])).reshape(shape)
                    
        kl_diff = 0
        for i_epoch in range(ppo_epoch):
            # NOTE(jh): for backward compatibility, when return_mode="new_gae", only call return_compute once.
            # if i_epoch==0 or policy.custom_config["return_mode"] in ["new_gae_trace"]:
            #     batch_with_return=self._compute_return(policy, batch) 
            # The return is computed in the rollout_func
                   
            # TODO: we probably want some fancy sampling methods.
            data_generator_fn=self._get_data_generator(policy, batch, num_mini_batch, shuffle=False)
                
            for idx, mini_batch in enumerate(data_generator_fn()):
                # TODO: compute_advantages: adv normalization/ value normalization etc.
                # move data to gpu
                global_timer.record("move_to_gpu_start")
                for key, value in mini_batch.items():
                    if isinstance(value, np.ndarray):
                        value = torch.FloatTensor(value)
                    mini_batch[key] = value.to(policy.device)
                if EpisodeKey.CUR_STATE not in batch:
                    mini_batch[EpisodeKey.CUR_STATE]=mini_batch[EpisodeKey.CUR_OBS]
                if EpisodeKey.CUR_GLOBAL_STATE not in batch:
                    mini_batch[EpisodeKey.CUR_GLOBAL_STATE]=mini_batch[EpisodeKey.CUR_GLOBAL_OBS]
                global_timer.time("move_to_gpu_start", "move_to_gpu_end", "move_to_gpu")
                
                global_timer.record("loss_start")
                step_optimizer = (idx==num_mini_batch-1)
                tmp_opt_result = self.loss(mini_batch, step_optimizer=step_optimizer, grad_accum_steps=num_mini_batch)
                global_timer.time("loss_start", "loss_end", "loss")                    
                
                for k, v in tmp_opt_result.items():
                    total_opt_result[k].append(v)

            if "approx_kl" in tmp_opt_result:
                if i_epoch == 0:
                    start_kl = tmp_opt_result["approx_kl"]
                else:
                    kl_diff += tmp_opt_result["approx_kl"] - start_kl
                    start_kl = tmp_opt_result["approx_kl"]

                if (
                    kl_early_stop is not None
                    and tmp_opt_result["approx_kl"] > kl_early_stop
                ):
                    break

                total_opt_result["kl_diff"] = kl_diff
            total_opt_result["training_epoch"] = i_epoch + 1

        # TODO(ziyu & ming): find a way for customize optimizer and scheduler
        # but now it doesn't affect the performance ...

        # TODO(jh)
        # if kwargs["lr_decay"]:
        #     epoch = kwargs["rollout_epoch"]
        #     total_epoch = kwargs["lr_decay_epoch"]
        #     assert total_epoch is not None
        #     update_linear_schedule(
        #         self.loss.optimizers["critic"],
        #         epoch,
        #         total_epoch,
        #         self.loss._params["critic_lr"],
        #     )
        #     update_linear_schedule(
        #         self.loss.optimizers["actor"],
        #         epoch,
        #         total_epoch,
        #         self.loss._params["actor_lr"],
        #     )

        return total_opt_result

    def preprocess(self, batch, **kwargs):
        pass

    def _compute_return(self, policy, batch):
        # compute return
        global_timer.record("compute_return_start")
        new_batch = compute_return(policy, batch)
        global_timer.time(
            "compute_return_start", "compute_return_end", "compute_return"
        )
        return new_batch        

    def _get_data_generator(self, policy, new_batch, num_mini_batch, shuffle):
        # build data generator
        if policy.custom_config["use_rnn"]:
            data_generator_fn = functools.partial(
                recurrent_generator,
                new_batch,
                num_mini_batch,
                policy.custom_config["rnn_data_chunk_length"],
                policy.device,
                shuffle
            )
        else:
            data_generator_fn = functools.partial(
                simple_data_generator, new_batch, num_mini_batch, policy.device, shuffle
            )
        
        return data_generator_fn