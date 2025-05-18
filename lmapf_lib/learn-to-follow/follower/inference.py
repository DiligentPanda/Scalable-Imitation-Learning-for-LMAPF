# noinspection PyUnresolvedReferences
from utils import fix_num_threads_issue

import json
from copy import deepcopy

from follower.training_config import Experiment
from follower.register_env import register_custom_components

import os
from argparse import Namespace
from collections import OrderedDict
from os.path import join

import numpy as np

from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from sample_factory.utils.utils import log
from pydantic import Extra, validator

from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

from follower.algorithm_utils import AlgoBase

from follower.register_training_utils import register_custom_model


class FollowerInferenceConfig(AlgoBase, extra=Extra.forbid):
    """
    A configuration class for the Proximal Policy Optimization (PPO) algorithm,
    with additional parameters specific to the Follower agent.

    Attributes:
    -----------
    name : Literal['Follower']
        A string literal specifying the name of the agent.
    path_to_weights : str
        A string specifying the path to the weights file used by the agent.
    planner_cfg : dict
    batched : bool
        A boolean indicating whether the agent should use a batched approach during training.
    """
    name: Literal['Follower'] = 'Follower'

    path_to_weights: str = "model/follower"
    preprocessing: str = 'FollowerPreprocessing'
    override_config: Optional[dict] = None
    training_config: Optional[Experiment] = None
    custom_path_to_weights: Optional[str] = None

    @classmethod
    def recursive_dict_update(cls, original_dict, update_dict):
        for key, value in update_dict.items():
            if key in original_dict and isinstance(original_dict[key], dict) and isinstance(value, dict):
                cls.recursive_dict_update(original_dict[key], value)
            else:
                if key not in original_dict:
                    raise ValueError(f"Key '{key}' does not exist in the original training config.")
                original_dict[key] = value

    @validator('training_config', always=True, pre=True)
    def load_training_config(cls, _, values, ):
        with open(join(values['path_to_weights'], 'config.json'), "r") as f:
            field_value = json.load(f)
        if values.get('override_config') is not None:
            cls.recursive_dict_update(field_value, deepcopy(values['override_config']))
        return field_value


class FollowerInference:
    """

    """

    def __init__(self, config):

        self.algo_cfg: FollowerInferenceConfig = config
        device = config.device

        register_custom_model()
        self.path = config.path_to_weights

        with open(join(self.path, 'config.json'), "r") as f:
            flat_config = json.load(f)
            self.exp = Experiment(**flat_config)
            flat_config = Namespace(**flat_config)
        env_name = self.exp.environment.env
        register_custom_components(env_name)
        config = flat_config

        config.num_envs = 1

        env = make_env_func_batched(config, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
        actor_critic = create_actor_critic(config, env.observation_space, env.action_space)
        actor_critic.eval()
        env.close()

        if device != 'cpu' and not torch.cuda.is_available():
            os.environ['OMP_NUM_THREADS'] = str(1)
            os.environ['MKL_NUM_THREADS'] = str(1)
            device = torch.device('cpu')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            log.warning('CUDA is not available, using CPU. This might be slow.')

        actor_critic.model_to_device(device)
        name_prefix = dict(latest="checkpoint", best="best")['latest']
        policy_index = 0 if 'policy_index' not in flat_config else flat_config.policy_index

        checkpoints = Learner.get_checkpoints(os.path.join(self.path, f"checkpoint_p{policy_index}"),
                                              f"{name_prefix}_*")
        # print(checkpoints), exit(0)
        if self.algo_cfg.custom_path_to_weights:
            checkpoints = [self.algo_cfg.custom_path_to_weights]

        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])
        log.info(f'Loaded {str(checkpoints)}')

        self.net = actor_critic
        self.device = device
        self.cfg = config

        self.rnn_states = None

    def act(self, observations):
        self.rnn_states = torch.zeros([len(observations), get_rnn_size(self.cfg)], dtype=torch.float32,
                                      device=self.device) if self.rnn_states is None else self.rnn_states

        obs = AttrDict(self.transform_dict_observations(observations))
        with torch.no_grad():
            policy_outputs = self.net(prepare_and_normalize_obs(self.net, obs), self.rnn_states)

        self.rnn_states = policy_outputs['new_rnn_states']
        return policy_outputs['actions'].cpu().numpy()

    def reset_states(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.rnn_states = None

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_parameters(self):
        return self.count_parameters(self.net)

    @staticmethod
    def transform_dict_observations(observations):
        """Transform list of dict observations into a dict of lists."""
        obs_dict = dict()
        if isinstance(observations[0], (dict, OrderedDict)):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            # handle flat observations also as dict
            obs_dict['obs'] = observations

        for key, x in obs_dict.items():
            obs_dict[key] = np.stack(x)

        return obs_dict

    def to_onnx(self, filename='follower.onnx'):
        self.net.eval()
        r = self.algo_cfg.training_config.preprocessing.network_input_radius
        log.info(f"Saving model with network_input_radius = {r}")
        d = 2 * r + 1
        obs_example = torch.rand(1, 2, d, d, device=self.device)
        rnn_example = torch.rand(1, 1, device=self.device)
        with torch.no_grad():
            q = self.net({'obs': obs_example}, rnn_example)
            print(q)
        input_names = ['obs', 'rnn_state']
        output_names = ['values', 'action_logits', 'log_prob_actions', 'actions', 'new_rnn_states']

        torch.onnx.export(self.net, ({'obs': obs_example}, rnn_example), filename,
                          input_names=input_names, output_names=output_names,
                          export_params=True)
