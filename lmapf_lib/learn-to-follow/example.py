import argparse

from env.create_env import create_env_base
from env.custom_maps import MAPS_REGISTRY
from env.warehouse_wfi import create_warehouse_wfi_env
from utils.eval_utils import run_episode
from follower.training_config import EnvironmentMazes
from follower.inference import FollowerInferenceConfig, FollowerInference
from follower.preprocessing import follower_preprocessor
from follower_cpp.inference import FollowerConfigCPP, FollowerInferenceCPP
from follower_cpp.preprocessing import follower_cpp_preprocessor


def create_custom_env(cfg):
    env_cfg = EnvironmentMazes(with_animation=cfg.animation)
    env_cfg.grid_config.num_agents = cfg.num_agents
    env_cfg.grid_config.map_name = cfg.map_name
    env_cfg.grid_config.seed = cfg.seed
    env_cfg.grid_config.max_episode_steps = cfg.max_episode_steps
    if cfg.map_name == 'wfi_warehouse':
        return create_warehouse_wfi_env(env_cfg)
    return create_env_base(env_cfg)


def run_follower(env):
    follower_cfg = FollowerInferenceConfig()
    algo = FollowerInference(follower_cfg)

    env = follower_preprocessor(env, follower_cfg)

    return run_episode(env, algo)


def run_follower_cpp(env):
    follower_cfg = FollowerConfigCPP(path_to_weights='model/follower-lite', num_threads=6)
    algo = FollowerInferenceCPP(follower_cfg)

    env = follower_cpp_preprocessor(env, follower_cfg)

    return run_episode(env, algo)


def main():
    parser = argparse.ArgumentParser(description='Follower Inference Script')
    parser.add_argument('--animation', action='store_false', help='Enable animation (default: %(default)s)')
    parser.add_argument('--num_agents', type=int, default=128, help='Number of agents (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='wfi_warehouse', help='Map name (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=256,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--show_map_names', action='store_true', help='Shows names of all available maps')

    parser.add_argument('--algorithm', type=str, choices=['Follower', 'FollowerLite'], default='Follower',
                        help='Algorithm to use: "Follower" or "FollowerLite" (default: "Follower")')

    args = parser.parse_args()

    if args.show_map_names:
        for map_ in MAPS_REGISTRY:
            print(map_)
        print('wfi_warehouse')
        return

    if args.algorithm == 'FollowerLite':
        print(run_follower_cpp(create_custom_env(args)))
    else:  # Default to 'Follower'
        print(run_follower(create_custom_env(args)))


if __name__ == '__main__':
    main()
