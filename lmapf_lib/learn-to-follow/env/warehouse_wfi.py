import numpy as np
from copy import deepcopy

from pogema.animation import AnimationConfig, AnimationMonitor

from env.create_env import ProvideGlobalObstacles, RuntimeMetricWrapper
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric

from pogema.wrappers.multi_time_limit import MultiTimeLimit

from pogema import GridConfig
from pogema.grid import GridLifeLong
from pogema.envs import Pogema


class WarehouseWFI(Pogema):

    def __init__(self, grid_config=GridConfig(num_agents=2)):
        grid_config, possible_locations = self.update_config(grid_config)
        assert grid_config.num_agents <= len(possible_locations['starts']), \
            f"Not enough possible locations for agents {grid_config.num_agents} > {len(possible_locations['starts'])}"
        assert grid_config.num_agents <= len(possible_locations['targets']), \
            f"Not enough possible locations for targets {grid_config.num_agents} > {len(possible_locations['targets'])}"
        super().__init__(grid_config)

        if grid_config.seed is not None:
            master_generator = np.random.default_rng(grid_config.seed)
            self.random_generators: list = [np.random.default_rng(master_generator.integers(0, 1e6)) for _ in
                                            range(grid_config.num_agents)]
        else:
            self.random_generators: list = [np.random.default_rng() for _ in range(grid_config.num_agents)]
        self.possible_locations = possible_locations

    def _generate_starts_goals(self):
        if self.grid_config.seed is not None:
            generator = np.random.default_rng(self.grid_config.seed)
        else:
            generator = np.random.default_rng()
        generator.shuffle(self.possible_locations['starts'])
        filled_positions = np.zeros(self.grid.obstacles.shape)

        for agent_idx in range(self.grid_config.num_agents):
            x, y = self.possible_locations['starts'][agent_idx]
            self.grid.positions_xy[agent_idx] = (x, y)
            filled_positions[x, y] = 1
            self.grid.finishes_xy[agent_idx] = self._get_new_goal(agent_idx)
        self.grid.positions = filled_positions
        self.grid._initial_xy = deepcopy(self.grid.positions_xy)

    def _get_new_goal(self, agent_idx):
        rng = self.random_generators[agent_idx]
        new_finish = rng.choice(self.possible_locations['targets'], 1)[0]
        return new_finish[0], new_finish[1]

    def _initialize_grid(self):
        self.grid: GridLifeLong = GridLifeLong(grid_config=self.grid_config)
        self._generate_starts_goals()

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        dones = [False] * self.grid_config.num_agents

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):
            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

            if self.grid.on_goal(agent_idx):
                self.grid.finishes_xy[agent_idx] = self._get_new_goal(agent_idx)

        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]
        obs = self._obs()
        terminated = [False] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
        return obs, rewards, terminated, truncated, infos

    @staticmethod
    def update_config(grid_config):
        obs_radius = grid_config.obs_radius
        grid = """
                ..............................................
                ..............................................
                .......##########.##########.##########.......
                ..............................................
                ..............................................
                ..............................................
                .......##########.##########.##########.......
                ..............................................
                ..............................................
                ..............................................
                .......##########.##########.##########.......
                ..............................................
                ..............................................
                ..............................................
                .......##########.##########.##########.......
                ..............................................
                ..............................................
                ..............................................
                .......##########.##########.##########.......
                ..............................................
                ..............................................
                ..............................................
                .......##########.##########.##########.......
                ..............................................
                ..............................................
                ..............................................
                .......##########.##########.##########.......
                ..............................................
                ..............................................
                ..............................................
                .......##########.##########.##########.......
                ..............................................
                ..............................................
                """
        starts = []
        for i in range(33):
            for j in [1, 2, 4, 5, 44, 43, 41, 40]:
                if i % 4 == 0:
                    continue
                starts.append((i + obs_radius, j + obs_radius))
        targets = []
        for i in range(1, 33, 2):
            for j in range(10):
                targets.append((i + obs_radius, j + 7 + obs_radius))
                targets.append((i + obs_radius, j + 18 + obs_radius))
                targets.append((i + obs_radius, j + 29 + obs_radius))
        possible_locations = {'starts': starts, 'targets': targets}
        raw_config = grid_config.dict()
        raw_config['map'] = grid
        return GridConfig(**raw_config), possible_locations


def create_warehouse_wfi_env(config):
    config.grid_config.map_name = 'wfi_warehouse'
    env = WarehouseWFI(grid_config=config.grid_config)
    env = ProvideGlobalObstacles(env)
    grid_config = env.grid_config
    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    if grid_config.on_target == 'restart':
        env = LifeLongAverageThroughputMetric(env)
    if config.with_animation:
        env = AnimationMonitor(env, AnimationConfig(directory='renders',
                                                    egocentric_idx=None,
                                                    show_lines=True
                                                    ))
    env = RuntimeMetricWrapper(env)
    return env


def main():
    env = WarehouseWFI(grid_config=GridConfig(num_agents=100, obs_radius=5, seed=1))
    env.reset()
    env.render()


if __name__ == '__main__':
    main()
