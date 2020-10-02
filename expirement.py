from abc import ABC

from simpleSim import Sim
import numpy as np
import gym
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.ppo as ppo
import ray
from ray.tune.logger import pretty_print
import argparse as ap
from ray.rllib.utils import try_import_tf

try:
    _, tf, version = try_import_tf(True)
    assert version == 2, "TF not version 2.xx"
except ImportError as e:
    raise e

assert tf.test.is_built_with_cuda(), "CUDA not available"


def _state_converter(state):
    return np.greater_equal(state, 0).astype(np.int8)


class Env(gym.Env, ABC):

    def __init__(self, config: dict):
        self.trade_off_ratio = config.pop('alpha', 1.)
        self.binary_state = config.pop('binary_state', True)
        self.max_pass_len = config.pop('max_pass_len', 1000)
        self.terminal_reward = config.pop('terminal_reward', -1000)
        self.sim = Sim(**config)
        self.action_space = gym.spaces.MultiDiscrete([2, ] * (self.sim.num_nodes ** 2))
        self.observation_space = gym.spaces.Box(low=-np.sum(self.sim.init_veh),
                                                high=np.infty,
                                                shape=(self.sim.num_nodes,)) \
            if not self.binary_state else gym.spaces.MultiBinary(self.sim.num_nodes)
        self.max_arr = np.max(self.sim.arr_rate)
        self.normalized_trip_time = self.sim.trip_time / np.max(self.sim.trip_time)

    def reset(self):
        queue_len = self.sim.reset()
        state = _state_converter(queue_len) if self.binary_state else queue_len

        return state

    def step(self, action: np.ndarray):
        virtual_arr = self._preprocess_action(action) * self.max_arr

        queue_len, terminate = self.sim.step(virtual_arr)
        reward = self._reward(queue_len, virtual_arr)
        state = _state_converter(queue_len) if self.binary_state else queue_len

        # print(f"state: {self.sim.queue_len}, action: {virtual_arr}, reward: {reward}")

        if np.all(np.greater(self.sim.queue_len, self.max_pass_len)):
            terminate = True
            reward = self.terminal_reward

        return state, \
            reward, \
            terminate, \
            {}

    def _preprocess_action(self, action):

        if np.any(np.isnan(action)):
            action = self.action_space.sample()
        action_mat = action.reshape((self.sim.num_nodes, self.sim.num_nodes))

        return action_mat

    def _reward(self, state, virt_arr):
        # arr = np.sum(self.sim.per_step_arrival)
        ser = np.sum(self.sim.per_step_throughput)
        service_rate = ser
        reb_cost = np.sum(virt_arr * self.normalized_trip_time)
        return service_rate - self.trade_off_ratio * reb_cost


if __name__ == '__main__':

    arg_parser = ap.ArgumentParser(prog="TwoSidedQueueNetwork")
    arg_parser.add_argument("--num_cpus", type=int, nargs=1, default=12)
    arg_parser.add_argument("--num_gpus", type=int, nargs=1, default=1)
    # num_stations = 2
    # routing_matrix = generate_random_routing(num_stations)

    env_config = {
        "rout_mat": np.array([[0, 1], [1, 0]]),
        "arr_rate": np.array([0.1, 0.1]),  # expected passenger arrival rate count/sec
        "trip_time": np.array([[100, 500], [500, 100]]),  # expected trip time
        "init_veh": np.array([100, 100]),
        "seed": 1010,
        "horizon": 1000,
        "time_per_step": 10.,
        "alpha": 1.,
        "binary_state": True,
    }
    total_run = 1000

    ray.init()
    all_config = ppo.DEFAULT_CONFIG.copy()
    all_config['num_workers'] = 12
    all_config['num_gpus'] = 1
    all_config['vf_clip_param'] = 1000
    all_config['env'] = Env
    all_config['env_config'] = env_config

    trainer = ppo.PPOTrainer(config=all_config)

    for _ in range(total_run):
        res = trainer.train()
        if (_ + 1) % 10 == 0:
            print(pretty_print(res))
        if (_ + 1) % 100 == 0:
            print(f"Model saved at {trainer.save()}")

