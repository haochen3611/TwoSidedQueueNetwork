import json
from abc import ABC
import copy
from simulator import SimpleSim
import numpy as np
import gym
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

if not tf.test.is_built_with_cuda():
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')


__all__ = [
    'PPOEnv',
    'PPOENV_DEFAULT_CONFIG',
    'PPOExpRunner'
]

NP_ARRAY_FIELDS = [
    "rout_mat",
    "arr_rate",
    "trip_time",
    "init_veh"
]

PPOENV_DEFAULT_CONFIG = {
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


def _state_converter(state):
    return np.greater_equal(state, 0).astype(np.int8)


class PPOEnv(gym.Env, ABC):

    def __init__(self, config: dict):
        self.trade_off_ratio = config.pop('alpha', 1.)
        self.binary_state = config.pop('binary_state', True)
        self.max_pass_len = config.pop('max_pass_len', 1000)
        self.terminal_reward = config.pop('terminal_reward', -1000)
        self.sim = SimpleSim(**config)
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


def config_file_parser(file_name):
    config_ = {}
    try:
        with open(file_name, 'r') as f:
            config_ = json.load(f)

    except FileNotFoundError:
        env_config_ = copy.deepcopy(PPOENV_DEFAULT_CONFIG)

    else:
        env_config_ = config_.pop('env_config', None)
        if env_config_ is not None:
            for def_env_key in PPOENV_DEFAULT_CONFIG:
                if def_env_key not in env_config_:
                    env_config_[def_env_key] = PPOENV_DEFAULT_CONFIG[def_env_key]
                else:
                    if def_env_key in NP_ARRAY_FIELDS:
                        env_config_[def_env_key] = np.array(env_config_[def_env_key])
        else:
            env_config_ = copy.deepcopy(PPOENV_DEFAULT_CONFIG)

    config_['env_config'] = env_config_

    return config_


class PPOExpRunner:

    def __init__(self, eager=False, checkpoint=None):

        self._cli_args = None
        self._all_config = None
        self._iterations = None
        self._trainer = None

        if eager:
            self.load_cli_args()
            self.run(checkpoint=checkpoint)

    @property
    def trainer(self):
        return self._trainer

    @property
    def policy(self):
        return self._trainer.get_policy()

    def load_cli_args(self):

        arg_parser = ap.ArgumentParser(prog="TwoSidedQueueNetwork")
        arg_parser.add_argument("--num_cpus", type=int, nargs='?', default=12)
        arg_parser.add_argument("--num_gpus", type=int, nargs='?', default=1)
        arg_parser.add_argument("--iter", type=int, nargs='?', default=1000)
        arg_parser.add_argument("--config", type=str, nargs='?', default="")
        arg_parser.add_argument("--debug", action="store_true", default=False)

        self._cli_args = vars(arg_parser.parse_args())

        self._iterations = self._cli_args['iter']
        file_config = config_file_parser(self._cli_args['config'])

        self._all_config = ppo.DEFAULT_CONFIG.copy()
        self._all_config.update(file_config)

        self._all_config['num_workers'] = int(self._cli_args ['num_cpus'])
        self._all_config['num_gpus'] = int(self._cli_args ['num_gpus'])
        self._all_config['vf_clip_param'] = 1000
        self._all_config['env'] = PPOEnv
        self._all_config['log_level'] = "ERROR"

    def run(self, checkpoint=None):

        if self._cli_args is None:
            self.load_cli_args()

        if not self._cli_args['debug']:
            try:
                print("Running in cluster!")
                ray.init(address='auto')
            except ConnectionError:
                print("Running in single node!")
                ray.init()
        else:
            ray.init(local_mode=True)

        self._trainer = ppo.PPOTrainer(config=self._all_config)
        if checkpoint is not None:
            self._trainer.restore(checkpoint)

        for _ in range(self._iterations):
            res = self._trainer.train()
            if (_ + 1) % 10 == 0:
                print(pretty_print(res))
            if (_ + 1) % 100 == 0:
                print(f"Model saved at {self._trainer.save()}")


if __name__ == '__main__':

    runner = PPOExpRunner(eager=True)
