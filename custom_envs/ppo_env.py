import json
from abc import ABC
import os
import copy
from simulator import SimpleSim, generate_random_routing
import numpy as np
import gym
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala
import ray
from ray.tune.logger import pretty_print
import argparse as ap
from ray.rllib.utils import try_import_tf
import warnings

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
    'PPOExpRunner',
    'binary_state_converter'
]

DEFAULT_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  os.path.pardir,
                                                  "config", "experiments"))

NP_ARRAY_FIELDS = {
    "rout_mat": 2,  # number means dimensions
    "arr_rate": 1,
    "trip_time": 2,
    "init_veh": 1
}

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


def binary_state_converter(state):
    return np.greater_equal(state, 0).astype(np.int8)


class PPOEnv(gym.Env, ABC):

    def __init__(self, config: dict, **kwargs):
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
        state = binary_state_converter(queue_len) if self.binary_state else queue_len

        return state

    def step(self, action: np.ndarray):
        virtual_arr = self._preprocess_action(action) * self.max_arr

        queue_len, terminate = self.sim.step(virtual_arr)
        reward = self._reward(queue_len, virtual_arr)
        state = binary_state_converter(queue_len) if self.binary_state else queue_len

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

    default_file_path = os.path.join(DEFAULT_CONFIG_DIR, file_name)
    if os.path.isfile(default_file_path):
        file_name = default_file_path
    try:
        with open(file_name, 'r') as f:
            config_ = json.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f'File \'{file_name}\' not found, fallback to default config')

    else:
        env_config_ = config_.pop('env_config', None)
        if env_config_ is not None:
            for def_env_key in PPOENV_DEFAULT_CONFIG:
                if def_env_key not in env_config_:
                    env_config_[def_env_key] = PPOENV_DEFAULT_CONFIG[def_env_key]

            for def_env_key in PPOENV_DEFAULT_CONFIG:
                if def_env_key in NP_ARRAY_FIELDS:
                    if isinstance(env_config_[def_env_key], (int, str)):
                        env_config_[def_env_key] = generate_random_routing(int(env_config_[def_env_key]),
                                                                           int(env_config_['seed']))
                    env_config_[def_env_key] = _validate_np_array_fields(env_config_[def_env_key],
                                                                         def_env_key)
        else:
            env_config_ = copy.deepcopy(PPOENV_DEFAULT_CONFIG)

        config_['env_config'] = env_config_

    return config_


def _validate_np_array_fields(array_, field_):
    """
    Convert any array to np.ndarray with format checking
    :param array_:
    :param field_:
    :return:
    """
    num_set_ = set('buifc')

    if field_ in NP_ARRAY_FIELDS:
        np_array_ = np.array(array_)
        dim_ = np_array_.ndim
        if dim_ != NP_ARRAY_FIELDS[field_]:
            raise TypeError(f"\'{field_}\' should have {NP_ARRAY_FIELDS[field_]} dimensions, got {dim_}")
        else:
            if dim_ == 2:
                shape_ = np_array_.shape
                assert shape_[0] == shape_[1], f"\'{field_}\' should be square matrix, got {shape_}"
            assert np_array_.dtype.kind in num_set_, f"\'{field_}\' should be numerical, got {np_array_.dtype}"
        return np_array_
    else:
        try:
            return PPOENV_DEFAULT_CONFIG[field_]
        except KeyError:
            return array_


class PPOExpRunner:

    def __init__(self, eager=False, checkpoint=None):

        self._cli_args = None
        self._all_config = None
        self._iterations = None
        self._trainer = None
        self._checkpoint_path = None

        if eager:
            self.load_cli_args()
            self.run(checkpoint=checkpoint)

    @property
    def result_folder(self):
        return os.path.realpath(os.path.join(self._checkpoint_path, ".."))

    @property
    def trainer(self) -> ppo.PPOTrainer:
        return self._trainer

    @property
    def policy(self):
        return self._trainer.get_policy()

    @property
    def rout_mat(self):
        try:
            return copy.deepcopy(self._all_config['env_config']['rout_mat'])
        except KeyError:
            return None

    @property
    def env_config(self):
        try:
            return copy.deepcopy(self._all_config['env_config'])
        except KeyError:
            return None

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

        self._all_config['num_workers'] = int(self._cli_args['num_cpus'])
        self._all_config['num_gpus'] = int(self._cli_args['num_gpus'])
        self._all_config['vf_clip_param'] = 1000
        self._all_config['env'] = PPOEnv
        self._all_config['log_level'] = "ERROR"

    def run(self, checkpoint=None, iters=None, dry_run=False):

        if self._cli_args is None:
            self.load_cli_args()

        if not self._cli_args['debug']:
            try:
                ray.init(address='auto')
            except ConnectionError:
                ray.init()
                print("Running in single node!")
            else:
                print("Running in cluster!")

        else:
            ray.init(local_mode=True)

        self._trainer = ppo.PPOTrainer(config=self._all_config)
        if checkpoint is not None:
            self._trainer.restore(checkpoint)

        if not dry_run:
            for _ in range(self._iterations if iters is None else int(iters)):
                res = self._trainer.train()
                if (_ + 1) % 10 == 0:
                    print(pretty_print(res))
                if (_ + 1) % 100 == 0:
                    self._checkpoint_path = self._trainer.save()
                    print(f"Model saved at {self._checkpoint_path}")


if __name__ == '__main__':

    runner = PPOExpRunner(eager=True)
