import warnings
try:
    from tensorflow.keras.initializers import RandomNormal, Constant
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import TensorBoard
except ImportError:
    warnings.warn('Tensorflow module cannot be imported', category=ImportWarning)
    RandomNormal = None
    Constant = None
    Sequential = None
    Dense = None
    Adam = None
    TensorBoard = None

import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque
import random
import glob
import time
import pickle
import gym
# from ray.rllib.agents.dqn.simple_q import SimpleQTrainer


class Recorder:

    def __init__(self, name: str, state_space, action_space):
        self._check_sate_action(state_space, action_space)
        sa_space = state_space + (action_space, )
        self.name = name
        self.state_space = state_space
        self.action_space = action_space
        self._q_table = np.zeros(sa_space, dtype=float)
        self.reward_hist = []
        self.trans_hist = []
        self.sa_counter = np.ones(sa_space, dtype=np.int)
        self.variables = {}

    @staticmethod
    def _check_sate_action(state, action):
        assert isinstance(state, (tuple, list, np.ndarray)), f'State is {type(state)}, not compatible'
        assert isinstance(action, int), 'Action must be single dimension'

    @property
    def q_table(self):
        return self._q_table

    def initialize_q_table(self, values):
        assert self._q_table.sum() == 0, 'Cannot call this if q table is modified.'
        self._q_table += values

    def count_sa(self, sa_pair):
        self.sa_counter[sa_pair] += 1

    def store_history(self, trans):
        self.trans_hist.append(trans)

    def store_reward(self, reward):
        self.reward_hist.append(reward)

    def store_variables_step(self, **kwargs):
        while len(kwargs) > 0:
            pair = kwargs.popitem()
            try:
                self.variables[pair[0]].append(pair[1])
            except KeyError:
                self.variables[pair[0]] = [pair[1], ]

    def store_variables_end(self, **kwargs):
        while len(kwargs) > 0:
            pair = kwargs.popitem()
            if pair[0] in self.variables.keys():
                raise Exception('Key exists')
            self.variables[pair[0]] = [pair[1], ]

    def save_data(self, directory):
        try:
            assert os.path.isdir(directory)
        except AssertionError:
            os.makedirs(directory)
        with open(os.path.join(directory, self.name+'.pkl'), 'wb') as f:
            pickle.dump({'r_hist': self.reward_hist,
                         'q_table': self._q_table,
                         'trans_counter': self.sa_counter,
                         'trans_hist': self.trans_hist,
                         'addition_var': self.variables},
                        f)


class BaseDQN:
    """Base class should not be called directly."""

    GAMMA = 0.95
    LEARNING_RATE = 0.001

    MEMORY_SIZE = 1000000
    BATCH_SIZE = 20
    REPLACE_INTERVAL = 50
    SAVE_INTERVAL = 500

    EXPLORATION_MAX = 1
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.995

    def __init__(self, observation_space, action_space):
        self.exploration_rate = self.EXPLORATION_MAX
        self.obs_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.w_initializer = RandomNormal(mean=0.0, stddev=0.05)
        self.b_initializer = Constant(value=0.1)
        self._learn_counter = 0
        self.loss_history = []

        if os.path.exists('./checkpoints'):
            try:
                files = glob.glob('./checkpoints/*.h5')
                for f in files:
                    os.remove(f)
            except FileNotFoundError:
                pass
        else:
            os.makedirs('./checkpoints')

        self._build_evalue_net()
        self._build_target_net()

    def _build_evalue_net(self):
        self.evalue_net = None

    def _build_target_net(self):
        self.target_net = None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _sync_weights(self):
        if self.target_net is not None:
            weights = self.evalue_net.get_weights()
            self.target_net.set_weights(weights)

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.evalue_net.predict(state)
        return np.argmax(q_values[0])

    def learn(self):
        if self._learn_counter % self.REPLACE_INTERVAL == 0:
            self._sync_weights()
        if self._learn_counter % self.SAVE_INTERVAL == 0:
            self.save_model()

        batch = self.experience_replay()
        if batch is None:
            return
        self.q_update(samples=batch)
        self.exploration_rate *= self.EXPLORATION_DECAY
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)
        self._learn_counter += 1

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        return random.sample(self.memory, self.BATCH_SIZE)

    def q_update(self, samples):
        x = []
        y = []
        for state, action, reward, state_next, terminal in samples:
            q_update = reward
            if not terminal:
                q_update = (reward + self.GAMMA * np.amax(self.target_net.predict(state_next)[0])) \
                    if self.target_net is not None else \
                    (reward + self.GAMMA * np.amax(self.evalue_net.predict(state_next)[0]))
            q_values = self.evalue_net.predict(state)
            q_values[0][action] = q_update
            y.append(q_values)
            x.append(state)
        x = np.stack(x, axis=0).squeeze()
        y = np.stack(y, axis=0).squeeze()
        history = self.evalue_net.fit(x, y, verbose=0, epochs=1)
        self.loss_history.append(history.history['loss'][0])

    def plot_loss(self):
        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.ylabel('Cost')
        plt.xlabel('Learning Setps')
        plt.savefig('loss.png')

    def save_model(self):
        self.evalue_net.save(f'./checkpoints/eval-{self._learn_counter}.h5')
        if self.target_net is not None:
            self.target_net.save(f'./checkpoints/tar-{self._learn_counter}.h5')


class SingleDQN(BaseDQN):

    def __init__(self, *args):
        super(SingleDQN, self).__init__(*args)
        self.tb = TensorBoard(log_dir='logs/{}'.format(time.time()))

    def _build_evalue_net(self):
        self.evalue_net = Sequential()
        self.evalue_net.add(Dense(24,
                                  input_shape=(self.obs_space,),
                                  kernel_initializer=self.w_initializer,
                                  bias_initializer=self.b_initializer,
                                  activation="relu"))
        self.evalue_net.add(Dense(24,
                                  activation="relu",
                                  kernel_initializer=self.w_initializer,
                                  bias_initializer=self.b_initializer,
                                  ))
        self.evalue_net.add(Dense(self.action_space, activation="linear"))
        self.evalue_net.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))

    def q_update(self, samples):
        """Seems better if we fit the net after each sample"""
        for state, action, reward, state_next, terminal in samples:
            q_update = reward
            if not terminal:
                q_update = (reward + self.GAMMA * np.amax(self.evalue_net.predict(state_next)[0]))
            q_values = self.evalue_net.predict(state)
            q_values[0][action] = q_update
            history = self.evalue_net.fit(state, q_values, verbose=0, epochs=1, callbacks=[self.tb])
            self.loss_history.append(history.history['loss'][0])


class CrossEntropy(BaseDQN):

    def __init__(self, *args):
        super(CrossEntropy, self).__init__(*args)

    def _build_evalue_net(self):
        self.evalue_net = Sequential()
        self.evalue_net.add(Dense(128,
                                  input_shape=(self.obs_space,),
                                  kernel_initializer=self.w_initializer,
                                  bias_initializer=self.b_initializer,
                                  activation="relu"))
        self.evalue_net.add(Dense(self.action_space, activation="linear"))
        self.evalue_net.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))


class QTable:
    GAMMA = 0.99
    LEARNING_RATE = 0.1

    EXPLORATION_MAX = 1
    EXPLORATION_MIN = 0.01
    EXPLORATION_DECAY = 0.9995

    RANDOM_STEPS = 1000

    SAVE_DIR = './tmp/cartpole_q_table'

    def __init__(self, state_space: tuple, action_space: int, name='q_table'):
        self.name = name
        self.action_space = action_space
        self.state_space = state_space
        self.recoder = Recorder(self.name, self.state_space, self.action_space)
        self.gamma = self.GAMMA
        self.alpha = self.LEARNING_RATE
        self.explorer = None
        self.exploration_rate = self.EXPLORATION_MAX
        self._learn_counter = 0
        self._total_reward = 0

    def add_explorer(self, explorer: str):
        if explorer in ('BMLE', 'bmle'):
            self.explorer = BMLE(self.recoder, random_steps=self.RANDOM_STEPS)
        elif explorer in ('UCBInfinite', 'UCBInf'):
            self.explorer = UCBInfinite(self.recoder, gamma=self.gamma)
        else:
            warnings.warn('Using E-Greedy.')

    @property
    def q_table(self):
        return self.recoder.q_table

    @property
    def memory(self):
        return self.recoder.trans_hist

    def act(self, state: tuple):
        if self.explorer is None:
            if np.random.rand() < self.exploration_rate:
                return random.randrange(self.action_space)
            return np.argmax(self.q_table[state])
        elif not hasattr(self.explorer, 'act'):
            return np.argmax(self.q_table[state])
        else:
            return self.explorer.act(state, self._learn_counter)

    def remember(self, state: tuple, action: int, reward, state_next: tuple, terminal):
        self.recoder.count_sa(state+(action, ))
        self.recoder.store_history((state, action, reward, state_next, terminal))
        self._total_reward += reward
        if terminal:
            self.recoder.store_reward(self._total_reward)

    def learn(self):
        assert len(self.memory) > 0
        state, action, reward, state_next, _ = self.memory[-1]

        if self.explorer is None or not hasattr(self.explorer, 'update_q'):
            best_q = np.amax(self.q_table[state_next])
            self.q_table[state+(action,)] += self.alpha * (reward + self.gamma * best_q - self.q_table[state+(action,)])
        else:
            self.explorer.update_q(state, action, reward, state_next)

        self.exploration_rate *= self.EXPLORATION_DECAY
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)
        self._learn_counter += 1

    def save_model(self):
        self.recoder.save_data(self.SAVE_DIR)


class UCBInfinite:

    def __init__(self, recorder: Recorder, **kwargs):
        self.recorder = recorder
        self.tmp_q_table = np.zeros(self.recorder.q_table.shape, dtype=float)
        try:
            self.gamma = kwargs.pop('gamma')
            self.error = kwargs.pop('error')
            self.delta = kwargs.pop('delta')
        except KeyError:
            warnings.warn('Initialize with default parameters')
            self.gamma = 0.9
            self.error = 0.01
            self.delta = 0.01

        self.n_states = 1
        for s in self.recorder.state_space:
            self.n_states *= s
        self.n_actions = self.recorder.action_space
        self._initialize()

    def _initialize(self):
        self.R = np.ceil(np.log(3 / (self.error * (1 - self.gamma))) / (1 - self.gamma))
        self.c_2 = 4 * np.sqrt(2)
        self.M = np.log(1 / ((1 - self.gamma) * self.error))
        self.e_1 = self.error / (24 * self.R * self.M * np.log(1 / (1 - self.gamma)))
        self.H = np.log(1 / ((1 - self.gamma) * self.e_1)) / np.log(1 / self.gamma)
        self.tmp_q_table += 1 / (1 - self.gamma)
        self.recorder.initialize_q_table(1/(1-self.gamma))

    def _alpha(self, k):
        return (self.H + 1)/(self.H + k)

    def _iota(self, k):
        return np.log(self.n_states*self.n_actions*(k+1)*(k+2)/self.delta)

    def update_q(self, state, action, reward, state_next):
        k = self.recorder.sa_counter[state+(action,)]
        b_k = self.c_2/(1-self.gamma) * np.sqrt(self.H*self._iota(k)/k)
        best_q = np.amax(self.recorder.q_table[state_next])
        self.tmp_q_table[state+(action,)] += \
            self._alpha(k)*(reward + b_k + self.gamma*best_q - self.tmp_q_table[state+(action,)])
        self.recorder.q_table[state+(action,)] = min(self.tmp_q_table[state+(action,)],
                                                     self.recorder.q_table[state+(action,)])


class BMLE:

    def __init__(self, recorder: Recorder, random_steps=1000):
        self.o = 1.5
        self.offset = 0.5
        self.scale = 2.1
        self.recorder = recorder
        assert isinstance(random_steps, int)
        self.random_steps = random_steps

    def alpha(self, t):
        return np.power(np.log(np.log(t)), self.o)

    def normalize(self, vec):
        assert isinstance(vec, np.ndarray)
        return vec / (np.max(np.abs(vec))) / self.scale + self.offset \
            if np.max(np.abs(vec)) != 0 else np.zeros(vec.shape)

    def compute_bmle_index(self, state, t, normalized_q):

        def bmle_index(v, n, a):
            assert isinstance(a, float)
            return (v * n + a) * np.log(v * n + a) + n * np.log(n) - (n + a) * np.log(n + a) - n * v * np.log(n * v)

        alpha_t = self.alpha(t)
        n_visits = self.recorder.sa_counter[state]
        index = bmle_index(normalized_q, n_visits, alpha_t)
        assert isinstance(index, np.ndarray)
        action = int(np.argmax(index[~np.isnan(index)]))

        return action, index

    def act(self, state, step):
        norm_q = self.normalize(self.recorder.q_table[state])

        if step < self.random_steps or np.linalg.norm(norm_q) == 0:
            action = random.randrange(self.recorder.action_space)
            index = [-float('Inf'), -float('Inf')]
        else:
            action, index = self.compute_bmle_index(state, step, norm_q)

        self.recorder.store_variables_step(bmle_idx=index)

        return action


if __name__ == '__main__':
    r = Recorder('s', (3,), 3)
    # r.q_table[0, 0] = 1
    # print(r.q_table[(2,)].shape)
    b = BMLE(r)
    print(hasattr(b, 'update_q'))
