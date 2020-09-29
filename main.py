import numpy as np
from collections import deque
from itertools import product
import heapq
import tqdm


class Sim:

    def __init__(self, rout_mat, hor, arr_rate, trip_time, init_veh, seed):

        self.rout_mat = rout_mat
        self.hor = hor
        self.arr_rate = arr_rate
        self.trip_time = trip_time
        self.init_veh = init_veh
        self._rng = np.random.default_rng(seed)
        self.num_nodes = len(arr_rate)

        self._curr_time = None
        self.veh_sched = None
        self._queue_state = None  # represents num pass - num veh
        self._veh_queue = None  # not used
        self._pass_queue = None
        self.throughput = None
        self.pass_sched = None

    @property
    def queue_len(self):
        return np.copy(self._queue_state)

    @property
    def time(self):
        return self._curr_time

    def reset(self):
        self._curr_time = 0
        self.veh_sched = list()
        self.pass_sched = list()
        self._veh_queue = None
        self._pass_queue = dict()
        self._queue_state = -np.array(self.init_veh)
        self.throughput = np.zeros(self.num_nodes)

        for _ in range(self.num_nodes):
            self._pass_queue[_] = deque()

        return self.queue_len

    def travel_time(self, start, end):
        """
        scale of exponential distribution is expected interarrival time.
        :param start:
        :param end:
        :return:
        """
        return self._rng.exponential(scale=self.trip_time[start, end]) if start != end else 1

    def pass_arr(self, virtual_arr: np.ndarray = None, station: int = None):
        """
        :param station: (int) Station id. Choose which station to get the next arrival. Default to all stations.
        :param virtual_arr: (num_nodes, ) virtual passenger arrival
        :return:
        """
        virtual_arr = np.zeros(self.num_nodes) if virtual_arr is None else virtual_arr
        true_arr_od = self.arr_rate.reshape((-1, 1)) * self.rout_mat
        virt_arr_od = virtual_arr.reshape((-1, 1)) * self.rout_mat
        if station is None:
            for nonzero_arr in np.argwhere(true_arr_od):
                # for each nonzero arrival rate OD pair, sample the next passenger arrival time
                next_arr_time = self._rng.exponential(1 / true_arr_od[nonzero_arr[0], nonzero_arr[1]])
                heapq.heappush(self.pass_sched,
                               (next_arr_time + self._curr_time,
                                nonzero_arr[0],
                                nonzero_arr[1],
                                False))  # passenger tuple (time of arr, origin, destination, is_virtual)
            for nonzero_arr in np.argwhere(virt_arr_od):
                next_arr_time = self._rng.exponential(1 / virt_arr_od[nonzero_arr[0], nonzero_arr[1]])
                heapq.heappush(self.pass_sched,
                               (next_arr_time + self._curr_time,
                                nonzero_arr[0],
                                nonzero_arr[1],
                                True))

        else:
            chosen_true_arr = true_arr_od[station, :]
            chosen_virt_arr = virt_arr_od[station, :]
            for nonzero_arr in np.argwhere(chosen_true_arr):
                nonzero_arr = nonzero_arr.item()
                next_arr_time = self._rng.exponential(1 / chosen_true_arr[nonzero_arr])
                heapq.heappush(self.pass_sched,
                               (next_arr_time + self._curr_time,
                                station,
                                nonzero_arr,
                                False))
            for nonzero_arr in np.argwhere(chosen_virt_arr):
                nonzero_arr = nonzero_arr.item()
                next_arr_time = self._rng.exponential(1 / chosen_virt_arr[nonzero_arr])
                heapq.heappush(self.pass_sched,
                               (next_arr_time + self._curr_time,
                                station,
                                nonzero_arr,
                                True))

    def next_event(self, action):
        if len(self.pass_sched) == 0:
            self.pass_arr(action)

        if len(self.veh_sched) == 0 or self.pass_sched[0][0] < self.veh_sched[0][0]:
            new_pass = heapq.heappop(self.pass_sched)
            self._curr_time = new_pass[0]

            if self._queue_state[new_pass[1]] < 0:
                heapq.heappush(self.veh_sched,
                               (self._curr_time + self.travel_time(new_pass[1], new_pass[2]),
                                new_pass[1],
                                new_pass[2]))  # vehicle tuple (arrival time, origin, dest)
                self.throughput[new_pass[1]] += 1
                self._queue_state[new_pass[1]] += 1
            else:
                if not new_pass[3]:
                    self._pass_queue[new_pass[1]].append((new_pass[2], self._curr_time))
                    # passenger tuple in waiting queue (dest, arrival time)
                    self._queue_state[new_pass[1]] += 1

            self.pass_arr(action, station=new_pass[1])
        else:
            cur_arr_veh = heapq.heappop(self.veh_sched)
            self._curr_time = cur_arr_veh[0]
            if self._queue_state[cur_arr_veh[2]] > 0:
                self.throughput[cur_arr_veh[2]] += 1
                # Vehicle go to the next passenger's dest immediately
                # Can add non-trivial match time here
                # Can observe passenger waiting time here
                next_pass = self._pass_queue[cur_arr_veh[2]].popleft()
                heapq.heappush(self.veh_sched,
                               (self._curr_time + self.travel_time(cur_arr_veh[2], next_pass[0]),
                                cur_arr_veh[2],
                                next_pass[0]))
            self._queue_state[cur_arr_veh[2]] -= 1

        return self.queue_len

    def step(self, action, t=None):
        """
        Time-based execution. Time is 1 sec at default. To accommodate RL controller.
        :param t: time span for each step
        :param action: virtual passenger arrival, same dim as arrival rate
        :return:
        """
        if t is None:
            t = 10.0
        else:
            t = float(t)
        now = self._curr_time
        while self._curr_time < t + now:
            self.next_event(action)

        return self.queue_len


def simple_policy(state: np.ndarray, pass_arr_rate: np.ndarray, routing_mat: np.ndarray):
    """

    :param routing_mat: (num_nodes, num_nodes)
    :param state: (num_nodes, ) SS node states
    :param pass_arr_rate: (num_nodes, )
    :return: np.ndarray (num_nodes, )
    """
    action = np.zeros(state.shape)

    # veh_arr_rate = np.dot(pass_arr_rate, routing_mat)
    below_zeros = np.argwhere(np.less(state, 0))
    above_zeros = np.any(np.greater(state, 0))
    # min_state = np.argmin(state)
    # max_state = np.argmax(state)
    for subzero in below_zeros:
        if above_zeros:
            action[subzero] = 2 * pass_arr_rate[subzero]

    return action


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    routing_matrix = np.array([[0, 1], [1, 0]])
    horizon = 10000
    arrival_rates = np.array([0.1, 0.1])  # expected passenger arrival rate count/sec
    trip_mean_time = np.array([[1, 10], [5, 1]])  # expected trip time
    initial_vehicle = np.array([100, 100])
    time_per_step = 50

    for _ in tqdm.trange(100):
        # _ = 66
        env = Sim(rout_mat=routing_matrix,
                  arr_rate=arrival_rates,
                  hor=horizon,
                  trip_time=trip_mean_time,
                  init_veh=initial_vehicle,
                  seed=_)
        record = list()
        timestamp = []
        obs = env.reset()
        record.append(obs)
        timestamp.append(env.time)
        for e in range(0, horizon, time_per_step):
            # act = simple_policy(obs, arrival_rates, routing_matrix)
            act = np.zeros(obs.shape)
            # print(f"Time: {env.time} State: {obs} Action: {act}")
            # obs = env.next_event(act)
            obs = env.step(act, t=time_per_step)
            record.append(obs)
            timestamp.append(env.time)

        record = np.vstack(record)
        timestamp = np.array(timestamp)

        plt.plot(timestamp, record)
        plt.legend(['1', '2'])
        plt.title('Imbalance vs time')
        plt.xlabel('Time')
        plt.ylabel('Imbalance')
        plt.savefig(f'plots/seed_{_}.png')
        plt.close("all")

    # env = Sim(rout_mat=routing_matrix,
    #           arr_rate=arrival_rates,
    #           hor=horizon,
    #           trip_time=trip_mean_time,
    #           init_veh=initial_vehicle,
    #           seed=65)
    # env.reset()
    # total_pass = [0, 0]
    # t_p = 0
    # for p in env.pass_sched:
    #     t_p += len(p)
    #     for _ in p:
    #         total_pass[_[0]] += 1
    # print(total_pass)
    # print(t_p)
