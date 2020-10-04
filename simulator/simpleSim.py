import numpy as np
from collections import deque
from itertools import product
import heapq
import tqdm


__all__ = [
    "SimpleSim",
    "generate_random_routing",
    "simple_policy",
]


class SimpleSim:

    def __init__(self,
                 rout_mat,
                 arr_rate,
                 trip_time,
                 init_veh,
                 horizon,
                 time_per_step,
                 seed):

        self.rout_mat = rout_mat
        self.arr_rate = arr_rate
        self.num_nodes = len(arr_rate)

        self.horizon = horizon
        self.time_per_step = time_per_step
        self.trip_time = trip_time
        self.init_veh = init_veh
        self._rng = np.random.default_rng(seed)

        self._validate_input_arrays()

        self._curr_time = None
        self.veh_sched = None
        self._queue_state = None  # represents num pass - num veh
        self._veh_queue = None  # not used
        self._pass_queue = None
        self._serviced_pass = None
        self.pass_sched = None
        self._total_pass = None
        self._serviced_per_step = None
        self._arrival_per_step = None
        self._step_true_span = None

    def _validate_input_arrays(self):
        """Use after variables are set. Check dimension coherence."""
        assert isinstance(self.rout_mat, np.ndarray)
        assert self.rout_mat.shape == (self.num_nodes, self.num_nodes)

        assert isinstance(self.arr_rate, np.ndarray)
        assert self.arr_rate.shape == (self.num_nodes, )

        assert isinstance(self.trip_time, np.ndarray)
        assert self.trip_time.shape == (self.num_nodes, self.num_nodes)

        assert isinstance(self.init_veh, np.ndarray)
        assert self.init_veh.shape == (self.num_nodes, )

    @property
    def queue_len(self):
        return self._queue_state

    @property
    def vehicle_on_road(self):
        return len(self.veh_sched)

    @property
    def total_arr(self):
        return self._total_pass

    @property
    def total_eff_arr_rate(self):
        return np.sum(self._total_pass) / self._curr_time

    @property
    def per_step_arrival(self):
        return self._arrival_per_step / self._step_true_span

    @property
    def total_throughput(self):
        return np.sum(self._serviced_pass) / self._curr_time

    @property
    def per_step_throughput(self):
        return self._serviced_per_step / self._step_true_span

    @property
    def time(self):
        return self._curr_time

    def reset(self):
        self._curr_time = 0
        self.veh_sched = list()
        self.pass_sched = list()
        self._veh_queue = None
        self._pass_queue = dict()
        self._total_pass = np.zeros((self.num_nodes, self.num_nodes))
        self._queue_state = -np.array(self.init_veh)
        self._serviced_pass = np.zeros((self.num_nodes, self.num_nodes))
        self._serviced_per_step = np.zeros((self.num_nodes, self.num_nodes))
        self._arrival_per_step = np.zeros((self.num_nodes, self.num_nodes))
        self._step_true_span = self.time_per_step

        for _ in range(self.num_nodes):
            self._pass_queue[_] = deque()

        return self._queue_state

    def travel_time(self, start, end):
        """
        scale of exponential distribution is expected interarrival time.
        :param start:
        :param end:
        :return:
        """
        return self._rng.exponential(scale=self.trip_time[start, end])

    def _gen_pass(self, true_arr: np.ndarray, virt_arr: np.ndarray, od_pair: tuple):
        """

        :param virt_arr: np.ndarray (num_nodes, num_nodes)
        :param true_arr: np.ndarray (num_nodes, num_nodes)
        :param od_pair: (int, int)
        :return:
        """
        total_arr = true_arr + virt_arr
        origin, dest = od_pair
        # for each nonzero arrival rate OD pair, sample the next passenger arrival time
        if total_arr[origin, dest] > 0:
            next_arr_time = self._rng.exponential(1 / total_arr[origin, dest])
            virt_prob = virt_arr[origin, dest] / total_arr[origin, dest]
            is_virt = self._rng.uniform(0, 1) <= virt_prob
            heapq.heappush(self.pass_sched,
                           (next_arr_time + self._curr_time,
                            origin,
                            dest,
                            is_virt))
            # passenger tuple (time of arr, origin, destination, is_virtual)
            if not is_virt:
                self._total_pass[origin, dest] += 1

    def pass_arr(self, virtual_arr: np.ndarray = None, od_pair: tuple = None):
        """
        Generate one passenger at a time
        :param od_pair: (int, int) choose which OD pair get next event. None for all.
        :param virtual_arr: (num_nodes, num_nodes) virtual passenger arrival
        :return:
        """
        virt_arr_od = np.zeros(self.num_nodes, self.num_nodes) if virtual_arr is None else virtual_arr
        true_arr_od = self.arr_rate.reshape((-1, 1)) * self.rout_mat
        if od_pair is None:
            for od in product(range(self.num_nodes), repeat=2):
                self._gen_pass(true_arr_od, virt_arr_od, od)
        else:
            self._gen_pass(true_arr_od, virt_arr_od, od_pair)

    def next_event(self, action: np.ndarray):
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
                if not new_pass[3]:
                    self._serviced_pass[new_pass[1], new_pass[2]] += 1
                self._queue_state[new_pass[1]] += 1
            else:
                if not new_pass[3]:
                    self._pass_queue[new_pass[1]].append((new_pass[2], self._curr_time))
                    # passenger tuple in waiting queue (dest, arrival time)
                    self._queue_state[new_pass[1]] += 1

            self.pass_arr(action, od_pair=(new_pass[1], new_pass[2]))
        else:
            cur_arr_veh = heapq.heappop(self.veh_sched)
            self._curr_time = cur_arr_veh[0]
            if self._queue_state[cur_arr_veh[2]] > 0:
                # Vehicle go to the next passenger's dest immediately
                # Can add non-trivial match time here
                # Can observe passenger waiting time here
                next_pass = self._pass_queue[cur_arr_veh[2]].popleft()
                heapq.heappush(self.veh_sched,
                               (self._curr_time + self.travel_time(cur_arr_veh[2], next_pass[0]),
                                cur_arr_veh[2],
                                next_pass[0]))
                self._serviced_pass[cur_arr_veh[2], next_pass[0]] += 1

            self._queue_state[cur_arr_veh[2]] -= 1

        return self._queue_state

    def step(self, action, t=None):
        """
        Time-based execution. Soft requirement for step interval c To accommodate RL controller.
        :param t: time span for each step
        :param action: virtual passenger arrival, same dim as arrival rate
        :return: None if time goes over horizon
        """

        t = self.time_per_step if t is None else float(t)
        now = self._curr_time
        pre_arr = self._total_pass
        pre_ser = self._serviced_pass
        while self._curr_time < t + now:
            self.next_event(action)

        # they must be updated together in order to keep them lined up
        self._step_true_span = self._curr_time - now
        self._arrival_per_step = self._total_pass - pre_arr
        self._serviced_per_step = self._serviced_pass - pre_ser

        return self._queue_state, self._curr_time >= self.horizon


def simple_policy(state: np.ndarray, pass_arr_rate: np.ndarray, routing_mat: np.ndarray):
    """

    :param routing_mat: (num_nodes, num_nodes)
    :param state: (num_nodes, ) SS node states
    :param pass_arr_rate: (num_nodes, )
    :return: np.ndarray (num_nodes, )
    """
    action = np.zeros(state.shape)
    below_zeros = np.argwhere(np.less(state, 0))
    above_zeros = np.any(np.greater(state, 0))
    for subzero in below_zeros:
        if above_zeros:
            action[subzero] = 4 * pass_arr_rate[subzero]
    return action


def generate_random_routing(num_nodes, seed=1):
    rng = np.random.default_rng(seed)
    rt_mat = rng.uniform(0, 1, (num_nodes, num_nodes))
    return rt_mat / rt_mat.sum(axis=1).reshape((-1, 1))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import multiprocessing as mp
    import warnings
    np.set_printoptions(precision=3, suppress=True)

    num_stations = 2
    # routing_matrix = generate_random_routing(num_stations)
    routing_matrix = np.array([[0, 1], [1, 0]])
    horizon_ = 50000
    arrival_rates = np.array([0.1, 0.1])  # expected passenger arrival rate count/sec
    trip_mean_time = np.array([[100, 500], [500, 100]])  # expected trip time
    initial_vehicle = np.array([100, 100])
    time_per_step_ = 10
    # seeds = [*range(100)]
    seeds = [0]

    def run_sim(sd):
        env = SimpleSim(rout_mat=routing_matrix,
                        arr_rate=arrival_rates,
                        horizon=horizon_,
                        trip_time=trip_mean_time,
                        init_veh=initial_vehicle,
                        time_per_step=time_per_step_,
                        seed=sd)
        record = list()
        timestamp = []
        obs = env.reset()
        record.append(obs)
        timestamp.append(env.time)
        first_enter = None
        while True:
            act = simple_policy(obs, arrival_rates, routing_matrix)
            # act = np.zeros(obs.shape, ob.shape)
            print(f"Time: {env.time: .2f} State: {obs} Action: {act} On road: {env.vehicle_on_road} "
                  f"Total pass: {env.total_arr}")
            # obs = env.next_event(act)
            obs, term = env.step(act, t=time_per_step_)

            if np.all(np.greater(obs, 0)):
                if first_enter is None:
                    first_enter = (env.total_arr, env.time)
                breakpoint()
                if np.all(np.greater(obs, 1000)):
                    chaos_arr_rate = (env.total_arr - first_enter[0]) / (env.time - first_enter[1])
                    warnings.warn(f"Aborted as both queues are positive: {obs}")
                    print(f"Arrival rate since chaos: {chaos_arr_rate}")
                    break
            record.append(obs)
            timestamp.append(env.time)
            if term:
                break

        record = np.vstack(record)
        timestamp = np.array(timestamp)

        plt.plot(timestamp, record)
        plt.legend(['0', '1'])
        plt.title('Imbalance vs time')
        plt.xlabel('Time')
        plt.ylabel('Imbalance')
        plt.grid()
        plt.savefig(f'plots/seed_{sd}.png')
        plt.close("all")
        print(f'Exp: {sd} Arr: {env.total_eff_arr_rate: .3f} Throughput: {env.total_throughput}')
        print(f'Routing Matrix:\n {routing_matrix}')


    # with mp.Pool(12) as pool:
    #     pool.map(run_sim, seeds)

    for _ in seeds:
        run_sim(_)

    # run_sim(*seeds)
