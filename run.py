from custom_envs import PPOExpRunner, binary_state_converter
from simulator import SimpleSim
import numpy as np
import warnings
import matplotlib.pyplot as plt
import ray.rllib.agents.ppo as ppo
import os


np.set_printoptions(precision=3, suppress=True)

runner = PPOExpRunner()
runner.load_cli_args()
print(runner.rout_mat)

runner.run()
# runner.run(dry_run=True,
#            checkpoint="/home/haochen/ray_results/PPO_PPOEnv_2020-10-05_17-58-10ntcdyccu/checkpoint_100/checkpoint-100")
trainer = runner.trainer
config = runner.env_config
config.pop('seed', None)
# seeds = [*range(100)]
seeds = [20]


def run_sim(sd):
    env = SimpleSim(**config, seed=sd)
    record = list()
    timestamp = []
    obs = env.reset()
    record.append(obs)
    timestamp.append(env.time)
    first_enter = None
    while True:
        act = trainer.compute_action(binary_state_converter(obs))
        # act = np.zeros(config['rout_mat'].shape)
        if (int(env.time) % 100) == 0:
            print("####################################################")
            print(f"Time: {env.time: .2f}\n"
                  f"State: {obs}\n"
                  f"Action: {act}\n"
                  f"On road: {env.vehicle_on_road}\n"
                  f"Total pass: \n{env.total_arr}\n"
                  f"Service rate:\n{env.per_step_throughput}")
            print("####################################################")

        obs, term = env.step(act)

        if np.all(np.greater(obs, 0)):
            if first_enter is None:
                first_enter = (env.total_arr, env.time)
            # breakpoint()
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
    plt.legend([str(x) for x in range(config['rout_mat'].shape[0])])
    plt.title('Imbalance vs time')
    plt.xlabel('Time')
    plt.ylabel('Imbalance')
    plt.grid()
    plt.savefig(f'plots/rl_trained/seed_{sd}.png')
    plt.close("all")
    # print("################################################")
    print(f'Exp: {sd}\nArr: {env.total_eff_arr_rate: .3f}\nThroughput: {env.total_throughput}')
    # print(f'Routing Matrix:\n {routing_matrix}')


for _ in seeds:
    run_sim(_)
