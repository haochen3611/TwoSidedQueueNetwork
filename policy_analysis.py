import ray.rllib.agents.ppo as ppo
import ray
import numpy as np
from custom_envs import PPOEnv, PPOENV_DEFAULT_CONFIG


total_run = 1000

ray.init()
all_config = ppo.DEFAULT_CONFIG.copy()
all_config['num_workers'] = 12
all_config['num_gpus'] = 1
all_config['vf_clip_param'] = 1000
all_config['env'] = PPOEnv
all_config['env_config'] = PPOENV_DEFAULT_CONFIG

trainer = ppo.PPOTrainer(config=all_config)
trainer.restore('./checkpts/checkpoint_1000')

trainer.compute_actions()
policy = trainer.get_policy()

