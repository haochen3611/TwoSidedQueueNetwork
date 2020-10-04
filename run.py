from custom_envs import PPOExpRunner


runner = PPOExpRunner()

runner.load_cli_args()

print(runner.routing_mat)

runner.run()
