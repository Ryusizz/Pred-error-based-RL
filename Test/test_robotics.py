import gym
env = gym.make('FetchReach-v1')
env.reset()
for _ in range(500):
    env.render()
    env.step(env.action_space.sample())
