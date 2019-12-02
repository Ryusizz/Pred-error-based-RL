import gym
import numpy as np
import mujoco_py

from wrappers import ExternalForceWrapper

mujoco_py.ignore_mujoco_warnings().__enter__()
# env = gym.make('UR5PickAndPlaceDense-v1')
env = gym.make("UR5ReachDense-v1")
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
# env = ExternalForceWrapper(env, 0.75)
ob = env.reset()
print(ob.shape)
eprew = 0
# for _ in range(500):
t = 0
grip = -1
while True:
    # print(ob)
    env.render()
    ac = env.action_space.sample()
    ac = np.zeros(4)
    # ac[:3] = 3*(ob[-6:-3] -ob[:3])
    ac[:3] = 3*(ob[-3:] - ob[:3])
    print(ac)
    # ac[1] = -0.5
    ac[3] = grip
    ob, rew, done, info = env.step(ac)
    eprew += rew
    t += 1
    if t > 25:
        grip = 1
    if done:
        ob = env.reset()
        print(eprew)
        eprew = 0
        grip = -1
        t = 0
