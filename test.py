import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper
import time

env = RandomizedEnvWrapper(gym.make('FetchPushRandomizedEnv-v0'), seed=123)
obs = env.reset()
action = np.array([0.1, 0.2, 0, 0.4])
print(action)
for i in range(2000):
    obs, _, _, _ = env.step(action)
    env.render()
    if i % 100 == 0:
        env.randomize(["default"])
        env.reset()

