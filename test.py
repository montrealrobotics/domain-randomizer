import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper
import time

env = RandomizedEnvWrapper(gym.make('ResidualNoisyHookRandomizedEnv-v0'), seed=123)
env.randomize([1, 1, 1])
obs = env.reset()
for i in range(2000):
    obs, _, done, _ = env.step(np.zeros((4)))
    env.render()
    if i % 100 == 0:
        env.randomize([1, 1, 1])
        env.reset()

