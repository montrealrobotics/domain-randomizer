import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper

env = RandomizedEnvWrapper(gym.make('ResidualPushDefaultEnv-v0'), seed=123)
# env = RandomizedEnvWrapper(gym.make('HalfCheetahRandomizedEnv-v0'), seed=123)
env.randomize()
obs = env.reset()

for _ in range(200):
    obs, _, _, _ = env.step(env.action_space.sample())
    env.render()
