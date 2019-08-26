import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper
import time

env = RandomizedEnvWrapper(gym.make('ResidualMPCPushRandomizedEnv-v0'), seed=123)
# env = RandomizedEnvWrapper(gym.make('HalfCheetahRandomizedEnv-v0'), seed=123)
env.randomize(["default", 1])
obs = env.reset()
# friction_ranges = [friction for friction in np.geomspace(0.05, 1, 10)]
for _ in range(200):
    obs, _, _, _ = env.step(env.action_space.sample())
    time.sleep(0.05)
    env.render()
    # env.randomize(["default", -1])

