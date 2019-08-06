import randomizer
import gym

from randomizer.wrappers import RandomizedEnvWrapper

# env = RandomizedEnvWrapper(gym.make('FetchPushRandomizedEnv-v0'), seed=123)
env = RandomizedEnvWrapper(gym.make('HalfCheetahRandomizedEnv-v0'), seed=123)
env.randomize()
env.reset()
for _ in range(200):
    env.step(env.action_space.sample())
    env.render()