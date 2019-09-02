import numpy as np
import os
from gym import utils
from .controllers.hook_controller import get_hook_control
from .randomized_fetch_hook import RandomizedFetchHookEnv



class ResidualFetchHookEnv(RandomizedFetchHookEnv, utils.EzPickle):

    def __init__(self, **kwargs):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'hook:joint': [1.35, 0.35, 0.4, 1., 0., 0., 0.],
        }
        RandomizedFetchHookEnv.__init__(self, initial_qpos=initial_qpos, xml_file=None, **kwargs)
        self.reset()
        utils.EzPickle.__init__(self)

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = RandomizedFetchHookEnv.step(self, action)
        self._last_observation = observation

        return observation, reward, done, debug_info

    def compute_reward(self, *args, **kwargs):
        return RandomizedFetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = RandomizedFetchHookEnv.reset(self)
        self._last_observation = observation
        return observation