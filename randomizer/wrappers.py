import os

import gym
import json
import numpy as np

import gym.spaces as spaces

from randomizer.dimension import Dimension


class RandomizedEnvWrapper(gym.Wrapper):
    """Creates a randomization-enabled enviornment, which can change
    physics / simulation parameters without relaunching everything
    """

    def __init__(self, env, seed):
        super(RandomizedEnvWrapper, self).__init__(env)
        self.config_file = os.path.join(os.path.dirname(__file__), self.unwrapped.config_file)

        self._load_randomization_dimensions(seed)
        self.unwrapped.update_randomized_params()
        self.randomized_default = ['random'] * len(self.unwrapped.dimensions)

    def _load_randomization_dimensions(self, seed):
        """ Helper function to load environment defaults ranges
        """
        self.unwrapped.dimensions = []

        with open(self.config_file, mode='r') as f:
            config = json.load(f)

        for dimension in config['dimensions']:
            self.unwrapped.dimensions.append(
                Dimension(
                    default_value=dimension['default'],
                    multiplier_min=dimension['multiplier_min'],
                    multiplier_max=dimension['multiplier_max'],
                    name=dimension['name']
                )
            )

        nrand = len(self.unwrapped.dimensions)
        self.unwrapped.randomization_space = spaces.Box(0, 1, shape=(nrand,), dtype=np.float32)

    def randomize(self, randomized_values=[-1]):
        """Sets the parameter values such that a call to`update_randomized_params()`
        will generate an environment with those settings.

        Passing a list of 'default' strings will give the default value
        Passing a list of 'random' strings will give a purely random value for that dimension
        Passing a list of -1 integers will have the same effect.
        """
        for dimension, randomized_value in enumerate(randomized_values):
            if randomized_value == 'default':
                self.unwrapped.dimensions[dimension].current_value = \
                    self.unwrapped.dimensions[dimension].default_value
            elif randomized_value != 'random' and randomized_value != -1:
                assert 0.0 <= randomized_value <= 1.0, "using incorrect: {}".format(randomized_value)
                self.unwrapped.dimensions[dimension].current_value = \
                    self.unwrapped.dimensions[dimension].rescale(randomized_value)
            else:  # random
                self.unwrapped.dimensions[dimension].randomize()

        self.unwrapped.update_randomized_params()

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)