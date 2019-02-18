import numpy as np


class Dimension(object):
    """Class which handles the machinery of randomizing a particular dimension
    """
    def __init__(self, default_value, multiplier_min=0.0, multiplier_max=1.0, name=None):
        self.default_value = default_value
        self.current_value = default_value
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max
        self.range_min = self.default_value * self.multiplier_min
        self.range_max = self.default_value * self.multiplier_max
        self.name = name

    def randomize(self):
        self.current_value = np.random.uniform(low=self.range_min, high=self.range_max)

    def rescale(self, value):
        return self.range_min + (self.range_max - self.range_min) * value

    def reset(self):
        self.current_value = self.default_value

    def set(self, value):
        self.current_value = value

