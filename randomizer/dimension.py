import numpy as np


class Dimension(object):
    """Class which handles the machinery for doing BO over a particular dimensions
    """
    def __init__(self, default_value, multiplier_min=0.0, multiplier_max=1.0, name=None):
        """Generates datapoints at specified discretization, and initializes BO
        """
        self.default_value = default_value
        self.current_value = default_value
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max
        self.range_min = self.default_value * self.multiplier_min
        self.range_max = self.default_value * self.multiplier_max
        self.name = name

    def _rescale(self, value):
        """Rescales normalized value to be within range of env. dimension
        """
        return self.range_min + (self.range_max - self.range_min) * value

    def randomize(self):
        self.current_value = np.random.uniform(low=self.range_min, high=self.range_max)

    def reset(self):
        self.current_value = self.default_value

    def set(self, value):
        self.current_value = value

