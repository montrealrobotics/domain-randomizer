import os
from gym import utils
from .residual_pick_and_place import RandomizedResidualPickPlaceEnv

MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')


# RANDOMIZED RESIDUAL PICK AND PLACE ENVIRONMENT
class ResidualPickAndPlaceRandomizedEnv(RandomizedResidualPickPlaceEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', **kwargs):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        RandomizedResidualPickPlaceEnv.__init__(self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
                                    gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
                                    obj_range=0.15, target_range=0.15, distance_threshold=0.05,
                                    initial_qpos=initial_qpos, reward_type=reward_type, **kwargs)

        utils.EzPickle.__init__(self)
