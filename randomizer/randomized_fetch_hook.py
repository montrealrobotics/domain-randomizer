from gym.envs.robotics import rotations, fetch_env
from gym import utils, spaces
import numpy as np
import os
import xml.etree.ElementTree as et
import mujoco_py
from .controllers.hook_controller import get_hook_control

import pdb

DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class RandomizedFetchHookEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, initial_qpos, xml_file=None, **kwargs):

        if xml_file is None:
            xml_file = os.path.join(DIR_PATH, 'assets_residual', 'hook.xml')

        self._goal_pos = np.array([1.65, 0.75, 0.42])
        self._object_xpos = np.array([1.8, 0.75])

        fetch_env.FetchEnv.__init__(
            self, xml_file, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=None, target_range=None, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')

        utils.EzPickle.__init__(self)
        # randomization
        self.xml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets_residual")
        self.reference_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets_residual", kwargs.get('xml_name'))
        self.reference_xml = et.parse(self.reference_path)
        self.config_file = kwargs.get('config')
        self.dimensions = []
        self.dimension_map = []
        self.suffixes = []
        self._locate_randomize_parameters()

    def _locate_randomize_parameters(self):
        self.root = self.reference_xml.getroot()
        self.hook_head = self.root.findall(".//body[@name='hook']/geom[@name='hook_head']")
        self.block_mass = self.root.findall(".//body[@name='hook']/geom")

    def _randomize_block_mass(self):
        mass = self.dimensions[0].current_value
        self.block_mass[0].set('mass', '{:3f}'.format(mass))

    def _create_xml(self):
        self._randomize_block_mass()
        return et.tostring(self.root, encoding='unicode', method='xml')

    def update_randomized_params(self):
        xml = self._create_xml()
        self._re_init(xml)

    def _re_init(self, xml):
        # TODO: Now, likely needs rank
        randomized_path = os.path.join(self.xml_dir, "tmp.xml")
        with open(randomized_path, 'wb') as fp:
            fp.write(xml.encode())
            fp.flush()
        self.model = mujoco_py.load_model_from_path(randomized_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros((4)))
        assert not done
        if self.viewer:
            self.viewer.update_sim(self.sim)

    def render(self, mode="human", *args, **kwargs):
        # See https://github.com/openai/gym/issues/1081
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer().render()
            width, height = 3350, 1800
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

        return super(RandomizedFetchHookEnv, self).render(*args, **kwargs)

    def _sample_goal(self):
        goal_pos = self._goal_pos.copy()
        goal_pos[:2] += self.np_random.uniform(-0.05, 0.05)
        return goal_pos

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 180.
        self.viewer.cam.elevation = -24.

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        object_xpos_x = 1.65 + self.np_random.uniform(-0.05, 0.05)
        while True:
            object_xpos_x = 1.8 + self.np_random.uniform(-0.05, 0.10)
            object_xpos_y = 0.75 + self.np_random.uniform(-0.05, 0.05)
            if (object_xpos_x - self._goal_pos[0]) ** 2 + (object_xpos_y - self._goal_pos[1]) ** 2 >= 0.01:
                break
        self._object_xpos = np.array([object_xpos_x, object_xpos_y])

        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = self._object_xpos
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])

        return obs

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)
