import numpy as np
from .controllers.hook_controller import get_hook_control
from .randomized_hook_env import RandomizedFetchHookEnv
from gym.envs.robotics import rotations, fetch_env
from gym import utils, spaces


class ResidualFetchHookEnv(RandomizedFetchHookEnv):

    def __init__(self, **kwargs):
        super(ResidualFetchHookEnv, self).__init__(**kwargs)

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


class NoisyResidualFetchHookEnv(RandomizedFetchHookEnv):

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = RandomizedFetchHookEnv.step(self, action)

        self._last_observation = observation

        return observation, reward, done, debug_info

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]  # object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)

    def compute_reward(self, *args, **kwargs):
        return RandomizedFetchHookEnv.compute_reward(self, *args, **kwargs)

    def reset(self):
        observation = RandomizedFetchHookEnv.reset(self)
        self._last_observation = observation

        return observation


class TwoFrameResidualHookNoisyEnv(RandomizedFetchHookEnv):
    def __init__(self, xml_file=None, **kwargs):
        super(TwoFrameResidualHookNoisyEnv, self).__init__(**kwargs)
        self.observation_space.spaces['observation'] = spaces.Box(low=np.hstack(
            (self.observation_space.spaces['observation'].low, self.observation_space.spaces['observation'].low)),
                                                                  high=np.hstack((self.observation_space.spaces[
                                                                                      'observation'].high,
                                                                                  self.observation_space.spaces[
                                                                                      'observation'].high)),
                                                                  dtype=np.float32)

    def _get_obs(self):
        obs = fetch_env.FetchEnv._get_obs(self)

        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        hook_pos = self.sim.data.get_site_xpos('hook')
        hook_pos = self._noisify_obs(hook_pos, noise=0.025)
        # rotations
        hook_rot = rotations.mat2euler(self.sim.data.get_site_xmat('hook'))
        hook_rot = self._noisify_obs(hook_rot, noise=0.025)
        # velocities
        hook_velp = self.sim.data.get_site_xvelp('hook') * dt
        hook_velr = self.sim.data.get_site_xvelr('hook') * dt
        # gripper state
        hook_rel_pos = hook_pos - grip_pos
        hook_velp -= grip_velp

        hook_observation = np.concatenate([hook_pos, hook_rot, hook_velp, hook_velr, hook_rel_pos])

        obs['observation'] = np.concatenate([obs['observation'], hook_observation])
        obs['observation'][3:5] = self._noisify_obs(obs['observation'][3:5], noise=0.025)
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][:3]  # object_pos - grip_pos
        obs['observation'][12:15] = self._noisify_obs(obs['observation'][6:9], noise=0.025)
        return obs

    def step(self, residual_action):
        residual_action = 2. * residual_action

        action = np.add(residual_action, get_hook_control(self._last_observation))
        action = np.clip(action, -1, 1)
        observation, reward, done, debug_info = RandomizedFetchHookEnv.step(self, action)

        obs_out = observation.copy()
        obs_out['observation'] = np.hstack((self._last_observation['observation'], observation['observation']))
        self._last_observation = observation

        return obs_out, reward, done, debug_info

    def reset(self):
        observation = RandomizedFetchHookEnv.reset(self)
        self._last_observation = observation.copy()
        observation['observation'] = np.hstack((self._last_observation['observation'], observation['observation']))
        return observation

    def _noisify_obs(self, obs, noise=1.):
        return obs + np.random.normal(0, noise, size=obs.shape)




