from gym.envs.registration import register

from common.envs.config import CONFIG_PATH
from common.envs.lunar_lander import LunarLanderRandomized
import os.path as osp

register(
    id='LunarLanderDefault-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/default.json'}
)

register(
    id='LunarLanderRandomized-v0',
    entry_point='common.envs.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'common/envs/config/LunarLanderRandomized/random.json'}
)

register(
    id='Pusher3DOFDefault-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/default.json'}
)

register(
    id='Pusher3DOFRandomized-v0',
    entry_point='common.envs.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'common/envs/config/Pusher3DOFRandomized/random.json'}
)