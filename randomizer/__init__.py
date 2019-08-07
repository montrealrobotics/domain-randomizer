from gym.envs.registration import register
import os.path as osp

register(
    id='LunarLanderDefault-v0',
    entry_point='randomizer.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'randomizer/config/LunarLanderRandomized/default.json'}
)

register(
    id='LunarLanderRandomized-v0',
    entry_point='randomizer.lunar_lander:LunarLanderRandomized',
    max_episode_steps=1000,
    kwargs={'config': 'randomizer/config/LunarLanderRandomized/random.json'}
)

register(
    id='Pusher3DOFDefault-v0',
    entry_point='randomizer.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'randomizer/config/Pusher3DOFRandomized/default.json'}
)

register(
    id='Pusher3DOFRandomized-v0',
    entry_point='randomizer.pusher3dof:PusherEnv3DofEnv',
    max_episode_steps=100,
    kwargs={'config': 'randomizer/config/Pusher3DOFRandomized/random.json'}
)

register(
    id='HumanoidRandomizedEnv-v0',
    entry_point='randomizer.humanoid:HumanoidRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/HumanoidRandomized/default.json',
        'xml_name': 'humanoid.xml'
    }
)

register(
    id='HalfCheetahRandomizedEnv-v0',
    entry_point='randomizer.half_cheetah:HalfCheetahRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/HalfCheetahRandomized/default.json',
        'xml_name': 'half_cheetah.xml'
    }
)

register(
    id='FetchPushRandomizedEnv-v0',
    entry_point='randomizer.randomized_fetchpush:FetchPushRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/FetchPushRandomized/default.json',
        'xml_name': 'push.xml'
    }
)
register(
    id='ResidualPushRandomizedEnv-v0',
    entry_point='randomizer.randomized_residual_push:ResidualPushRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/ResidualPushRandomized/random.json',
        'xml_name': 'pusher.xml'
    }
)
register(
    id='ResidualPushDefaultEnv-v0',
    entry_point='randomizer.randomized_residual_push:ResidualPushRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/ResidualPushRandomized/default.json',
        'xml_name': 'pusher.xml'
    }
)
