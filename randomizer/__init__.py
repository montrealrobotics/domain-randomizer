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
        'xml_name': 'push.xml'
    }
)
register(
    id='ResidualPushDefaultEnv-v0',
    entry_point='randomizer.randomized_residual_push:ResidualPushRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/ResidualPushRandomized/default.json',
        'xml_name': 'push.xml'
    }
)

register(
    id='ResidualPickAndPlaceRandomizedEnv-v0',
    entry_point='randomizer.randomized_pick_and_place:ResidualPickAndPlaceRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/ResidualPickAndPlaceRandomized/random.json',
        'xml_name': 'pick_and_place.xml'
    }
)
register(
    id='ResidualPickAndPlaceDefaultEnv-v0',
    entry_point='randomizer.randomized_pick_and_place:ResidualPushRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/ResidualPickAndPlaceRandomized/default.json',
        'xml_name': 'pick_and_place.xml'
    }
)
register(
    id='ResidualMPCPushRandomizedEnv-v0',
    entry_point='randomizer.randomized_mpc_push:ResidualMPCPushRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/ResidualMPCPushRandomized/random.json',
        'xml_name': 'push.xml'
    }
)
register(
    id='ResidualMPCPushDefaultEnv-v0',
    entry_point='randomizer.randomized_mpc_push:ResidualMPCPushRandomizedEnv',
    max_episode_steps=1000,
    kwargs={
        'config': 'randomizer/config/ResidualMPCPushRandomized/default.json',
        'xml_name': 'push.xml'
    }
)