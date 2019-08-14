# domain-randomizer
A standalone library to randomize various OpenAI Gym Environments.

[Domain Randomization](https://arxiv.org/abs/1703.06907) is a idea that helps with sim2real transfer, but surprisingly has no general open source implementations. This library hopes to fill in that gap by providing a _standalone_ library that you can use in your own work.

## Supported Implementations
Within our library, you can find randomized-environment implementations for:
* `LunarLander-v0`
* `Pusher3DoF-v0`
* `Humanoid-v2`
* `HalfCheetah-v2`
* `FetchPush-v1`
* `ResidualPushEnv-v0`
* `ResidualPickAndPlaceEnv-v0`

## Usage

```python

import randomizer
import gym

from randomizer.wrappers import RandomizedEnvWrapper

env = RandomizedEnvWrapper(gym.make('ResidualPickAndPlaceRandomizedEnv-v0'), seed=123)
env.randomize()

'''
List of the parameters that you wish to randomize.
The elements in the list can either be a string ("default") or a number between 0 and 1. (refer below example) 
Important Note : Please maintain the same order as in the corresponding .json file.
''' 

env.randomize(["default", 0.1])  
env.reset()
```

We also provide an implementation for vectorized environments, which is useful in reinforcement learning for parallel training or evaluation.

## Create Your Own Environment

There are three main file to get your randomized environment to work correctly: environment file, configuration file, and environment registration.

### Environment File

Since the standardized `gym` environments (MuJoCo, Box2D, Pybullet) don't include randomization, we'll need to create our own randomization files. For Box2D, the instructions are quite similar (and you can use [Lunar Lander](https://github.com/montrealrobotics/domain-randomizer/blob/master/randomizer/lunar_lander.py) as a starting example), but in this example, we'll look at MuJoCo. 

For MuJoCo, our library rewrites the XML (some parameters however, can be changed [after the XML is compiled](https://github.com/openai/mujoco-py/issues/148), but we find it inconvenient to do this when some parameters can and can't). We [load the XML](https://github.com/montrealrobotics/domain-randomizer/blob/master/randomizer/pusher3dof.py#L21-L22) and then [locate the randomization parameters we're interested in](https://github.com/montrealrobotics/domain-randomizer/blob/master/randomizer/pusher3dof.py#L24-L32). The main function is called `update_randomized_params()`, which gets called by our library on [reset](https://github.com/montrealrobotics/domain-randomizer/blob/master/randomizer/pusher3dof.py#L34-L54).

### Configuration File

To set the ranges that you can randomize, you'll need a [configuration file](https://github.com/montrealrobotics/domain-randomizer/blob/master/randomizer/config/Pusher3DOFRandomized/default.json), which just is a list of JSON objects that tells us what dimensions to randomize, and the ranges (specified by a default value + min/max multipliers).

### Environment Registration

To use the environment, you'll need to [register the environment](https://github.com/montrealrobotics/domain-randomizer/blob/master/randomizer/__init__.py#L25-L30). While this looks like a normal environment file, you'll want to notice the `kwargs`, which our library uses to find the correct configuration file to pull the randomized values from.

