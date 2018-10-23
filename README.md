# gym-buttons

A simple environment for OpenAI gym to test a learning from observation framework. N buttons are presented, only one produces reward. The agent is tasked with using observation to infer which 

Possible actions are:
* 0: do nothing
* [i]: push button i

There are three phases/environments defined:

1. Familiarization phase: no rewards are given. The agent can push buttons.
2. Observation phase: actions do nothing -- buttons are pushed by some unknown process. Here they are uniformaly randomly pushed. Rewars are given when the reward button is pressed. 
3. Test phase: agent can push buttons, and rewards are given. 

The episode terminates after a given number of steps have been taken (by
default 1,000). 

Training with actor-critic (A2C from OpenAI's baselines with one worker) takes
about five minutes to achieve good reward. After about 20 minutes of training,
expect your graphs to look something like:

## Installation

`pip install --user git+https://github.com/benlansdell/gym-buttons`

## Usage

```
import gym_buttons

env = gym.make("Buttons")

# Adjust number of steps before termination
env.n_buttons = 10
env.max_steps = 2000
# Adjust random start
env.random_start = False
```
