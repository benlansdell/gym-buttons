"""
A simple OpenAI gym environment consisting of a number of buttons that can be pushed, one of which 
gives reward.
"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class ALE(object):
    def __init__(self):
        self.lives = lambda: 0


class ButtonsFamEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Environment parameters
        self.n_buttons = 10
        self.random_start = True
        self.max_steps = 1000

        # environment setup
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(1, self.n_buttons))
        self.centre = np.array([1, self.n_buttons])
        self.action_space = spaces.Discrete(self.n_buttons+1)
        self.viewer = None

        self._seed()

        # Needed by atari_wrappers in OpenAI baselines
        self.ale = ALE()
        seed = None
        self.np_random, _ = seeding.np_random(seed)

        self.state = np.zeros(self.n_buttons)
        self.count = np.zeros(self.n_buttons)

        self._reset()

    def _reset(self):
        #Set random button state
        if self.random_start:
            self.state = np.zeros(self.n_buttons)
            self.count = np.zeros(self.n_buttons)
        else:
            self.state = np.zeros(self.n_buttons)
            self.count = np.zeros(self.n_buttons)
        self.steps = 0
        ob = self._get_ob()
        return ob

    # This is important because for e.g. A3C each worker should be exploring
    # the environment differently, therefore seeds the random number generator
    # of each environment differently. (This influences the random start
    # location.)
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_ob(self):
        return self.state

    def get_action_meanings(self):
        return ['NOOP'] + [str(i) for i in range(self.n_buttons)]

    def _step(self, action):
        assert action >= 0 and action <= self.n_buttons

        prev_pos = self.pos[:]

        if action == 0:
            # NOOP
            pass
        elif action == 1:
            self.pos[1] += 1
        elif action == 2:
            self.pos[0] += 1
        elif action == 3:
            self.pos[1] -= 1
        elif action == 4:
            self.pos[0] -= 1
        self.pos[0] = np.clip(self.pos[0],
                              self.dot_size[0], 159 - self.dot_size[0])
        self.pos[1] = np.clip(self.pos[1],
                              self.dot_size[1], 209 - self.dot_size[1])

        ob = self._get_ob()

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        dist1 = np.linalg.norm(prev_pos - self.centre)
        dist2 = np.linalg.norm(self.pos - self.centre)
        if dist2 < dist1:
            reward = 1
        elif dist2 == dist1:
            reward = 0
        else:
            reward = -1

        return ob, reward, episode_over, {}

    # Based on gym's atari_env.py
    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # We only import this here in case we're running on a headless server
        from gym.envs.classic_control import rendering
        assert mode == 'human', "Button only supports human render mode"
        img = self._get_ob()
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)

class ButtonsObsEnv(ButtonsFamEnv):
    pass

class ButtonsTestEnv(ButtonsFamEnv):
    pass