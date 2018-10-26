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

        self.rewardaction = np.random.randint(self.n_buttons)

        self.reset()

    def reset(self):
        #Set random button state
        if self.random_start:
            self.state = np.zeros(self.n_buttons)
            self.count = np.zeros(self.n_buttons)
        else:
            self.state = np.random.rantint(2, size=self.n_buttons)
            self.count = np.random.rantint(self.maxtime, size=self.n_buttons)*self.state
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
        return [str(i) for i in range(self.n_buttons)] + ['NOOP']

    def step(self, action):
        assert action >= 0 and action <= self.n_buttons

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        #Familiarization environment: no rewards
        reward = 0

        #Push a button
        if action < self.n_buttons:
            if self.count[action] == 0:
                self.count[action] = 2 + np.random.randint(5)

        #Decrease counter
        self.count -= 1
        self.count = np.maximum(0, self.count)

        #Update state
        self.state = (self.count > 0).astype(int)

        ob = self._get_ob()

        return ob, reward, episode_over, {}

    # Based on gym's atari_env.py
    def render(self, mode='human', close=False):
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
    def __init__(self):
        super(ButtonsObsEnv, self).__init__()
        self.action_prob = 0.5

    def step(self, action):
        assert action >= 0 and action <= self.n_buttons

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        #Observation environment: rewards
        reward = 0

        action = self.n_buttons
        #Choose an action at random
        if np.random.rand() < self.action_prob:
            action = np.random.randint(self.n_buttons)
            if self.count[action] == 0:
                self.count[action] = 2 + np.random.randint(5)
                if action == self.rewardaction:
                    reward = 1

        #Decrease counter
        self.count -= 1
        self.count = np.maximum(0, self.count)

        #Update state
        self.state = (self.count > 0).astype(int)

        ob = self._get_ob()

        return ob, reward, episode_over, {}

class ButtonsTestEnv(ButtonsFamEnv):
    def step(self, action):
        assert action >= 0 and action <= self.n_buttons

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        reward = 0

        #Push a button
        if action < self.n_buttons:
            if self.count[action] == 0:
                self.count[action] = 2 + np.random.randint(5)
                if action == self.rewardaction:
                    reward = 1

        #Decrease counter
        self.count -= 1
        self.count = np.maximum(0, self.count)

        #Update state
        self.state = (self.count > 0).astype(int)

        ob = self._get_ob()

        return ob, reward, episode_over, {}