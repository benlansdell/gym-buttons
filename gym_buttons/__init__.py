from gym.envs.registration import register

register(
    id='ButtonsFam-v0',
    entry_point='gym_buttons.envs:ButtonsFamEnv'
)

register(
    id='ButtonsObs-v0',
    entry_point='gym_buttons.envs:ButtonsObsEnv'
)

register(
    id='ButtonsTest-v0',
    entry_point='gym_buttons.envs:ButtonsTestEnv'
)
