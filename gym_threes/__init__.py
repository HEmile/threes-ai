from gym.envs.registration import register

register(
    id='threes-v0',
    entry_point='gym_threes.envs:ThreesEnv',
)
