from gym.envs.registration import register


########################################################################################
# POINTMASS
register(
    id='pointmass-v0',
    entry_point='gym_env.pointmass.pointmass_env:PointMassEnv',
    kwargs={'args': None},
    max_episode_steps=64
)
