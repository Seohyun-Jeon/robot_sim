from gym.envs.registration import register
register(
    id='arakno-v0', 
    entry_point='arakno.envs:AraknoEnv'
)