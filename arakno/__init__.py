from gym.envs.registration import register
register(
    id='arakno-v0', 
    entry_point='arakno.envs:AraknoEnv',
    max_episode_steps=1000
)