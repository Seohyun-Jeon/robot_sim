import gym
import arakno
import pybullet as p
import numpy as np
import pybullet_envs
from stable_baselines3 import A2C
import os

# initialize environment and agent
#env = gym.make("AntBulletEnv-v0")
env = gym.make('arakno-v0')
print(env.observation_space.shape[0], env.action_space.shape[0])

env.render()

# create RL model
model = A2C("MlpPolicy", env, verbose=1)

# train model on environment
model.learn(total_timesteps=int(1e6))

# test model on environment
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()


"""
# Start the state logging
#log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "temp_video.mp4")

env.reset()
#env.step(env.action_space.sample())
for i in range(1000):
    #to learn to walk on a flat plane-> the camera is not needed for now
    env.render()
    #TO DO: step reward action
    #    obs, rewards, dones, info = env.step(action)
    env.render()
#    env.step(env.action_space.sample()) # take a random action
    p.stepSimulation()    

# Stop the state logging
#p.stopStateLogging(log_id)

env.close()"""