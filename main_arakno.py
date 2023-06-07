import gymnasium as gym
from gymnasium import spaces
import araknoBot
import pybullet_envs
from stable_baselines3 import A2C, PPO
import os

import numpy as np
import time
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,VecVideoRecorder

# from stable_baselines3.common.env_checker import check_env
# env = gym.make('arakno-v0')
# # If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)


#TEST WITH RANDOM ACTIONS
#####################################################################################
# env = gym.make('arakno-v0')
# env.reset()
# while True:
#     env.render()
#     observation, reward, done,_,_ = env.step(env.action_space.sample())
#     if done:
#         observation = env.reset()
#         time.sleep(1/30)
#####################################################################################


#Record a mp4 video 
# video_folder = "/videos/"
# video_length = 100

# load model trained
# model = A2C.load("models/a2c-AraknoEnv-v0.zip")
#comment gym.make for use this
# env = DummyVecEnv([lambda: gym.make('arakno-v0', render_mode="rgb_array")])

# obs = env.reset()


#RECORD VIDEO
#################################################################################################
# Record the video starting at the first step
# vec_env = VecVideoRecorder(env, video_folder,
#                        record_video_trigger=lambda x: x == 0, video_length=video_length,
#                        name_prefix=f"random-agent-arakno-v0")
# env.reset()
# for _ in range(video_length + 1):
#   action, _ = model.predict(obs, deterministic=True)
#   obs, _, _, _ = env.step(action)
# # env.render()
# # Save the video
# env.close()
#################################################################################################


# for i in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
#         print("Goal reached!", "reward=", reward)
#         break

# mean_reward, std_reward = evaluate_policy(model, env)

# print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# env.close()

# # # Enjoy trained agent
model = PPO.load("models/a2c-AraknoEnv-v0_1.zip")
env = DummyVecEnv([lambda: gym.make('arakno-v0')])
env = VecNormalize.load("models/vec_normalize_1.pkl", env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

# print("Enjoy trained agent")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()