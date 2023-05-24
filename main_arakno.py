import gym
import arakno
import pybullet as p
import numpy as np
import pybullet_envs
from stable_baselines3 import A2C
import os

import gym
import numpy as np
import time


import gym
import pybullet as p
import numpy as np
import pybullet_envs
from stable_baselines3 import PPO, A2C, DQN
import os

import gym
import numpy as np
import time
import pybullet_envs
import panda_gym
import gym

import os

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login


env = gym.make('arakno-v0')

# n_cpu = 4
# #total_timesteps = 200000000
# total_timesteps = 200000
# env = SubprocVecEnv([lambda: gym.make('arakno-v0') for i in range(n_cpu)])
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=total_timesteps)
# model.save("experience_learned/ppo2_araknor_v0_testing")
# del model # remove to demonstrate saving and loading


# env.reset()
# while True:
#     env.render()
#     observation, reward, done, info = env.step(env.action_space.sample())
#     if done:
#         observation = env.reset()
#         time.sleep(1/30)

# print("ENDDDDDDDDDDDD")

#Instantiate the env
vec_env = make_vec_env(env, n_envs=1)
env = SubprocVecEnv([lambda: gym.make('arakno-v0') for i in range(4)])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = A2C("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(20000)#200000000

# Save the model and  VecNormalize statistics when saving the agent
model.save("a2c-AraknoEnv-v0")
env.save("vec_normalize.pkl")

# # initialize environment and agent
# env = gym.make('arakno-v0')
# #env.render()

# # create RL model
# model = A2C("MlpPolicy", env, verbose=1)

# # train model on environment
# model.learn(total_timesteps=int(1e6))

# # test model on environment
# obs = env.reset()

# for i in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         print(f'Done at step {step}')
#         obs = env.reset()
#         time.sleep(1/30)

#evaluate the performance
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# import gym
# import pybullet as p
# import numpy as np
# import pybullet_envs
# from stable_baselines3 import PPO, A2C, DQN
# import os

# import gym
# import numpy as np
# import time

# # # # Enjoy trained agent
# # Load the agent
# model = A2C.load("a2c-AraknoEnv-v0")
# env = DummyVecEnv([env])
# eval_env = VecNormalize.load("vec_normalize.pkl", env)
# #  do not update them at test time
# eval_env.training = False
# # reward normalization is not needed at test time
# eval_env.norm_reward = False

# obs = eval_env.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     eval_env.render()

# Load the saved statistics
#eval_env = DummyVecEnv([env])

#mean_reward, std_reward = evaluate_policy(model, env)

#print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
