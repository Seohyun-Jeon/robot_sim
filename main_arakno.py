import gym
import arakno
import pybullet as p

env = gym.make('arakno-v0')

#EXAMPLE
#obs = env.reset()
#for _ in range(1000):
#    action, _states = model.predict(obs)
#    obs, rewards, dones, info = env.step(action)
#    env.render()
#    env.step(env.action_space.sample()) # take a random action

env.reset()
#env.step(env.action_space.sample())
while True:
    #to learn to walk on a flat plane-> the camera is not needed for now
    env.render()
    #TO DO: step reward action
    p.stepSimulation()    

env.close()