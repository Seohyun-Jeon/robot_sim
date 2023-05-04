import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

#TO DO:reward neagative or only positive ? 
#

#----------------------------------------------------------------#
#GOAL of the ENV -> reach a specific endpoint in a plane 
#in the fewest step possible ? or in the more stable way(cosider an uneven terrain)
#----------------------------------------------------------------#

import gym
from gym import spaces
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data

import time
import math
import numpy as np

class AraknoEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']
    }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __init__(self, render = True):
        super(AraknoEnv, self).__init__()

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        p.setGravity(0, 0, -10)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # used by loadURDF

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self._seed()

        p.resetSimulation()
        p.setGravity(0, 0, -10)  # m/s^2
        p.setTimeStep(0.01)   # sec

        #load models
        self.plane = p.loadURDF("plane.urdf")
        
        path_urdf = 'arakno/resources/urdfs/arakno.urdf'
        self.init_position = [0,0,0.25]
        self.init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.araknoId = p.loadURDF(path_urdf, self.init_position, self.init_orientation)

        # calculate the position of the endpoint 1 kilometer away from the start position
        direction = np.array([1, 0, 0])  # direction as a unit vector
        self.endpoint = self.init_position + 2 * direction

        self.init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.goalId = p.loadURDF('arakno/resources/urdfs/goal.urdf', self.endpoint, self.init_orientation)

        num_joints = p.getNumJoints(self.araknoId)
        for joint in range(num_joints):
            info = p.getJointInfo(self.araknoId, joint)
            print("INFO: ", info[0], ": ", info[1], " joint type: ", info[2])
            print("---------------------------------------------------")
            info = p.getDynamicsInfo(self.araknoId, joint)
            print("mass: ", info[0], " inertia: ", info[2])
            print("---------------------------------------------------")
        
        # Set the joint angles init config
        self.joint_angles = [0.0,0.0,1.0, 0.0,0.0,1.0, 0.0,0.0,1.0, 0.0,0.0,1.0]
        for i in range(len(self.joint_angles)):
            p.resetJointState(self.araknoId, i, self.joint_angles[i])

        # set the start position
        self.start_position,_ = p.getBasePositionAndOrientation(self.araknoId)

        # calculate the position of the endpoint 1 kilometer away from the start position
        direction = np.array([1, 0, 0])  # direction as a unit vector
        self.endpoint = self.start_position + 2 * direction #100 meters

        self.movingJoints = [0,1,2,3,4,5,6,7,8,9,10,11]

        # Define the action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32) #35
    
    def reset(self):
        #reset the environment
        p.resetSimulation()
        p.setGravity(0, 0, -10)  # m/s^2
        p.setTimeStep(0.01) 
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything

        self.envStepCounter = 0

        self.vt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.vd = 0
        self.maxV = 5# 8.72  # 0.12sec/60 deg = 500 deg/s = 8.72 rad/s

        #load models
        self.plane = p.loadURDF("plane.urdf")
        p.resetBasePositionAndOrientation(self.araknoId,self.init_position, self.init_orientation)

        self.araknoId = p.loadURDF('arakno/resources/urdfs/arakno.urdf', self.init_position, self.init_orientation)

        p.resetJointState(self.araknoId, 2, 1.0)
        p.resetJointState(self.araknoId, 5, 1.0)
        p.resetJointState(self.araknoId, 8, 1.0)
        p.resetJointState(self.araknoId, 11, 1.0)

        p.addUserDebugText('GOAL', [2, 0 ,0.4], [1, 0, 0])

        self.goalId = p.loadURDF('arakno/resources/urdfs/goal.urdf', self.endpoint, self.init_orientation)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        #init of the prev state 
        self.prev_pos, self.prev_ori = p.getBasePositionAndOrientation(self.araknoId)

        #get observation 
        observation = self.compute_observation()
        
        return observation

    def step(self, action):

        self.assign_throttle(action)

        observation = self.compute_observation()

        reward = self.compute_reward()

        done = self.check_done()

        self.envStepCounter += 1

        return observation, reward, done, {}
    
    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)
    
    def moveLeg(self, id, target): 
      p.setJointMotorControl2(
          bodyUniqueId=self.araknoId,
          jointIndex=id,
          controlMode=p.POSITION_CONTROL,
          targetPosition=target)

    #manage the speed of a robot's wheels or motors
    #allows the robot to move at a desired speed
    #it can help to prevent the robot from moving too quickly or exerting too much force, 
    # which could cause damage to the robot or its surroundings.
    def assign_throttle(self, action):
        for i, key in enumerate(self.movingJoints):
            #clamp the value to a range [-2,2]
            self.vt[i] = self.clamp(self.vt[i] + action[i], -2, 2)
            self.moveLeg(id=key,  target=self.vt[i])
    
    def compute_observation(self):

        observation = []

        n_joints = p.getNumJoints(self.araknoId)

        #access to the current position and orientation of the base of the body 
        base_pos, base_or = p.getBasePositionAndOrientation(self.araknoId)

        observation.append(base_pos[0])
        observation.append(base_pos[1])

        observation.append(base_pos[2])
        observation.append(base_or[0])
        observation.append(base_or[1])
        observation.append(base_or[2])
        observation.append(base_or[3])

        #take generalized coordinates
        for i in range(n_joints):
            joint_state = p.getJointState(self.araknoId, i)[0] # position
            observation.append(joint_state)
        
        #access to linear and angular velocity of the base of a body 
        base_v, base_w = p.getBaseVelocity(self.araknoId)
        observation.append(base_v[0])
        observation.append(base_v[1])
        observation.append(base_v[2])
        observation.append(base_w[0])
        observation.append(base_w[1])
        observation.append(base_w[2])

        #take generalized velocities
        for i in range(n_joints):
            joint_state = p.getJointState(self.araknoId, i)[1] # velocity
            observation.append(joint_state)

        #take externals forces
        JointStates = p.getJointStates(self.araknoId, self.movingJoints)
        for joint in JointStates:
            observation.append(joint[2])

        return observation
    
    def calculate_center_of_mass(self, robot_id):
        total_mass = 0
        com = np.array([0.0, 0.0, 0.0])
        for link_id in range(p.getNumJoints(robot_id)):
            link_info = p.getJointInfo(robot_id, link_id)
            link_mass = link_info[0]
            link_pos, link_orn = p.getLinkState(robot_id, link_id)[:2]
            link_com = np.array(p.getLinkState(robot_id, link_id, computeLinkVelocity=0, computeForwardKinematics=1)[0]) - np.array(link_pos)
            com += link_mass * link_com
            total_mass += link_mass
        com /= total_mass

        return com
    
    def compute_reward(self):

        baseOri = p.getBasePositionAndOrientation(self.araknoId)
        xposbefore = baseOri[0][0]

        dx_before = (self.endpoint[0] - xposbefore)
 
        xvelbefore = p.getBaseVelocity(self.araknoId)[0][0]

        p.stepSimulation()
        
        baseOri = p.getBasePositionAndOrientation(self.araknoId)
        xposafter = baseOri[0][0]

        dx_after = (self.endpoint[0] - xposafter)

        xvelafter  = p.getBaseVelocity(self.araknoId)[0][0]

        # Compute the difference between the current progress and the previous progress
        progress_diff = dx_after - dx_before

        com = self.calculate_center_of_mass(self.araknoId)

        # Compute the robot's stability
        roll_diff = abs(com[0] - baseOri[0][0])
        pitch_diff = abs(com[1] - baseOri[0][1])
        yaw_diff = abs(com[2] - baseOri[0][2])
        stability = 1.0 / (1.0 + roll_diff + pitch_diff + yaw_diff)

        # forward_reward = (xposafter - xposbefore)
        forward_reward = 20 * (xvelbefore - xvelafter)

        #reward from applied torque to the motor in the last timestep
        JointStates = p.getJointStates(self.araknoId, self.movingJoints)
        torques = np.array([np.array(joint[3]) for joint in JointStates])
        ctrl_cost = 1.0 * np.square(torques).sum()

        #help the robot to detect when it has made contact with the ground, and provide 
        # information about the force and direction of the ground reaction forces
        #chech contact point with ground
        ContactPoints = p.getContactPoints(self.araknoId, self.plane)
        contact_cost = 5 * 1e-1 * len(ContactPoints)

        # Penalize if the robot has fallen or is not alive
        alive = self.is_alive()
        if not alive:
            alive_penalty = -1.0
        else:
            alive_penalty = 0.0
        
        reward = forward_reward - ctrl_cost - contact_cost + alive_penalty + progress_diff + stability

        #reward = reward if reward > 0 else 0
        
        return reward
    
    def check_done(self):
        #check if the experiment is done by checking the following conditions:
        #1.reached the endpoint
        #2.fallen on the ground
        #3.maximum number timesteps reached
        done = False
        (x,y,z), _ = p.getBasePositionAndOrientation(self.araknoId)
        curr_pos = (x,y,z)
        if (not self.is_alive()) or (self.envStepCounter >= 1000) :
            print("You Dead")
            done = True

        # Return whether the episode is done or not
        return done

    def is_alive(self):
        # Get the position and orientation of the robot's base
        pos, orn = p.getBasePositionAndOrientation(self.araknoId)
        #COM pos, height of body base => 0,15
        ground_height = 0.087
        # Check if the center of mass (z-component) is above the ground
        if pos[2] > ground_height:
            return True
        else:
            return False
    

    def render(self, mode='human', close = False):
        pass

    def close(self):
        p.disconnect()