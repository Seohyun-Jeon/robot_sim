import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

#TO DO: reset function-> do it need a return ?
#change the reward function
#add prev_state

#----------------------------------------------------------------#
#GOAL of the ENV -> reach a specific endpoint in a plane 
#in the fewest step possible ? or in the more stable way(cosider an uneven terrain)
#----------------------------------------------------------------#

class AraknoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        client = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81, physicsClientId=client)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf", basePosition = [0, 0, 0])

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        
        path_urdf = 'arakno/resources/urdfs/arakno.urdf'
        init_position = [0,0,0.1]
        init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.araknoId = p.loadURDF(path_urdf, init_position, init_orientation)
        num_joints = p.getNumJoints(self.araknoId)

        for joint in range(num_joints):
            info = p.getJointInfo(self.araknoId, joint)
            print("INFO: ", info[0], ": ", info[1], " joint type: ", info[2])
            print("---------------------------------------------------")
        
        self._max_n_steps = 1000
        self.episode = 0
        self.endpoint = (3,3) #point in the 2d plane

        #init of the prev state 
        #(x, y, z), ori = p.getBasePositionAndOrientation(self.araknoId)
        self.prev_pos = None
        self.prev_ori = None

        self.num_joints = 3
        self.num_legs = 4
        # Define the action space
        #like the action space of the gym ant env
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints * self.num_legs,), dtype=np.float32)
        # Define the observation space
        # assume 6 observations for spider + 4 legs with (num_joints_per_leg + 1) states per leg (including foot)
        #Each leg has num_joints + 1 observation dim, representing the joint angles and the foot position.
        #include 6 dim to represent the position and orientation of the spider robot in 3D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_legs * (self.num_joints + 1) + 6,), dtype=np.float32)

    def get_pos(self):
        # Get the robot's base position and orientation
        pos, ori = p.getBasePositionAndOrientation(self.araknoId)
        x, y, z = pos  # Extract the x, y, and z components
        return (x, y, z)
    
    def check_done(self):
        #check if the experiment is done by checking the following conditions:
        #1.reached the endpoint
        #2.fallen on the ground
        #3.maximum number timesteps reached
        done = False
        curr_pos = self.get_pos()
        if (self.compute_progress(self,curr_pos, self.endpoint)<0.005) or (self.is_alive()) or (self.episode >= self._max_n_steps) :
            done = True

        # Return whether the episode is done or not
        return done

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        target_positions = self.calculate_target_positions(action)
        p.setJointMotorControlArray(self.robot, self.joint_indices, p.POSITION_CONTROL, targetPositions=target_positions)
        p.stepSimulation()
        
        observation = self.get_observation()
        #prev_pos, curr_pos, prev_orient, curr_orient
        reward = 0 # self.compute_reward(#params)
        done = self.check_done()
        info = {}

        p.stepSimulation()

        return observation, reward, done, info
    
    def get_observation(self):
        #the observation space in a 37-dimensional array, normalize 0-1
        #3 joint per each leg(4 legs)
        #generalized coordinates + generalized velocities -> 19 dim + 18 dim (like aliengo lol)

        #take as observation space the position and the velocity of each joint in array

        #getJointState -> output[0]: position output[1] velocity
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
        
        #print("obs dim: ", len(observation)) #should be 37 dim, canceled the camera_joint

        return np.array(observation)
    
    #Implement termination condition
    #define the terminal status, if true the experiment should stop 
    def is_alive(self):
        # Get the position and orientation of the robot's base
        pos, orn = p.getBasePositionAndOrientation(self.araknoId)
        #COM pos, height of body base => 0,15
        ground_height = 0.076
        # Check if the center of mass (z-component) is above the ground
        return pos[2] > ground_height

    def compute_progress(self,curr_pos):
        # Compute the distance between the current position and the endpoint

        num_steps = 1
        step_size = 1

        # Compute the robot's progress
        #divided by the total distance it could have traveled in num_steps time steps of size step_size
        # Compute the robot's progress in both x and y directions
        dx = (self.endpoint[0] - curr_pos[0]) / (num_steps * step_size)
        dy = (self.endpoint[1] - curr_pos[1]) / (num_steps * step_size)
        progress = math.sqrt(dx**2 + dy**2)

        return progress
    
    #Progress -> define an endpoint that the robot need to reach in 2d plane
    #define the difference between the pos of the robot and the endpoint
    #use the Frobenius norm of the difference of the position
    #This encourages the robot to move forward and make progress over time
    def compute_reward(self,prev_pos, curr_pos, prev_orient, curr_orient):
        # Compute the robot's forward velocity
        forward_vel = (curr_pos[0] - prev_pos[0]) / dt

        # Compute the robot's stability
        roll_diff = abs(curr_orient[0] - prev_orient[0])
        pitch_diff = abs(curr_orient[1] - prev_orient[1])
        yaw_diff = abs(curr_orient[2] - prev_orient[2])
        stability = 1.0 / (1.0 + roll_diff + pitch_diff + yaw_diff)

        # Compute the difference between the current progress and the previous progress
        progress_diff = self.compute_progress(curr_pos, self.endpoint) - self.compute_progress(prev_pos, self.endpoint)

        alive = self.is_alive()
        # Penalize if the robot has fallen or is not alive
        if not alive:
            alive_penalty = -1.0
        else:
            alive_penalty = 0.0

        # Compute the reward as a weighted sum of forward velocity, stability, and progress
        reward = forward_vel * 0.5 + stability * 0.3 + progress_diff * 0.2 + alive_penalty

        return reward

    def reset(self):
        #reset the environment
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setGravity(0,0,-10)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf", basePosition = [0, 0, 0])

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        
        path_urdf = 'arakno/resources/urdfs/arakno.urdf'
        init_position = [0,0,0.1]
        init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.araknoId = p.loadURDF(path_urdf, init_position, init_orientation)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        
        #return 

    #render the camera images
    def render(self, mode='human'):
        #TO DO: attach the camera to the camera joint and fixed the joint to a specific angulation  
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()