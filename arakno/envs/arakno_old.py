import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

#TO DO:The Env work with the gym functions but the robot doesn't move, recheck the reward function, check_done and step_iter if need it

#----------------------------------------------------------------#
#GOAL of the ENV -> reach a specific endpoint in a plane 
#in the fewest step possible ? or in the more stable way(cosider an uneven terrain)
#----------------------------------------------------------------#

class AraknoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        p.connect(p.GUI)
        p.setGravity(0, 0, -10)

        filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
        self.planeId = p.loadSDF(filename)

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        
        path_urdf = 'arakno/resources/urdfs/arakno.urdf'
        self.init_position = [0,0,0.5]
        self.init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.araknoId = p.loadURDF(path_urdf, self.init_position, self.init_orientation)

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
        
        self._max_n_steps = 2000
        self.envStepCounter = 0

        # set the start position
        self.start_position,_ = p.getBasePositionAndOrientation(self.araknoId)

        # calculate the position of the endpoint 1 kilometer away from the start position
        direction = np.array([1, 0, 0])  # direction as a unit vector
        self.endpoint = self.start_position + 2 * direction #100 meters
        #urdfRootPath=pybullet_data.getDataPath()
        #self.box = p.loadURDF("mug.urdf",self.endpoint)
        #pandaUid = p.loadURDF(os.path.join(urdfRootPath, "mug.urdf"),useFixedBase=True, basePosition = self.endpoint)

        #make all simulations similar
        self._seed()

        #init of the prev state 
        self.prev_pos, self.prev_orient = p.getBasePositionAndOrientation(self.araknoId)

        self.num_joints = 3
        self.num_legs = 4
        # Define the action space
        #like the action space of the gym ant env
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints * self.num_legs,), dtype=np.float32)
        # Define the observation space
        # assume 6 observations for spider + 4 legs with (num_joints_per_leg + 1) states per leg (including foot)
        #Each leg has num_joints observation dim, representing the joint angles
        #include 6 dim to represent the position and orientation of the spider robot in 3D
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_legs * (self.num_joints) + 6,), dtype=np.float32)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def check_done(self):
        #check if the experiment is done by checking the following conditions:
        #1.reached the endpoint
        #2.fallen on the ground
        #3.maximum number timesteps reached
        done = False
        (x,y,z), _ = p.getBasePositionAndOrientation(self.araknoId)
        curr_pos = (x,y,z)
        if (self.compute_progress(curr_pos)<0.005) or (not self.is_alive()) or (self.envStepCounter >= self._max_n_steps) :
            done = True

        # Return whether the episode is done or not
        return done

    def apply_action(self,action):
        #action -> from the main, expected to be a list of action for all the joints 
        # list dim = 12 
        #position control
        for leg in range(self.num_legs):
            for joint in range(self.num_joints):
                joint_id = leg * 3 + joint
                p.setJointMotorControl2(self.araknoId, joint_id, p.POSITION_CONTROL, targetPosition=action[joint_id])
        #simulate the action
        p.stepSimulation()

    def step(self, action):
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # #compute an action 
        # #self.apply_action(action)

        for leg in range(12):
            #for joint in range(self.num_joints):
            #joint_id = leg * 3 + joint
            p.setJointMotorControl2(self.araknoId, leg, p.POSITION_CONTROL, targetPosition=action[leg])
        #simulate the action
        p.stepSimulation()

        # #get the results
        # observation = self.get_observation()

        # #compute the reward
        # #get the current pose of the robot base after taking the action
        # curr_pos = (observation[0],observation[1],observation[2])
        # curr_orient = (observation[3],observation[4],observation[5])
        # curr_orient = p.getQuaternionFromEuler(curr_orient)

        # reward = self.compute_reward(self.prev_pos, curr_pos, self.prev_orient, curr_orient,action)

        # self.envStepCounter += 1

        # #check if done
        # done = self.check_done()

        # # Update the previous pose to be the current pose
        # self.prev_pos = curr_pos
        # self.prev_orient = curr_orient

        # #info-> if you want to check additional informations 
        # #info = {}

        #return observation, reward, done, {}
        return {}, {},{},{}
    
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
        base_or = p.getEulerFromQuaternion(base_or)
        observation.append(base_pos[0])
        observation.append(base_pos[1])
        observation.append(base_pos[2])
        observation.append(base_or[0])
        observation.append(base_or[1])
        observation.append(base_or[2])
        #observation.append(base_or[3])

        #take generalized coordinates
        for i in range(n_joints):
            joint_state = p.getJointState(self.araknoId, i)[0] # position
            observation.append(joint_state)
        
        #access to linear and angular velocity of the base of a body 
        """ 
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
            observation.append(joint_state)"""
        
        #print("obs dim: ", len(observation)) #should be 37 dim, canceled the camera_joint

        return np.array(observation)
    
    #Implement termination condition
    #define the terminal status, if true the experiment should stop 
    def is_alive(self):
        # Get the position and orientation of the robot's base
        pos, orn = p.getBasePositionAndOrientation(self.araknoId)
        #COM pos, height of body base => 0,15
        ground_height = 0.08
        # Check if the center of mass (z-component) is above the ground
        return pos[2] > ground_height

    def compute_progress(self,pos):
        # Compute the distance between the current position and the endpoint
        num_steps = 1
        step_size = 1

        # Compute the robot's progress
        #divided by the total distance it could have traveled in num_steps time steps of size step_size
        # Compute the robot's progress in both x and y directions
        dx = (self.endpoint[0] - pos[0]) / (num_steps * step_size)
        dy = (self.endpoint[1] - pos[1]) / (num_steps * step_size)
        progress = math.sqrt(dx**2 + dy**2)

        return progress
    
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
    
    #Progress -> define an endpoint that the robot need to reach in 2d plane
    #define the difference between the pos of the robot and the endpoint
    #use the Frobenius norm of the difference of the position
    #This encourages the robot to move forward and make progress over time
    def compute_reward(self,prev_pos, curr_pos, prev_orient, curr_orient, action):

        com = self.calculate_center_of_mass(self.araknoId)

        # Compute the robot's stability
        roll_diff = abs(com[0] - curr_pos[0])
        pitch_diff = abs(com[1] - curr_pos[1])
        yaw_diff = abs(com[2] - curr_pos[2])
        stability = 1.0 / (1.0 + roll_diff + pitch_diff + yaw_diff)

        # Compute the difference between the current progress and the previous progress
        progress_diff = self.compute_progress(prev_pos) - self.compute_progress(curr_pos)

        # Penalize if the robot has fallen or is not alive
        alive = self.is_alive()
        if not alive:
            alive_penalty = -1.0
        else:
            alive_penalty = 0.0
        
        ctrl_cost = 0.5 * np.square(action).sum()

        # Compute the reward as a weighted sum of forward velocity, stability, and progress
        reward = stability*0.2 + progress_diff*0.8  + alive_penalty - ctrl_cost
        #print("reward: ", reward)
        #reward -> If your action was beneficial or not
        return reward

    def reset(self):
        #reset the environment
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        p.setGravity(0,0,-10)

        #p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #self.planeId = p.loadURDF('arakno/resources/urdfs/simpleplane.urdf', basePosition = [0, 0, 0])
        filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
        self.planeId = p.loadSDF(filename)

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        
        path_urdf = 'arakno/resources/urdfs/arakno.urdf'
        self.init_position = [0,0,0.5]
        self.araknoId = p.loadURDF(path_urdf, self.init_position, self.init_orientation)
        
        #for i in range(12):
        #    p.changeDynamics(self.araknoId, i, linearDamping=0, angularDamping=0)
        
        # Set the joint angles init config
        for i in range(len(self.joint_angles)):
            p.resetJointState(self.araknoId, i, self.joint_angles[i])

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        #init of the prev state 
        self.prev_pos, self.prev_ori = p.getBasePositionAndOrientation(self.araknoId)

        #get observation 
        observation = self.get_observation()

        self.envStepCounter = 0
        #urdfRootPath=pybullet_data.getDataPath()
        #self.box = p.loadURDF("mug.urdf",self.endpoint)
        #pandaUid = p.loadURDF(os.path.join(urdfRootPath, "mug.urdf"),useFixedBase=True, basePosition = self.endpoint)
        print ("reset")
        
        return np.array(observation).astype(np.float32)

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