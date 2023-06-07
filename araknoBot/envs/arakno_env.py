import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import numpy as np
import pybullet_envs
import pybullet_data
import os
import math

#path_urdf = 'araknoBot/resources/urdfs/arakno.urdf'
#'araknoBot/resources/urdfs/goal.urdf'


class AraknoEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'] #human
    }

    def __init__(self, render = True): #True
        super(AraknoEnv, self).__init__()

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        p.setGravity(0, 0, -10)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath())  # used by loadURDF

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # m/s^2
        p.setTimeStep(0.01)   # sec

        #load models
        self.plane = p.loadURDF("plane.urdf")
        
        path_urdf = 'araknoBot/resources/urdfs/arakno.urdf'
        self.init_position = [0,0,0.15]
        self.init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.araknoId = p.loadURDF(path_urdf, self.init_position, self.init_orientation)

        # calculate the position of the endpoint 1 kilometer away from the start position
        direction = np.array([1, 0, 0])  # direction as a unit vector
        self.endpoint = self.init_position + 10 * direction

        self.init_orientation = p.getQuaternionFromEuler([0,0,0])
        self.goalId = p.loadURDF('araknoBot/resources/urdfs/goal.urdf', self.endpoint, self.init_orientation)

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

        self.movingJoints = [0,1,2,3,4,5,6,7,8,9,10,11]

        #store previous values
        self.xposprev = self.start_position[0]
        self.prev_distance_to_endpoint = math.sqrt((self.start_position[0] - self.endpoint[0]) ** 2 + (self.start_position[1] - self.endpoint[1]) ** 2)

        # Define the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32) #49 109
    
    def reset(self,seed=None, options=None):
        """
        Important: the observation must be a numpy array
        """
        super().reset(seed=seed, options=options)

        #reset the environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # m/s^2
        p.setTimeStep(0.01) 
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything

        self.vt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.vd = 0
        self.maxV = 5

        #load models
        self.plane = p.loadURDF("plane.urdf")
        #p.resetBasePositionAndOrientation(self.araknoId,self.init_position, self.init_orientation)

        self.araknoId = p.loadURDF('araknoBot/resources/urdfs/arakno.urdf', self.init_position, self.init_orientation)

        p.resetJointState(self.araknoId, 2, 1.0)
        p.resetJointState(self.araknoId, 5, 1.0)
        p.resetJointState(self.araknoId, 8, 1.0)
        p.resetJointState(self.araknoId, 11, 1.0)

        p.addUserDebugText('GOAL', [10, 0 ,0.4], [1, 0, 0])

        self.goalId = p.loadURDF('araknoBot/resources/urdfs/goal.urdf', self.endpoint, self.init_orientation)
        

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        #init of the prev state 
        self.prev_pos, self.prev_ori = p.getBasePositionAndOrientation(self.araknoId)

        #get observation 
        observation = self.compute_observation()
        
        return observation, {}

    def step(self, action):

        #store the previous x pos
        self.xposprev = self.compute_observation()[0]
        self.distance_to_endpoint = math.sqrt((self.compute_observation()[0] - self.endpoint[0]) ** 2 + (self.compute_observation()[1] - self.endpoint[1]) ** 2)

        self.assign_throttle(action)

        #simulate the action
        p.stepSimulation()

        observation = self.compute_observation()

        reward = self.comp_reward(action)

        done = self.check_done()
        truncated = False
        info = {}

        #return observation, reward, done, {}
        return (
            observation,
            reward,
            done,
            truncated,
            info,
        )
    
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
        baseOri = p.getBasePositionAndOrientation(self.araknoId)
        JointStates = p.getJointStates(self.araknoId, self.movingJoints)
        BaseAngVel = p.getBaseVelocity(self.araknoId)

        obs = np.array([
          baseOri[0][0],
          baseOri[0][1],
          baseOri[0][2],
          # orientation (quarternion x,y,z,w) of the Torso -> 4
          baseOri[1][0],
          baseOri[1][1],
          baseOri[1][2],
          baseOri[1][3],
          JointStates[0][0],
          JointStates[1][0],
          JointStates[2][0],
          JointStates[3][0],
          JointStates[4][0],
          JointStates[5][0],
          JointStates[6][0],
          JointStates[7][0],
          JointStates[8][0],
          JointStates[9][0],
          JointStates[10][0],
          JointStates[11][0],
          # 3-dim directional velocity and 3-dim angular velocity -> 3+3=6
          BaseAngVel[0][0],
          BaseAngVel[0][1],
          BaseAngVel[0][2],
          BaseAngVel[1][0],
          BaseAngVel[1][1],
          BaseAngVel[1][2],
          JointStates[0][1],
          JointStates[1][1],
          JointStates[2][1],
          JointStates[3][1],
          JointStates[4][1],
          JointStates[5][1],
          JointStates[6][1],
          JointStates[7][1],
          JointStates[8][1],
          JointStates[9][1],
          JointStates[10][1],
          JointStates[11][1]
          ])
        
        return np.array(obs, dtype=np.float32)
    
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
    
    def comp_reward(self,action):

        baseOri = p.getBasePositionAndOrientation(self.araknoId)

        com = self.calculate_center_of_mass(self.araknoId)
        # Compute the robot's stability
        roll_diff = abs(com[0] - baseOri[0][0])
        pitch_diff = abs(com[1] - baseOri[0][1])
        yaw_diff = abs(com[2] - baseOri[0][2])

        stability = 1.0 / (1.0 + roll_diff + pitch_diff + yaw_diff)

        #control if robot flipped
        # contact points of arakno with plane
        # list should have a length of self.num_of_legs (for each end link per leg) if spider is walking normally 
        contact_points = p.getContactPoints(self.araknoId, self.plane)
        contact_cost = 0  # 5 * 1e-1 * len(ContactPoints)
        # list of joint_ids of end links of for each leg
        end_joint_ids = [(4 * x + 3) for x in range(4)]
        # list to store contact points that are not end links
        end_contact_points_list = []
        # iterate over contact list
        for x in range(len(contact_points)):
            # append to end contact points list if joint id from an end link
            if contact_points[x][3] not in end_joint_ids:
                end_contact_points_list.append(contact_points[x])
        # checks if any links (other than end links) of the spiderbot touches plane, i.e "flipped" 
        if len(end_contact_points_list) != 0:
          contact_cost = 10
        
        #A negative reward for penalising the ant if it takes actions that are too large. 
        ctrl_cost = 0.5 * np.square(action).sum() 

        #reward based on position from along x-axis
        #frametime is 0.01 - making the default dt = 5 * 0.01 = 0.05
        # the time between actions and is dependent on the frame_skip parameter (default is 5)

        #move_forward = (self.xposprev - p.getBasePositionAndOrientation(self.araknoId)[0][0])/0.01
        #print("check ", move_forward)

        # Calculate the Euclidean distance between the agent and the endpoint
        distance_to_endpoint = math.sqrt((p.getBasePositionAndOrientation(self.araknoId)[0][0] - self.endpoint[0]) ** 2 + (p.getBasePositionAndOrientation(self.araknoId)[0][1] - self.endpoint[1]) ** 2)
        move_forward  = distance_to_endpoint - self.prev_distance_to_endpoint 
        #print("check ", move_forward)

        #based on velocity in x axis 
        forward_vel = 5 * p.getBaseVelocity(self.araknoId)[0][0] - 5

        # Penalize if the robot has fallen or is not alive
        alive = self.is_alive()
        if not alive:
            alive_penalty = -1.0
        else:
            alive_penalty = 1.0
    
        reward = alive_penalty + move_forward - contact_cost - ctrl_cost #+ stability  + forward_vel
        
        return reward
    
    def check_done(self):
        #check if the experiment is done by checking the following conditions:
        #1.reached the endpoint
        #2.fallen on the ground or jump to high
        #if flipped
        #3.maximum number timesteps reached timeout 
        done = False
        curr_pos, _ = p.getBasePositionAndOrientation(self.araknoId)
        if (not self.is_alive()):
            #print("You Dead")
            done = True
        if (self.endpoint[0] - curr_pos[0]) < 0.01 and (self.endpoint[1] - curr_pos[1]) < 0.01 :
            print("Arrived")
            done = True

        # Return whether the episode is done or not
        return done

    #check if the araknoBot have healthy behaviors
    def is_alive(self):
        # Get the position and orientation of the robot's base
        pos, orn = p.getBasePositionAndOrientation(self.araknoId)
        #The ant is considered healthy if the z-coordinate of the torso is in this range [0.08, 1.0]
        if pos[2] > 0.07 and pos[2]< 2.2:
            return True
        else:
            return False
    

    def render(self, mode='human', close = False):
        pass

    def close(self):
        p.disconnect()
        #pass