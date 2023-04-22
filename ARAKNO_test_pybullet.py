import os
#import pybullet env
import pybullet as p
#library contains all available objects and robots
#provide many example of URDF files
import pybullet_data
import numpy as np
import math
import time

#start the simulation graphical user interface
client = p.connect(p.GUI)
#set gravity force, by default is not enable
#p.setGravity(0, 0, -10) why 10?
p.setGravity(0, 0, -9.81, physicsClientId=client)

#load a plane to the scene
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")

#using my own URDF
path_urdf = 'arakno/resources/urdfs/arakno.urdf'
#specify the spawn pose
init_position = [0,0,0.5]
init_orientation = p.getQuaternionFromEuler([0,0,0])
araknoId = p.loadURDF(path_urdf, init_position, init_orientation)
#getNumJoints returns an integer value representing the number of joints
num_joints = p.getNumJoints(araknoId)

#debug interface for joints
#addUserDebugParameter lets you add custom sliders and buttons to tune parameters
BL_lowerleg_joint = p.addUserDebugParameter('BL_lowerleg_joint', -np.pi, np.pi, 0)
BL_shoulder_joint = p.addUserDebugParameter('BL_shoulder_joint', -np.pi, np.pi, 0)
BL_upperleg_joint = p.addUserDebugParameter('BL_upperleg_joint', -np.pi, np.pi, 0)

BR_lowerleg_joint = p.addUserDebugParameter('BR_lowerleg_joint', -np.pi, np.pi, 0)
BR_shoulder_joint = p.addUserDebugParameter('BR_shoulder_joint', -np.pi, np.pi, 0)
BR_upperleg_joint = p.addUserDebugParameter('BR_upperleg_joint', -np.pi, np.pi, 0)

FL_lowerleg_joint = p.addUserDebugParameter('FL_lowerleg_joint', -np.pi, np.pi, 0)
FL_shoulder_joint = p.addUserDebugParameter('FL_shoulder_joint', -np.pi, np.pi, 0)
FL_upperleg_joint = p.addUserDebugParameter('FL_upperleg_joint', -np.pi, np.pi, 0)

FR_lowerleg_joint = p.addUserDebugParameter('FR_lowerleg_joint', -np.pi, np.pi, 0)
FR_shoulder_joint = p.addUserDebugParameter('FR_shoulder_joint', -np.pi, np.pi, 0)
FR_upperleg_joint = p.addUserDebugParameter('FR_upperleg_joint', -np.pi, np.pi, 0)

#camera_joint = p.addUserDebugParameter('camera_joint', -np.pi, np.pi, 0)

#get information about position of the joints in the vector
for joint in range(num_joints):
    #INFO given by getJointInfo:
    #jointIndex, jointName, jointType, qIndex, uIndex, flags, jointDamping, jointFriction, jointLowerLimit, jointUpperLimit
    #jointMaxForce, jointMaxVelocity, linkName, jointAxis, parentFramePos, parentFrameOrn, parentIndex
    info = p.getJointInfo(araknoId, joint)
    print("INFO: ", info[0], ": ", info[1], " joint type: ", info[2])
    print("---------------------------------------------------")
    info = p.getDynamicsInfo(araknoId, joint)
    print("mass: ", info[0], ": ", info[2], " inertia diagonal: ", info[2])
    print("---------------------------------------------------")

#set the camera angle to another view(close view)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,\
    cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

#infinite loop simulation, each time step is 1/240 of a second
while True:
    #smooth simulation rendering
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

    #This lets you read the value of the parameter
    user_BR_shoulder = p.readUserDebugParameter(BR_shoulder_joint)
    user_BL_shoulder = p.readUserDebugParameter(BL_shoulder_joint)
    user_FR_shoulder = p.readUserDebugParameter(FR_shoulder_joint)
    user_FL_shoulder = p.readUserDebugParameter(FL_shoulder_joint)

    user_BR_upperleg = p.readUserDebugParameter(BR_upperleg_joint)
    user_BL_upperleg = p.readUserDebugParameter(BL_upperleg_joint)
    user_FR_upperleg = p.readUserDebugParameter(FR_upperleg_joint)
    user_FL_upperleg = p.readUserDebugParameter(FL_upperleg_joint)

    user_BR_lowerleg = p.readUserDebugParameter(BR_lowerleg_joint)
    user_BL_lowerleg = p.readUserDebugParameter(BL_lowerleg_joint)
    user_FR_lowerleg = p.readUserDebugParameter(FR_lowerleg_joint)
    user_FL_lowerleg = p.readUserDebugParameter(FL_lowerleg_joint)

    #user_camera = p.readUserDebugParameter(camera_joint)

    #reports the current position and orientation of the base (or root link) of the body in Cartesian world coordinates
    #orientation is a quaternion
    pos, ore = p.getBasePositionAndOrientation(araknoId)
    #print("pos: ", pos ," orient: ", ore)

    #We can control a robot by setting a desired control mode for one or more joint motors
    #the actual implementation of the joint motor controller is as a constraint for POSITION_CONTROL and VELOCITY_CONTROL
    #required params -> p.setJointMotorControl2(objUid, jointIndex,controlMode=mode, force=maxForce)

    #velocity control
    #p.setJointMotorControl2(araknoId, 0, p.VELOCITY_CONTROL,targetVelocity = user_BL_shoulder , force = 1000)

    #position control
    p.setJointMotorControl2(araknoId,0, p.POSITION_CONTROL, targetPosition = user_BL_shoulder)
    p.setJointMotorControl2(araknoId,1, p.POSITION_CONTROL, targetPosition = user_BL_upperleg)
    p.setJointMotorControl2(araknoId,2, p.POSITION_CONTROL, targetPosition = user_BL_lowerleg)

    p.setJointMotorControl2(araknoId,3, p.POSITION_CONTROL, targetPosition = user_BR_shoulder)
    p.setJointMotorControl2(araknoId,4, p.POSITION_CONTROL, targetPosition = user_BR_upperleg)
    p.setJointMotorControl2(araknoId,5, p.POSITION_CONTROL, targetPosition = user_BR_lowerleg)

    p.setJointMotorControl2(araknoId,6, p.POSITION_CONTROL, targetPosition = user_FL_shoulder)
    p.setJointMotorControl2(araknoId,7, p.POSITION_CONTROL, targetPosition = user_FL_upperleg)
    p.setJointMotorControl2(araknoId,8, p.POSITION_CONTROL, targetPosition = user_FL_lowerleg)

    p.setJointMotorControl2(araknoId,9, p.POSITION_CONTROL, targetPosition = user_FR_shoulder)
    p.setJointMotorControl2(araknoId,10, p.POSITION_CONTROL, targetPosition = user_FR_upperleg)
    p.setJointMotorControl2(araknoId,11, p.POSITION_CONTROL, targetPosition = user_FR_lowerleg)

    #p.setJointMotorControl2(araknoId,12, p.POSITION_CONTROL, targetPosition = user_camera)
     
    #runs one step of the simulation
    p.stepSimulation()
    
p.disconnect()