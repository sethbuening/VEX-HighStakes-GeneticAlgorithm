import pybullet as p
#import time # For visualization
import pybullet_data, math, torch, os
import NeuralNetwork

def suppress_output(func, *args, **kwargs):
    """
    Suppresses all output (stdout and stderr) during the execution of the given function.
    This includes Python and lower-level C library outputs.
    """
    # Redirect low-level stdout and stderr
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(1)  # Save the original stdout
        old_stderr = os.dup(2)  # Save the original stderr
        os.dup2(devnull.fileno(), 1)  # Redirect stdout to /dev/null
        os.dup2(devnull.fileno(), 2)  # Redirect stderr to /dev/null
        try:
            # Call the function while output is suppressed
            return func(*args, **kwargs)
        finally:
            # Restore original stdout and stderr
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
suppress_output(p.connect, p.GUI)#  p.GUI (graphical) or p.DIRECT (non-graphical)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally for the basic URDF files
p.setGravity(0, 0, -9.81)

# Load URDF files
_ = p.loadURDF("plane.urdf")
startPos = [0,0,0.01]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = suppress_output(p.loadURDF, "husky/husky.urdf", startPos, startOrientation) #change this to robot when ready
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
startOrientation2 = p.getQuaternionFromEuler([90,0,0])
box1 = p.loadURDF("urdf/box.urdf",[3,1,0.3],startOrientation2)
box2 = p.loadURDF("urdf/box.urdf",[5,0,0.3],startOrientation2)
box3 = p.loadURDF("urdf/box.urdf",[4,3.5,0.3],startOrientation2)
boxes = [box1, box2, box3]

# Robot control
targetVelLeft = 0 #rad/s
targetVelRight = 0 #rad/s
maxForce = 100  #Newton
weights = []
biases = []
instance = NeuralNetwork.NeuralNetwork(torch.randn(272, dtype=torch.float32), torch.randn(26, dtype=torch.float32)) # import weights and biases from the genetic loop file
# JOINTS:
#2,4 left
#3,5 right
p.setTimeStep(0.01)
for i in range (14400):# 14400 is 1 minute
    # Gather data (gets x and y positions of the boxes and the robot)
    dataObjects = boxes
    dataObjects.append(robotId)
    input = []
    for i in range(4):
        position,orientation = p.getBasePositionAndOrientation(dataObjects[i])
        input.append(position[0])
        input.append(position[1])
    # Calculate motor velocity with neural network
    targetVelLeft,targetVelRight = instance.forward(torch.tensor(input))
    # Move the motors
    p.setJointMotorControl(robotId, 2, p.VELOCITY_CONTROL, targetVelLeft, maxForce)
    p.setJointMotorControl(robotId, 3, p.VELOCITY_CONTROL, targetVelRight, maxForce)
    p.setJointMotorControl(robotId, 4, p.VELOCITY_CONTROL, targetVelLeft, maxForce)
    p.setJointMotorControl(robotId, 5, p.VELOCITY_CONTROL, targetVelRight, maxForce)
    p.stepSimulation()
    #time.sleep(1./240.)#remove this for a not real-time simulation


#Scoring: 100 is the best score possible
score = 100
for i in range(3):
    position,orientation = p.getBasePositionAndOrientation(boxes[i])
    score -= math.fabs(10-position[1])
print("\nSCORE IS: " + str(score))
#cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
#print(cubePos,cubeOrn)
suppress_output(p.disconnect)