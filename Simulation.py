import pybullet as p
import math, torch, os
import NeuralNetwork

def suppress_output(func, *args, **kwargs):
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

def runSimulation(weights, biases):
    p.resetSimulation()

    # Load URDF files
    _ = p.loadURDF("plane.urdf")
    startPos = [0,0,0.01]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    robotId = suppress_output(p.loadURDF, "husky/husky.urdf", startPos, startOrientation)
    #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    startOrientation2 = p.getQuaternionFromEuler([90,0,0])
    boxes = [
        p.loadURDF("urdf/box.urdf",[3,1,0.3],startOrientation2),
        p.loadURDF("urdf/box.urdf",[5,0,0.3],startOrientation2),
        p.loadURDF("urdf/box.urdf",[4,3.5,0.3],startOrientation2)
        ]

    # Robot control
    targetVelLeft = 0 #rad/s
    targetVelRight = 0 #rad/s
    maxForce = 10  #Newton
    network = NeuralNetwork.NeuralNetwork(weights, biases) # import weights and biases from the genetic loop file
    # JOINTS:
    #2,4 left
    #3,5 right
    for i in range (14400):# 14400 is 1 minute
        # Gather data (gets x and y positions of the boxes and the robot)
        input = []
        for obj in boxes + [robotId]:
            position, _ = p.getBasePositionAndOrientation(obj)
            input.extend([position[0], position[1]])
        # Calculate motor velocity with neural network
        targetVelLeft,targetVelRight = network.forward(torch.tensor(input).unsqueeze(0))
        # Move the motors
        p.setJointMotorControl(robotId, 2, p.VELOCITY_CONTROL, targetVelLeft, maxForce)
        p.setJointMotorControl(robotId, 3, p.VELOCITY_CONTROL, targetVelRight, maxForce)
        p.setJointMotorControl(robotId, 4, p.VELOCITY_CONTROL, targetVelLeft, maxForce)
        p.setJointMotorControl(robotId, 5, p.VELOCITY_CONTROL, targetVelRight, maxForce)
        p.stepSimulation()

    #Scoring: 100 is the best score possible
    score = 100
    for box in boxes:
        position, _ = p.getBasePositionAndOrientation(box)
        score -= math.fabs(10-position[1])
    return score