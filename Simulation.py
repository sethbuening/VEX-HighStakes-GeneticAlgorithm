import pybullet as p
import torch, random, time
import NeuralNetwork

def runSimulation(weights, biases, robotId, boxes):
    # Robot control
    velLeft = 0 #rad/s
    velRight = 0 #rad/s
    maxForce = 0  #Newton
    maxVel = 10
    network = NeuralNetwork.NeuralNetwork(weights, biases) # import weights and biases from the genetic loop file
    score = 0
    for i in range (14400):# 14400 is 1 minute
        # Gather data (gets x and y positions of the boxes and the robot)
        #position, orientation = p.getBasePositionAndOrientation(robotId)
        #input = [position[1], p.getEulerFromQuaternion(orientation)[2]]
        '''for obj in boxes + [robotId]:
            position, _ = p.getBasePositionAndOrientation(obj)
            input.extend([position[0], position[1]])'''
        # Calculate motor velocity with neural network
        velLeft, velRight = network.forward(torch.tensor(i/240).unsqueeze(0))
        # Joints: 2,4 left
        #         3,5 right

        p.setJointMotorControl(robotId, 2, p.VELOCITY_CONTROL, velLeft*maxVel)
        p.setJointMotorControl(robotId, 3, p.VELOCITY_CONTROL, velRight*maxVel)
        p.setJointMotorControl(robotId, 4, p.VELOCITY_CONTROL, velLeft*maxVel)
        p.setJointMotorControl(robotId, 5, p.VELOCITY_CONTROL, velRight*maxVel)
        p.stepSimulation()
        # Scoring: 0 is the best score possible
        #boxPosition, _ = p.getBasePositionAndOrientation(boxes[2])
        robotPosition, _ = p.getBasePositionAndOrientation(robotId)
        #score += (1/960)*math.sqrt(math.pow((abs(boxPosition[0]-robotPosition[0])), 2)+math.pow(abs(boxPosition[1]-robotPosition[1]), 2)) # the last value is the robot
        score -= (1/960)*(abs(10-robotPosition[0])) # the last value is the robot
        '''for box in boxes:
            position, _ = p.getBasePositionAndOrientation(box)
            score -= (10/240)*abs(10-position[1])'''
    p.resetBasePositionAndOrientation(robotId, [0,0,0.02], [0, 0, 0, 1])
    for id in boxes:
        p.resetBasePositionAndOrientation(id, [random.uniform(-5, 5), random.uniform(-5, 5), 0.3], [90, 0, 0, 1])
    return score