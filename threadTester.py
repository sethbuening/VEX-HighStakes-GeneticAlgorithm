import pybullet as p
import pybullet_data
import math # For current scoring
import NeuralNetwork
import torch
import random
import concurrent.futures

def evaluate_individual(individual, index):
    weights = [individual[i] for i in range(16*8 + 8*16 + 2*8)]
    biases = [individual[i] for i in range(16*8+8*16+2*8, 16*8+8*16+2*8 + 16+8+2)]
    score = runSimulation(torch.tensor(weights, dtype=torch.float32), torch.tensor(biases, dtype=torch.float32))
    return score, index

def runSimulation(weights, biases):
    p.connect(p.DIRECT) # or p.GUI for graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #gains access to the basic URDF files from pybullet
    p.setGravity(0, 0, -9.81)

    # Load URDF files
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,0.01]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    robotId = p.loadURDF("husky/husky.urdf",startPos, startOrientation)
    #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    startOrientation2 = p.getQuaternionFromEuler([90,0,0])
    boxes = [
        p.loadURDF("urdf/box.urdf",[3,1,0.3],startOrientation2),
        p.loadURDF("urdf/box.urdf",[5,0,0.3],startOrientation2),
        p.loadURDF("urdf/box.urdf",[4,3.5,0.3],startOrientation2)
    ]

    # Robot control
    maxForce = 100  # in Newtons
    instance = NeuralNetwork.NeuralNetwork(weights, biases)
    # JOINTS:
    #2,4 left
    #3,5 right
    for _ in range (14400):# Simulate for 1 minute
        # Gather data (gets x and y positions of the boxes and the robot)
        input = []
        for obj in boxes + [robotId]:
            position, _ = p.getBasePositionAndOrientation(obj)
            input.extend([position[0], position[1]])

        # Calculate motor velocity with neural network
        targetVelLeft, targetVelRight = instance.forward(torch.tensor(input))
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

    p.disconnect()
    return score

# Randomly initialize 10 individuals
invalid_individuals = [[random.uniform(-1, 1) for _ in range(298)] for _ in range(10)]

# Begin threading setup
num_threads = 5
fitnesses = [None] * len(invalid_individuals)

# Begin parallel processing. Uses ThreadPoolExecutor to do so.
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(evaluate_individual, ind, idx): idx for idx, ind in enumerate(invalid_individuals)}
    
    for future in concurrent.futures.as_completed(futures):
        fitness, idx = future.result()  # Get index and fitness from result
        fitnesses[idx] = fitness  # Store result in the correct position

print(fitnesses)