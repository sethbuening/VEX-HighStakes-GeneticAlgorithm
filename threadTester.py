import pybullet as p
import pybullet_data, math, NeuralNetwork, torch, random, concurrent.futures, os

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
def evaluate_individual(individual, index):
    weights = [individual[i] for i in range(16*8 + 8*16 + 2*8)]
    biases = [individual[i] for i in range(16*8+8*16+2*8, 16*8+8*16+2*8 + 16+8+2)]
    score = runSimulation(weights, biases)
    return score, index

def runSimulation(weights, biases):
    simulationId = p.connect(p.DIRECT) # or p.GUI for graphical version
    print(simulationId)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #gains access to the basic URDF files from pybullet
    p.setGravity(0, 0, -9.81)

    # Load URDF files
    planeId = p.loadURDF("plane.urdf")
    startPos = [0,0,0.01]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    robotId=p.loadURDF("husky/husky.urdf", startPos, startOrientation)
    #robotId = suppress_output(p.loadURDF, "husky/husky.urdf", startPos, startOrientation)
    #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    startOrientation2 = p.getQuaternionFromEuler([90,0,0])
    boxes = [
        p.loadURDF("urdf/box.urdf",[3,1,0.3],startOrientation2),
        p.loadURDF("urdf/box.urdf",[5,0,0.3],startOrientation2),
        p.loadURDF("urdf/box.urdf",[4,3.5,0.3],startOrientation2)
    ]

    # Robot control
    maxForce = 10  # in Newtons
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
        targetVelLeft, targetVelRight = instance.forward(torch.tensor(input).unsqueeze(0))
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
invalid_individuals = [[random.uniform(-1, 1) for _ in range(298)] for _ in range(100)]

# Begin threading setup
num_threads = 100
fitnesses = [None] * len(invalid_individuals)
count = 1

# Begin parallel processing. Uses ThreadPoolExecutor to do so.
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(evaluate_individual, ind, idx): idx for idx, ind in enumerate(invalid_individuals)}
    print("Threads started!")
    for future in concurrent.futures.as_completed(futures):
        fitness, idx = future.result()  # Get index and fitness from result
        fitnesses[idx] = fitness  # Store result in the correct position
        print("Evaluated a simulation!: " + count)

print(fitnesses)