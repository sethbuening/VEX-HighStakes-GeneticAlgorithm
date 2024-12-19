import pybullet as p
#import time # For visualization
import pybullet_data, torch, time, random
import NeuralNetwork


WEIGHTS = 10+50+10
BIASES = 10+5+2
PARAMETERS = WEIGHTS+BIASES

individual = [-0.3272738957012836, 0.30788870449625755, 0.3279581164315438, 0.21126351987453965, -0.64203133770425, -0.16003364314089008, -0.6105643846608878, -0.07821253724573929, 0.16753771066560763, -0.1932462481048598, 0.4266305747898521, 0.10293589102361743, -0.37775806032732784, 0.0914666537763712, 0.7470950450313814, 0.6354465869994764, 0.3536225249620519, 0.050092356536623917, 0.2306144944736554, 0.43179177089770115, 0.2842375934035385, -0.24939172328772252, -0.8315221588152004, 0.02612652444801944, 0.3783040878397731, 0.16955664595605457, -0.07460244062738554, 0.7465852603202756, 0.46766995872676986, 0.1701760266618878, -0.039095145723343355, -0.4445732630433482, -0.6421084523868673, -0.12804046588494525, -0.8115766919312684, -0.155311650727326, -0.5969773067192821, -0.2906833678069004, -0.4618309338016633, -1.0390669455454669, -0.13232551131979436, -0.09869472905261317, 0.04688903201377753, 0.3109579565643192, -0.27083755115313946, 0.0471331598699013, -0.2408572926498579, 0.3231698769582791, -0.23923436382244467, 0.2269941923861223, 0.45132789201432855, -0.19618425011116866, -0.010228729926293284, -0.5505429468932412, -0.04551454420012142, 0.01347982176254689, -0.1012319435301382, -0.6638236082456934, -0.3597036484788465, -0.47318897779599667, -0.36659054454531725, 0.9965476523340362, 0.3693950895773901, -0.09921009858503599, -0.6516788391050536, -0.526587314535459, 1.707979275437218, 0.03289198903043522, -0.05963161193170963, -0.33072250497235245, -0.043694408416602284, -0.07107459519703212, -1.0138621545584914, -0.3678793709883731, 0.309809987745263, -0.6028028877497467, 0.45837385969455546, 0.7813185970287005, -0.5514755012816631, 0.056065694710664804, -0.659902563243064, 0.44202410274087744, -0.09979978757044566, -0.2707154670442725, -0.08458032689212791, 0.2915120419310185, 0.4241046556593471]
weights = [individual[i] for i in range(WEIGHTS)] #torch.randn(272, dtype=torch.float32)
biases = [individual[i] for i in range(WEIGHTS, PARAMETERS)] # torch.randn(26, dtype=torch.float32)

simId = p.connect(p.GUI)
p.setGravity(0, 0, -9.82)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# Load URDF files
_ = p.loadURDF("plane.urdf")
startPos = [0,0,0.01]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("husky/husky.urdf", startPos, startOrientation)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
startOrientation2 = p.getQuaternionFromEuler([90,0,0])
boxes = [
    p.loadURDF("urdf/box.urdf",[3,1,0.3],startOrientation2),
    p.loadURDF("urdf/box.urdf",[5,-2,0.3],startOrientation2),
    p.loadURDF("urdf/box.urdf",[4,3.5,0.3],startOrientation2)
    ]

# Robot control
velLeft = 0 #rad/s
velRight = 0 #rad/s
maxForce = 0  #Newton
maxVel = 10
network = NeuralNetwork.NeuralNetwork(weights, biases) # import weights and biases from the genetic loop file
# JOINTS:
#2,4 left
#3,5 right
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

    # Move the motors
    # 1: forward
    # 2: backward
    # 3: turn left
    # 4: turn right
    # 0: stop
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
    time.sleep(1/240)
    '''for box in boxes:
        position, _ = p.getBasePositionAndOrientation(box)
        score -= (10/240)*abs(10-position[1])'''
print("Score is: " + str(score))
p.disconnect()