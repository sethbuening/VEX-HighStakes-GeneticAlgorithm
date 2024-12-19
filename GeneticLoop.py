from deap import base
from deap import creator
from deap import tools
import random, pybullet_data, Simulation, pybullet, statistics, time

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Number of weights and biases in each AI
WEIGHTS = 10+50+10 #previously 272
BIASES = 10+5+2 #previously 26
PARAMETERS = WEIGHTS+BIASES
INITIAL_RANGE = 0.5

# Variables for evaluation loop
populationSize = 100
loadingSize = 50

# Selection variables
ELITEP = .04 #.02 # Fraction of top individuals that will proceed no matter what
TS = 3 # 3 for real thing # Tournament size
NOT = int(populationSize * (1-ELITEP)) # to keep the population size the same over time

# Probabilites for crossover and mutation
CXPB = 0.85
MUTPB = 0.25

def evaluate_individual(individual):
    weights = [individual[i] for i in range(WEIGHTS)]
    biases = [individual[i] for i in range(WEIGHTS, PARAMETERS)]
    return Simulation.runSimulation(weights, biases, robotId, boxes)

# Multiple of the same individual may end up selected, but this is ok because they will all crossover and mutate
def select_individuals(population):
    selected = []
    # Elitism
    selected.extend([toolbox.clone(ind) for ind in tools.selBest(population, int(len(population)*ELITEP))])
    # Tournament Selection
    selected.extend([toolbox.clone(ind) for ind in tools.selTournament(population, NOT, TS)])
    return selected

toolbox = base.Toolbox()

# Function for creating random numbers for the initial weights and biases
toolbox.register("randomNum", random.uniform, -INITIAL_RANGE, INITIAL_RANGE)
# Creates individual() which initializes an individual with n random float values inside of its list
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.randomNum, n=PARAMETERS)

# Creates the 4 core genetic algorithm functions and allows access to them from the toolbox
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", select_individuals)
toolbox.register("mate", tools.cxUniform, indpb=0.5) # Used to use indpb=0.5, cxUniform. indpb = Probability for the genes to switch
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.25, indpb=0.2) # mean = 0, stdev = 0.15, 
                                                                            # chance for each value (not individual) to mutate = 0.075
# function to initialize population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Initialize and evaluate the first population

#Set up the simulation to be run
simId = pybullet.connect(pybullet.DIRECT)
pybullet.setGravity(0, 0, -9.82)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
# Load URDF files
_ = pybullet.loadURDF("plane.urdf")
startPos = [0,0,0.02]
startOrientation = pybullet.getQuaternionFromEuler([0,0,0])
robotId = pybullet.loadURDF("husky/husky.urdf", startPos, startOrientation)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
startOrientation2 = pybullet.getQuaternionFromEuler([90,0,0])
boxes = [
    pybullet.loadURDF("urdf/box.urdf",[3,1,0.3],startOrientation2),
    pybullet.loadURDF("urdf/box.urdf",[5,0,0.3],startOrientation2),
    pybullet.loadURDF("urdf/box.urdf",[4,3.5,0.3],startOrientation2)
    ]
pybullet.setTimeStep(1/240)

population = toolbox.population(n=populationSize)
for ind in population:
    ind.fitness.values = (toolbox.evaluate(ind),)
    i = population.index(ind)+1
    x = int(((i/populationSize)*loadingSize))
    timeLeft = 1*(populationSize-i) # 1 second is the average time to run a single simulation
    print(" "*(loadingSize+50), end="\r") # clear the line to print again
    print(
        f"Gen1:{str(len(population))}Loading|" + "#"*x + "-"*(loadingSize-x) + 
        f"| eta: {str(int(timeLeft/60))} minutes, {str(int(timeLeft%60))} seconds", 
        end="\r"
        )
print(f"Gen1:{str(len(population))}Evaluated" + " "*(35+loadingSize))
# Determine the highest fitness value, initialize hall of fame
fitnesses = [ind.fitness.values[0] for ind in population]
stdev = statistics.stdev(fitnesses)
bestFitness = max(fitnesses)
print(f"Gen1:BestFitness: {str(bestFitness)} |stdev: {str(stdev)}")
hallOfFame = [population[fitnesses.index(bestFitness)]]

# Begin the loop (selection, crossover, mutation, evaluate fitness)
# Loop runs until a specified fitness goal is reached
fitnessGoal = 10 # 1000 is the best score possible
generationNum = 2
while (generationNum <= 25): #bestFitness < fitnessGoal
    # Selection
    offspring = toolbox.select(population)
    repeatedBackup = []

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if(random.random() < CXPB):
            repeatedBackup.extend([child1, child2])
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            
    # Mutation
    for individual in offspring:
        if(random.random() < MUTPB):
            repeatedBackup.append(individual)
            toolbox.mutate(individual)
            del individual.fitness.values

    # Evaluate fitness if the individual does not already have a fitness
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    for i in range(len(invalid_individuals)):
        invalid_individuals[i].fitness.values = (toolbox.evaluate(invalid_individuals[i]),)
        x = int(((i+1)/(len(invalid_individuals)))*loadingSize)
        timeLeft = 1*(len(invalid_individuals)-(i+1))
        print(" "*(loadingSize+50), end="\r") # clear the line to print again
        print(
            f"Gen{str(generationNum)}:{str(len(invalid_individuals))}Loading|" + "#"*x + "-"*(loadingSize-x) + 
            f"| eta: {str(int(timeLeft/60))} minutes, {str(int(timeLeft%60))} seconds", 
            end="\r"
            )
    print(f"Gen{str(generationNum)}:{str(len(invalid_individuals))}Evaluated" + " "*(35+loadingSize))
    
    # Update fitnesses
    fitnesses[:] = [ind.fitness.values[0] for ind in offspring]

    # Update fitness again and print off statistics 
    fitnesses = [ind.fitness.values[0] for ind in population]
    stdev = statistics.stdev(fitnesses)
    bestFitness = max(fitnesses)
    print("Gen" + str(generationNum) + ":BestFitness: " + str(bestFitness) + " | stdev: " + str(stdev))

    # The offspring entirely replaces the population
    population[:] = offspring
    hallOfFame.append(population[fitnesses.index(bestFitness)])
    generationNum += 1
print(hallOfFame[-1])
# TODO: Write the entire hall of fame into a separate file as it processes
pybullet.disconnect()