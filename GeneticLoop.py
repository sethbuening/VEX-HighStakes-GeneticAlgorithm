from deap import base
from deap import creator
from deap import tools
import random, pybullet_data, Simulation, pybullet, time

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Number of weights and biases in each AI
WEIGHTS = 272
BIASES = 26
PARAMETERS = WEIGHTS+BIASES
INITIAL_RANGE = 1

# Variables for evaluation loop
populationSize = 50
loadingSize = 50

# Selection variables
ELITEP = .02 #.02 # Fraction of top individuals that will proceed no matter what
TS = 3 # 3 for real thing # Tournament size
NOT = int(populationSize * (1-ELITEP)) # to keep the population size the same over time

# Probabilites for crossover and mutation
CXPB = 0.8
MUTPB = 0.2

def evaluate_individual(individual):
    weights = [individual[i] for i in range(WEIGHTS)]
    biases = [individual[i] for i in range(WEIGHTS, PARAMETERS)]
    return Simulation.runSimulation(weights, biases)

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
toolbox.register("mate", tools.cxUniform, indpb=0.5) # Probability for the genes to switch
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.075) # mean = 0, stdev = 0.15, 
                                                                            # chance for each value (not individual) to mutate = 0.075
# function to initialize population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Initialize and evaluate the first population

#Set up the simulation to be run
simId = pybullet.connect(pybullet.DIRECT)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setGravity(0, 0, -9.81)

population = toolbox.population(n=populationSize)
print("Evaluating " + str(len(population)) + " individuals:")
for ind in population:
    ind.fitness.values = (toolbox.evaluate(ind),)
    i = population.index(ind)+1
    x = int(((i/populationSize)*loadingSize))
    timeLeft = 1.25*(populationSize-i) # 1 second is the average time to run a single simulation
    print(" "*(loadingSize+43), end="\r") # clear the line to print again
    print(
        "Gen1:Loading|" + "#"*x + "-"*(loadingSize-x) + 
        "| eta: " + str(int(timeLeft/60)) + " minutes, " + str(int(timeLeft%60)) + " seconds", 
        end="\r"
        )
print("Gen1:Evaluated" + " "*(29+loadingSize))
# Determine the highest fitness value, initialize hall of fame
fitnesses = [ind.fitness.values[0] for ind in population]
bestFitness = max(fitnesses)
hallOfFame = [population[fitnesses.index(bestFitness)]]

# Begin the loop (selection, crossover, mutation, evaluate fitness)
# Loop runs until a specified fitness goal is reached
fitnessGoal = 100 # 100 is the best score possible
generationNum = 2
while (generationNum <= 30): #bestFitness < fitnessGoal
    # Selection
    offspring = toolbox.select(population)

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if(random.random() < CXPB):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    # Mutation
    for individual in offspring:
        if(random.random() < MUTPB):
            toolbox.mutate(individual)
            del individual.fitness.values

    # Evaluate fitness if the individual does not already have a fitness
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    print("Evaluating " + str(len(invalid_individuals)) + " individuals:")
    for ind in invalid_individuals:
        ind.fitness.values = (toolbox.evaluate(ind),)
        i = invalid_individuals.index(ind)+1
        x = int(((i/(len(invalid_individuals)))*loadingSize))
        timeLeft = 1.25*(populationSize-i)
        print(" "*(loadingSize+43), end="\r") # clear the line to print again
        print(
            "Gen" + str(generationNum) + ":Loading|" + "#"*x + "-"*(loadingSize-x) + 
            "| eta: " + str(int(timeLeft/60)) + " minutes, " + str(int(timeLeft%60)) + " seconds", 
            end="\r"
            )
    print("Gen" + str(generationNum) + ":Evaluated" + " "*(29+loadingSize))
    # The offspring entirely replaces the population
    population[:] = offspring
    # Update the best fitness
    fitnesses = [ind.fitness.values[0] for ind in population]
    bestFitness = max(fitnesses)
    hallOfFame.append(population[fitnesses.index(bestFitness)])
    generationNum += 1
print(hallOfFame[-1])
pybullet.disconnect(simId)