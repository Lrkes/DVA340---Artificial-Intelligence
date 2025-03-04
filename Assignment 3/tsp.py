import pandas as pd
import random
from scipy.spatial.distance import euclidean

# --- PARAMETERS ---
FILE_PATH = "./Assignment 3 berlin52.tsp"
POPULATION_SIZE = 100  # Number of initial routes
N_CITIES = 52  # Total number of cities in Berlin52
TOURNAMENT_SIZE = 5  # How many routes to compare in selection


def load_tsp_file(file_path):
    """Reads and parses the TSP file to extract city coordinates."""
    locations = []
    with open(file_path, "r") as file:
        lines = file.readlines()

    start_index = lines.index("NODE_COORD_SECTION\n") + 1
    for line in lines[start_index:]:
        if line.strip() == "EOF":
            break
        parts = line.split()
        location_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
        locations.append({"ID": location_id, "X": x, "Y": y})

    return pd.DataFrame(locations)


def compute_distance_matrix(df):
    """Computes the Euclidean distance matrix for all city pairs."""
    locations = df[['X', 'Y']].values
    size = len(locations)
    return {(i, j): euclidean(locations[i], locations[j]) for i in range(size) for j in range(size)}


def generate_random_route(size):
    """Generates a random valid TSP route starting and ending at city 1 (index 0)."""
    route = list(range(1, size))  # Exclude city 1 from shuffle
    random.shuffle(route)
    return [0] + route + [0]  # Ensure start & end at city 1


def route_distance(route, distance_matrix):
    """Computes the total distance of a given TSP route."""
    return sum(distance_matrix[(route[i], route[i+1])] for i in range(len(route) - 1))


def initialize_population(pop_size, n_cities):
    """Creates an initial population of random routes."""
    return [generate_random_route(n_cities) for _ in range(pop_size)]


def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE): # TODO: Look again
    """Selects the best route from k randomly chosen routes."""
    selected = random.sample(list(zip(population, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]  # shortest


def ordered_crossover(parent1, parent2):
    """Performs Ordered Crossover (OX1) on two parent routes."""
    start, end = sorted(random.sample(range(1, len(parent1) - 1), 2))
    print(f"Selected segment: {start} to {end}")  # Debugging

    child1 = [None] * len(parent1)
    child2 = [None] * len(parent2)

    child1[0] = 0
    child1[-1] = 0
    child2[0] = 0
    child2[-1] = 0

    # Copy the segments between start and end
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    # print(f"Child1 (Segment copied): {child1}")
    # print(f"Child2 (Segment copied): {child2}")

    # Fill in the remaining elements
    fill_in1 = [city for city in parent2 if city not in child1]
    fill_in2 = [city for city in parent1 if city not in child2]

    for i in range(len(parent1)):
        if child1[i] is None:
            child1[i] = fill_in1.pop(0)
        if child2[i] is None:
            child2[i] = fill_in2.pop(0)

    return child1, child2


def mutate(route, mutation_rate=0.1):
    """Mutates a TSP route by swapping two cities with a given probability."""
    if random.random() < mutation_rate:
        city1, city2 = random.sample(range(1, len(route) - 1), 2)
        route[city1], route[city2] = route[city2], route[city1]
    return route


def genetic_algorithm(max_fitness_calculations=250000):
    """Runs the genetic algorithm until we reach the max fitness calculations."""

    # Step 1: Initialize Population
    population = initialize_population(POPULATION_SIZE, N_CITIES)

    # Step 2: Compute Initial Fitness Scores
    fitness_scores = [route_distance(route, distance_matrix) for route in population]


berlin52_df = load_tsp_file(FILE_PATH)
distance_matrix = compute_distance_matrix(berlin52_df)

# ----------------------------------------------------------
# 🚀 Key Insights About Crossover in Genetic Algorithms (TSP)
# ✅ 1. Crossover Mixes Good Traits
#
# We select two good parents from the population.
# We assume their city sequences contain good patterns.
# By mixing parts of both, we hope to create a better offspring.
# ✅ 2. Crossover is Random—It Can Go Wrong
#
# Sometimes, we accidentally combine bad parts instead of good ones.
# In the early generations, many offspring won't be better than their parents.
# That’s okay! Not every child needs to be better—just a few!
# ✅ 3. Why Crossover Works in the Long Run
#
# Many crossovers happen, so some lucky children inherit great routes.
# Selection ensures only the best offspring survive.
# Mutation adds diversity, preventing the algorithm from getting stuck in bad solutions.
# ✅ 4. The Balance Between Exploration & Exploitation
#
# Exploitation = Keeping and refining the best solutions.
# Exploration = Allowing new possibilities (mutation helps here).
# A good GA balances both to find an optimal route over generations.
