import pandas as pd
import random
from scipy.spatial.distance import euclidean

# --- PARAMETERS ---
FILE_PATH = "./Assignment 3 berlin52.tsp"
POPULATION_SIZE = 400  # Number of initial routes
N_CITIES = 52  # Total number of cities in Berlin52
TOURNAMENT_SIZE = 3  # How many routes to compare in selection
MUTATION_RATE = 0.2  # Probability of mutation

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
    """Computes Euclidean distance for all city pairs."""
    locations = df[['X', 'Y']].values
    size = len(locations)
    return {(i, j): euclidean(locations[i], locations[j]) for i in range(size) for j in range(size)}


def generate_random_route(size):
    """Generates random routes starting from and ending at [0]."""
    return [0] + random.sample(range(1, size), size - 1) + [0]


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


def mutate(route):
    """Mutates a TSP route by swapping two pairs of cities or reversing a segment."""
    if random.random() < MUTATION_RATE:
        if random.random() < 0.5:  # 50% chance to swap two pairs
            city1, city2 = random.sample(range(1, len(route) - 1), 2)
            route[city1], route[city2] = route[city2], route[city1]
            city3, city4 = random.sample(range(1, len(route) - 1), 2)
            route[city3], route[city4] = route[city4], route[city3]
        else:  # 50% chance to reverse a segment
            start, end = sorted(random.sample(range(1, len(route) - 1), 2))
            route[start:end] = reversed(route[start:end])
    return route


def genetic_algorithm(max_fitness_calculations=250000):
    """Runs the genetic algorithm until we reach the max fitness calculations."""

    # Step 1: Initialize Population
    population = initialize_population(POPULATION_SIZE, N_CITIES)

    # Step 2: Compute Initial Fitness Scores
    fitness_scores = [route_distance(route, distance_matrix) for route in population]

    # Step 3: Track Fitness Calculations & Best Route
    fitness_calculations = len(population)  # We already evaluated all initial routes
    best_route = None
    best_distance = float("inf")

    # Step 4: Start Evolution Loop
    while fitness_calculations < max_fitness_calculations:
        # Selection Step
        num_selected = POPULATION_SIZE // 2  # Select top 50%
        selected_population = [tournament_selection(population, fitness_scores) for _ in range(num_selected)]

        # Crossover Step: Generate Offspring
        new_population = []
        for i in range(0, num_selected, 2):
            if i + 1 < num_selected:
                if random.random() < 0.9:  # 90% chance of crossover
                    child1, child2 = ordered_crossover(selected_population[i], selected_population[i + 1])
                    new_population.extend([child1, child2])
                else:
                    new_population.extend(
                        [selected_population[i], selected_population[i + 1]])  # Keep parents unchanged

        # Mutation Step
        new_population = [mutate(route) for route in new_population]

        # Evaluate Fitness of New Population
        new_fitness_scores = [route_distance(route, distance_matrix) for route in new_population]
        fitness_calculations += len(new_population)  # Update fitness count

        # Keep the Best Solution Found
        min_distance = min(new_fitness_scores)
        if min_distance < best_distance:
            best_distance = min_distance
            best_route = new_population[new_fitness_scores.index(min_distance)]

        if fitness_calculations % 5000 < len(new_population):
            print(f"Fitness Calc: {fitness_calculations} | Best Distance: {best_distance}")

        # Stop if We Find a Solution <9000
        # if best_distance < 9000:
        #     break

        # Keep the best 50% of parents, replace the rest
        num_elite = POPULATION_SIZE // 2
        elite_population = sorted(zip(population, fitness_scores), key=lambda x: x[1])[:num_elite]  # Top 50%
        elite_population = [x[0] for x in elite_population]  # Extract just the routes

        # Replace only the weaker half of the population
        population = elite_population + new_population[:num_elite]
        fitness_scores = [route_distance(route, distance_matrix) for route in population]

    return best_route, best_distance


berlin52_df = load_tsp_file(FILE_PATH)
distance_matrix = compute_distance_matrix(berlin52_df)


# Run the genetic algorithm and get the best route
best_route, best_distance = genetic_algorithm()

# Print the best route and its total distance
print("\n--- Best Route Found ---")
print("Route:", best_route)
print("Total Distance:", best_distance)

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
