import random
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


N_CITIES = 52
N_ANTS = 50  # Number of ants per iteration -> Number of paths per iteration
MAX_ITERATIONS = 100  # Number of iterations
EVAPORATION_RATE = 0.1  # Pheromone evaporation rate
ALPHA = 1  # Influence of pheromone: higher values give more importance to pheromones
BETA = 2  # Influence of heuristic (1/distance): higher values give more importance to distance
Q = 100  # Pheromone deposit factor - How much pheromone an ant deposits on its path


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
        locations.append({"ID": int(parts[0]), "X": float(parts[1]), "Y": float(parts[2])})

    return pd.DataFrame(locations)


def compute_distance_matrix(df):
    """Computes Euclidean distance for all city pairs."""
    locations = df[['X', 'Y']].values
    return {(i, j): euclidean(locations[i], locations[j]) for i in range(len(locations)) for j in range(len(locations))}


def initialize_pheromone_matrix(initial_value=1.0):
    """Initializes the pheromone matrix with a default value for all city pairs."""
    # Uniform initial values -> equal likelihood of choosing any city
    return {(i, j): initial_value for i in range(N_CITIES) for j in range(N_CITIES)}


def route_distance(route, distance_matrix):
    """Computes the total distance of a given TSP route."""
    return sum(distance_matrix[(route[i], route[i+1])] for i in range(len(route) - 1))


def construct_ant_route(pheromone, distance_matrix, alpha=ALPHA, beta=BETA):
    """Constructs a TSP route based on pheromone levels and heuristic."""

    # Each ant starts at 0 and chooses the next city based on pheromone levels and heuristic
    route = [0]  # Start at city 0
    available_cities = set(range(1, N_CITIES))

    while available_cities:
        current_city = route[-1]  # Last city in route
        probabilities = []
        for city in available_cities:
            pheromone_level = pheromone[(current_city, city)] ** alpha
            heuristic = (1 / distance_matrix[(current_city, city)]) ** beta
            probabilities.append(pheromone_level * heuristic)  # Calculate probability for each available city

        probabilities = [p / sum(probabilities) for p in probabilities]  # Normalize: Scores to probabilities
        next_city = random.choices(list(available_cities), probabilities)[0]  # Choose next city based on probabilities

        # Update route and available cities
        route.append(next_city)
        available_cities.remove(next_city)

    route.append(0)  # Return to start
    return route


def update_pheromones(pheromone, ant_routes, distance_matrix, evaporation_rate=EVAPORATION_RATE):
    """Updates pheromone levels based on ant routes and fitness."""

    # Balances Evaporation (Forgetting Bad Paths) and Deposition (Reinforces Good Paths)

    # Evaporate pheromones
    for key in pheromone:
        pheromone[key] *= (1 - evaporation_rate)

    # Deposit new pheromones based on ant performance
    for route in ant_routes:
        distance = route_distance(route, distance_matrix)
        contribution = Q / distance  # Contribution = DepositFactor / TotalDistance: More contribution for shorter paths
        for i in range(len(route) - 1):
            pheromone[(route[i], route[i+1])] += contribution
            pheromone[(route[i+1], route[i])] += contribution  # both ways


def ant_colony_optimization(max_iterations=MAX_ITERATIONS):
    """Runs the ACO algorithm to find an optimal TSP route."""
    pheromone = initialize_pheromone_matrix()
    best_route = None
    best_distance = float("inf")  # infinite
    performance = []

    for iteration in range(max_iterations):
        # Generate routes for all ants
        ant_routes = [construct_ant_route(pheromone, distance_matrix) for _ in range(N_ANTS)]
        distances = [route_distance(route, distance_matrix) for route in ant_routes]

        # Keeping track best solution
        min_distance = min(distances)
        if min_distance < best_distance:
            best_distance = min_distance
            best_route = ant_routes[distances.index(min_distance)]

        # Update pheromones based on performance
        update_pheromones(pheromone, ant_routes, distance_matrix)

        # Print progress every 10 iterations
        if iteration % 5 == 0:
            performance.append(best_distance)
            print(f"Iteration {iteration}: Best Distance = {best_distance}")

    return best_route, best_distance, performance


berlin52_df = load_tsp_file("./Assignment 3 berlin52.tsp")
distance_matrix = compute_distance_matrix(berlin52_df)

best_route, best_distance, aco_performance = ant_colony_optimization()

print("\n--- Best Route Found   ---")
print("Route:", best_route)
print("Total Distance:", best_distance)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot([i * 5 for i in range(len(aco_performance))], aco_performance, label="Ant Colony Optimization", color='green')
plt.axhline(y=9000, color='red', linestyle='dashed', label="Threshold: 9000")
plt.xlabel("Iterations")
plt.ylabel("Shortest Distance")
plt.title("Ant Colony Optimization Performance")

plt.text(0, max(aco_performance) * 0.95, "Updates every 5 iterations", fontsize=10, color='gray')
plt.legend()

plt.savefig('plots/aco_performance.png')
plt.show()