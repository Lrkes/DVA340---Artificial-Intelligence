import heapq

# Graph representation: Each city points to a list of tuples (neighbor, distance)
graph = {
    "Almeria": [("Granada", 167), ("Murcia", 218)],
    "Granada": [("Almeria", 167), ("Jaen", 92), ("Malaga", 123)],
    "Murcia": [("Albacete", 146), ("Alicante", 81), ("Almeria", 218)],
    "Alicante": [("Albacete", 167), ("Murcia", 81)],
    "Albacete": [("Murcia", 146), ("Alicante", 167), ("Cuenca", 144), ("Madrid", 257)],
    "Jaen": [("Granada", 92), ("Cordoba", 120), ("Madrid", 331)],
    "Cordoba": [("Sevilla", 140), ("Jaen", 120), ("CiudadReal", 195)],
    "Malaga": [("Sevilla", 206), ("Cadiz", 235), ("Granada", 123)],
    "Cadiz": [("Huelva", 210), ("Sevilla", 121), ("Malaga", 235)],
    "Sevilla": [("Malaga", 206), ("Cadiz", 121), ("Cordoba", 140), ("Merida", 192)],
    "Huelva": [("Cadiz", 210)],
    "Merida": [("Sevilla", 192), ("Badajoz", 64), ("Caceres", 75)],
    "Badajoz": [("Merida", 64)],
    "Caceres": [("Merida", 75), ("Salamanca", 202), ("Madrid", 301)],
    "Valencia": [("Alicante", 166), ("Castellon", 74), ("Cuenca", 199)],
    "Castellon": [("Valencia", 74), ("Tarragona", 187), ("Teruel", 144)],
    "Teruel": [("Cuenca", 148), ("Zaragoza", 171), ("Castellon", 144)],
    "Tarragona": [("Barcelona", 99), ("Castellon", 187)],
    "Barcelona": [("Lleida", 163), ("Tarragona", 99), ("Gerona", 103)],
    "Lleida": [("Barcelona", 163), ("Gerona", 229), ("Huesca", 112), ("Zaragoza", 152)],
    "Gerona": [("Barcelona", 103), ("Lleida", 229)],
    "Zaragoza": [("Huesca", 74), ("Pamplona", 178), ("Soria", 159), ("Guadalajara", 256), ("Lleida", 152),
                 ("Teruel", 171)],
    "Huesca": [("Zaragoza", 74), ("Lleida", 112), ("Pamplona", 165)],
    "Pamplona": [("Huesca", 165), ("Zaragoza", 178)],
    "Soria": [("Zaragoza", 159), ("Guadalajara", 171), ("Logrono", 101), ("Burgos", 142)],
    "Guadalajara": [("Zaragoza", 256), ("Soria", 171), ("Madrid", 60)],
    "Logrono": [("Soria", 101), ("Vitoria", 94)],
    "Vitoria": [("Logrono", 94), ("Burgos", 118), ("SanSebastian", 100), ("Bilbao", 62)],
    "SanSebastian": [("Vitoria", 100), ("Bilbao", 101)],
    "Bilbao": [("Vitoria", 62), ("SanSebastian", 101), ("Santander", 100)],
    "Santander": [("Bilbao", 100), ("Burgos", 181), ("Oviedo", 192)],
    "Oviedo": [("Santander", 192), ("Leon", 125), ("Lugo", 227)],
    "Leon": [("Zamora", 141), ("Lugo", 223), ("Oviedo", 125)],
    "Lugo": [("Leon", 223), ("Oviedo", 227), ("Coruna", 98), ("Pontevedra", 195)],
    "Coruna": [("Lugo", 98), ("Oviedo", 287), ("Santiago", 75)],
    "Santiago": [("Coruna", 75), ("Pontevedra", 64)],
    "Pontevedra": [("Santiago", 64), ("Orense", 119)],
    "Orense": [("Pontevedra", 119), ("Zamora", 259)],
    "Zamora": [("Orense", 259), ("Leon", 141), ("Salamanca", 66)],
    "Salamanca": [("Zamora", 66), ("Avila", 109), ("Caceres", 202)],
    "Avila": [("Salamanca", 109), ("Segovia", 66), ("Madrid", 109)],
    "Segovia": [("Avila", 66), ("Valladolid", 115), ("Madrid", 92)],
    "Valladolid": [("Segovia", 115), ("Leon", 136), ("Zamora", 100), ("Palencia", 51)],
    "Palencia": [("Valladolid", 51), ("Burgos", 92)],
    "Burgos": [("Palencia", 92), ("Soria", 142), ("Logrono", 118), ("Santander", 181)],
    "Madrid": [("Jaen", 331), ("Albacete", 257), ("Cuenca", 168), ("Guadalajara", 60), ("Burgos", 245), ("Segovia", 92),
               ("Avila", 109), ("Caceres", 301), ("Toledo", 72)],
    "CiudadReal": [("Cordoba", 195), ("Toledo", 118)],
    "Cuenca": [("Valencia", 199), ("Albacete", 144), ("Madrid", 168), ("Teruel", 148)],
    "Toledo": [("CiudadReal", 118), ("Madrid", 72)]
}

# Heuristic: Straight-line distances to Valladolid
straight_line_distances = {
    "Almeria": 571, "Granada": 507, "Jaen": 439, "Cordoba": 419, "Malaga": 550, "Huelva": 525, "Sevilla": 487,
    "Cadiz": 586, "Murcia": 510, "Albacete": 383, "Alicante": 515, "Valencia": 441, "Castellon": 435, "Tarragona": 502,
    "Barcelona": 576, "Lleida": 445, "Gerona": 627, "Merida": 334, "Badajoz": 363, "Caceres": 280, "CiudadReal": 305,
    "Toledo": 208, "Cuenca": 280, "Guadalajara": 173, "Zaragoza": 319, "Teruel": 337, "Huesca": 362, "Logrono": 209,
    "Vitoria": 215, "Bilbao": 232, "SanSebastian": 292, "Santander": 215, "Oviedo": 212, "Coruna": 357, "Santiago": 343,
    "Pontevedra": 335, "Orense": 271, "Lugo": 278, "Madrid": 162, "Leon": 126, "Zamora": 87, "Salamanca": 109,
    "Segovia": 94, "Valladolid": 0, "Burgos": 114, "Palencia": 43, "Soria": 187, "Pamplona": 284, "Avila": 110
}


def greedyFirstSearch(node):
    """
        Greedy Best-First Search algorithm.
        It selects the next city based only on the heuristic (straight-line distance to the goal).
    """
    min_heap = []
    visited = set()  # closed-list
    came_from = {}

    # Push first node (start)
    heapq.heappush(min_heap, (straight_line_distances[node], node))

    # Process heap
    while min_heap:
        # Get city with the shortest estimated distance (h(n))
        _, city = heapq.heappop(min_heap)

        # skip if the city, if it was already visited
        if city in visited:
            continue

        if city == "Valladolid":
            print(f"Goal reached: {city}")
            return reconstruct_path(came_from, city)

        # Add it to the list of visited cities
        visited.add(city)

        for neighbor, _ in graph[city]:  # Explore neighbors
            if neighbor not in visited:  # again skip if the city was already visited
                heapq.heappush(min_heap, (straight_line_distances[neighbor], neighbor))  # Push neighbors + heuristic
                if neighbor not in came_from:
                    came_from[neighbor] = city  # not revisited in GFS


def aStar(node, goal="Valladolid"):
    """
        A* search algorithm.
        Uses both the actual cost (g) and the heuristic (h) to find the optimal path.
        """
    min_heap = []
    came_from = {}  # Stores previous node (best path)
    g_score = {city: float("inf") for city in graph}  # Store actual cost to reach city (g(n))
    g_score[node] = 0  # Start node has g(n) = 0

    # Push first node (start)
    heapq.heappush(min_heap, (straight_line_distances[node], node, 0))  # (f, city, g)

    while min_heap:
        # Get the city with the shortest estimated cost f(n) (min heap)
        _, city, g = heapq.heappop(min_heap)

        if city == goal:
            print(f"Goal reached: {city}")
            return reconstruct_path(came_from, city)

        for neighbor, cost in graph[city]:  # Explore neighbors
            tentative_g = g + cost  # Compute new path (potential)

            # If the new path is better, update it
            if tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + straight_line_distances[neighbor]  # f(n) = g + h
                heapq.heappush(min_heap, (f, neighbor, tentative_g))  # Push updated node
                came_from[neighbor] = city  # Track optimal path

    print("No path found")
    return None


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    path.reverse()
    return path


print(greedyFirstSearch("Malaga"))
print("=====================================")
print(aStar("Malaga"))

"""
Greedy Best-First Search:
- Uses only the heuristic function h(n).
- Always expands the node that appears closest to the goal based on h(n).
- Ignores the actual cost so far (g(n)), which can lead to suboptimal solutions.

A* Search:
- Uses f(n) = g(n) + h(n).
- Balances between the shortest known path so far (g(n)) and the estimated cost to the goal (h(n)).
- Guarantees optimality if h(n) is admissible (never overestimates the actual cost).

Key Differences:
- Greedy is faster but not necessarily optimal (can get stuck in local minima).
- A* is optimal and complete as long as h(n) is admissible.
"""
