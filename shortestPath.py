import heapq

# Graph representation: Each city points to a list of tuples (neighbor, distance)
graph = {
    "Almeria": [("Granada", 167), ("Murcia", 218)],
    "Granada": [("Jaen", 92), ("Malaga", 123), ("Almeria", 167)],
    "Murcia": [("Albacete", 146), ("Alicante", 81), ("Almeria", 218)],
    "Alicante": [("Albacete", 167), ("Murcia", 81)],
    "Albacete": [("Murcia", 146), ("Alicante", 167), ("Cuenca", 144)],
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

# Straight-line distances to Valladolid
straight_line_distances = {
    "Almeria": 571, "Granada": 507, "Jaen": 439, "Cordoba": 419, "Malaga": 550,
    "Huelva": 525, "Sevilla": 487, "Cadiz": 586, "Murcia": 510, "Albacete": 383,
    "Alicante": 515, "Valencia": 441, "Castellon": 435, "Tarragona": 502, "Barcelona": 576,
    "Lleida": 445, "Gerona": 627, "Merida": 334, "Badajoz": 363, "Caceres": 280,
    "CiudadReal": 305, "Toledo": 208, "Cuenca": 280, "Guadalajara": 173, "Zaragoza": 319,
    "Teruel": 337, "Huesca": 362, "Logrono": 209, "Vitoria": 215, "Bilbao": 232,
    "SanSebastian": 292, "Santander": 215, "Oviedo": 212, "Coruna": 357, "Santiago": 343,
    "Pontevedra": 335, "Orense": 271, "Lugo": 278, "Madrid": 162, "Leon": 126,
    "Zamora": 87, "Salamanca": 109, "Segovia": 94, "Valladolid": 0, "Burgos": 114,
    "Palencia": 43, "Soria": 187, "Pamplona": 284, "Avila": 110
}


def greedyFirstSearch(node):
    min_heap = []
    visited = set()
    came_from = {}

    # Push first node (start)
    heapq.heappush(min_heap, (straight_line_distances[node], node))

    # Process heap
    while min_heap:
        print("Heap:", min_heap)

        # Get the city with the shortest straight-line distance
        _, city = heapq.heappop(min_heap)

        if city in visited:
            continue

        if city == "Valladolid":
            print(f"Goal reached: {city}")
            return

        # Add it to the list of visited cities
        visited.add(city)
        print(f"Visiting: {city}")

        for neighbor, _ in graph[city]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (straight_line_distances[neighbor], neighbor))
                came_from[neighbor] = city
    print("Visited:", visited)


def aStar(node):
    min_heap = []
    visited = set()
    came_from = {}  # Stores previous node (of the best path) for each node
    g_score = {city: float("inf") for city in graph}  # store the lowest known cost to reach each node.
    # TODO: But how do we know path with that cost?

    # Push first node (start)
    heapq.heappush(min_heap, (straight_line_distances[node], node, 0))  # (f, city, g)

    # Process heap
    while min_heap:
        print("Heap:", min_heap)

        # Get the city with the shortest straight-line distance
        _, city = heapq.heappop(min_heap)

        if city in visited:
            continue

        if city == "Valladolid":
            print(f"Goal reached: {city}")
            return

        # Add it to the list of visited cities
        visited.add(city)
        print(f"Visiting: {city}")

        for neighbor, _ in graph[city]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (straight_line_distances[neighbor], neighbor, x))
                came_from[neighbor] = city
    print("Visited:", visited)


greedyFirstSearch("Malaga")
