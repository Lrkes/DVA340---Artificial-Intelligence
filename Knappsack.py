from collections import deque

"""
0-1 Knapsack Problem

Goal:
Select items to maximize total value without exceeding the knapsack's weight capacity.

Given:
- A list of items, each with a benefit (value) and weight.
- A maximum weight capacity for the knapsack.

Constraints:
- Each item can only be included once (0 or 1).
- Total weight of selected items ≤ max weight.
"""

# Problem data
max_weight = 420
items = [
    # (ID, Benefit, Weight)
    (1, 20, 15),
    (2, 40, 32),
    (3, 50, 60),
    (4, 36, 80),
    (5, 26, 43),
    (6, 64, 120),
    (7, 54, 77),
    (8, 18, 6),
    (9, 46, 93),
    (10, 28, 35),
    (11, 25, 37),
]


def DFS(target_ratio=None):
    stack = [(0, 0, 0, [])]  # (item_idx, value, weight, solution)
    # If target_ratio is set, return the first valid solution ("good enough")
    threshold_weight = target_ratio * max_weight if target_ratio is not None else max_weight

    best_solution, best_value = [], 0  # Track the best solution for optimal value

    while stack:
        item_idx, value, weight, solution = stack.pop()

        # If target_ratio is set, return the first valid solution ("good enough")
        if target_ratio is not None and weight >= threshold_weight:
            print(f"First good enough value: {value}, Solution: {solution}")
            return solution, value

        if target_ratio is None and weight <= max_weight and value > best_value:
            best_value, best_solution = value, solution

        # Base case: all items processed
        if item_idx == len(items):
            continue

        item_id, item_value, item_weight = items[item_idx]

        # Add item if it fits
        if weight + item_weight <= max_weight:
            stack.append((item_idx + 1, value + item_value, weight + item_weight, solution + [item_id]))

        # Skip item
        stack.append((item_idx + 1, value, weight, solution))

    # Return best solution if no threshold solution was found
    print(f"Optimal value: {best_value}, Solution: {best_solution}")
    return best_solution, best_value


def BFS(target_ratio=None):
    queue = deque([(0, 0, 0, [])])  # (item_idx, value, weight, solution (tracks path))
    threshold_weight = target_ratio * max_weight if target_ratio is not None else max_weight
    best_solution, best_value = [], 0

    while queue:
        item_idx, value, weight, solution = queue.popleft()

        # Stop early if target ratio is met (good enough solution)
        if weight >= threshold_weight:
            print(f"First good enough value found: {value}, Solution: {solution}")
            return solution, value

        # Track the best if no target ratio is set
        if target_ratio is None and value > best_value and weight <= max_weight:
            best_solution, best_value = solution, value

        # Base case: all items processed
        if item_idx == len(items):
            continue

        item_id, item_value, item_weight = items[item_idx]

        # Explore including the current item (if it fits)
        if weight + item_weight <= max_weight:
            queue.append((item_idx + 1, value + item_value, weight + item_weight, solution + [item_id]))

        # Explore excluding the current item
        queue.append((item_idx + 1, value, weight, solution))

    print(f"Optimal value found: {best_value}, Solution: {best_solution}")
    return best_solution, best_value


# Running DFS for both cases
print("DFS with target ratio (fastest first okay solution):")
DFS(target_ratio=0.9)

print("\nDFS with no target ratio (search for best solution):")
DFS()
print("---------------------------------------------------------------")
# Running BFS for both cases
print("BFS with target ratio (fastest first okay solution):")
BFS(target_ratio=0.9)

print("\nBFS with no target ratio (search for best solution):")
BFS()

# DFS is uses less memory than BFS
# BFS is usually faster(?)
