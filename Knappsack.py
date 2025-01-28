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

# TODO: Not parsing okay?

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


def DFS(max_weight, items):
    # Initialize stack
    stack = [(0, 0, 0, [])]

    # track the best solution and value
    best_value = 0
    best_solution = []

    threshold_weight = 0.9 * max_weight

    while stack:
        # get the current item
        item_idx, value, weight, solution = stack.pop()

        if weight >= threshold_weight:
            if value > best_value:
                best_value = value
                best_solution = solution.copy()
            return

        # Base case: if we have processed all items
        if item_idx == len(items):
            if value > best_value:
                best_value = value
                best_solution = solution.copy()
            continue

        item_id, item_value, item_weight = items[item_idx]

        stack.append((item_idx + 1, value, weight, solution))

        if weight + item_weight <= max_weight:
            stack.append((item_idx + 1, value + item_value, weight + item_weight, solution + [item_id]))

    print(f"Best value: {best_value}, Best solution: {best_solution}")


DFS(max_weight, items)
