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

print(items)
