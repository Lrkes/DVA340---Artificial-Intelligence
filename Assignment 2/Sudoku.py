def parse_sudoku_from_text(file_path):
    """Extracts Sudoku puzzles from .txt -> list."""
    sudokus = []
    current_grid = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("SUDOKU") or line == "EOF" or line == "":
                # Ignore metadata lines
                continue
            if len(line) == 9 and line.isdigit():
                current_grid.append([int(num) for num in line])
                if len(current_grid) == 9:
                    sudokus.append(current_grid)
                    current_grid = []

    return sudokus


file_path = "sudoku.txt"
sudoku_puzzles = parse_sudoku_from_text(file_path)

print("First Sudoku Grid:")
for row in sudoku_puzzles[0]:
    print(row)


def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    if num in[board[i][col] for i in range(9)]:
        return False
    # Check 3x3 grid
    # check if // 3 for starting point