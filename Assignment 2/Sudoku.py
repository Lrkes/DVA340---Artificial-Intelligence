import time


def parse_sudoku_from_text(file_path):
    """Extracts Sudoku puzzles from .txt -> list."""
    sudokus = []
    current_grid = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("SUDOKU") or line == "EOF" or line == "":
                continue
            if len(line) == 9 and line.isdigit():
                current_grid.append([int(num) for num in line])
                if len(current_grid) == 9:
                    sudokus.append(current_grid)
                    current_grid = []

    return sudokus


def find_empty(board):
    """Find the first empty cell in the Sudoku board."""
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None, None


def is_valid(board, row, col, num):
    """Check if 'num' can be placed at position without violating Sudoku rules."""

    # Check row
    if num in board[row]:
        return False

    # Check column
    for i in range(9):
        if board[i][col] == num:
            return False

    # Check 3x3 grid
    start_row, start_col = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[start_row + i][start_col + j] == num:
                return False
    return True


def solve_sudoku(board):
    """Solve the Sudoku puzzle using backtracking."""
    row, col = find_empty(board)

    if row is None:  # No empty cells left, puzzle is solved
        return True

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num

            if solve_sudoku(board):
                return True

            board[row][col] = 0  # Undo the current cell for backtracking

    return False


def print_board(board):
    """Pretty Sudoku Board."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - -")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(board[i][j], end=" ")
        print()


file_path = "sudoku.txt"
sudoku_puzzles = parse_sudoku_from_text(file_path)

total_time = 0

for idx, puzzle in enumerate(sudoku_puzzles):
    print(f"\nSudoku {idx + 1} Starting Position:")
    print_board(puzzle)

    start_time = time.time()

    if solve_sudoku(puzzle):
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\n🏆 Sudoku {idx + 1} Solved in {elapsed_time:.4f} seconds:")
        print_board(puzzle)
    else:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n✖ Sudoku {idx + 1} has no solution!")

    total_time += elapsed_time

    print(f"\nTotal Time Taken: {total_time:.4f} seconds for {len(sudoku_puzzles)} puzzles.")

