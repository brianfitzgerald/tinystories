from synthetic_data.utils import extract_code_block


MOCK_CODE = """
Here's a Python solution for the classic N-Queens problem using backtracking:

```python
def solve_n_queens(n):
    def is_safe(board, row, col):
        # Check this row on left side
        for i in range(col):
            if board[row][i] == 'Q':
                return False
        # Check upper diagonal on left side
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False
        # Check lower diagonal on left side
        for i, j in zip(range(row, n), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False
        return True

    def solve(board, col, solutions):
        # If all queens are placed, add the solution
        if col >= n:
            solutions.append(["".join(row) for row in board])
            return
        # Try placing queen in all rows one by one
        for i in range(n):
            if is_safe(board, i, col):
                board[i][col] = 'Q'
                solve(board, col + 1, solutions)
                board[i][col] = '.'  # Backtrack

    # Initialize the board and the solutions list
    board = [["." for _ in range(n)] for _ in range(n)]
    solutions = []
    solve(board, 0, solutions)
    return solutions
```

### Usage
To find solutions for \( N = 4 \):

```python
n = 4
solutions = solve_n_queens(n)
for i, solution in enumerate(solutions, 1):
    print(f"Solution {i}:")
    for row in solution:
        print(row)
    print()
```

### Explanation
1. **`is_safe` function** checks if placing a queen at a given position would be safe by checking for queens in the same row, and both diagonals.
2. **`solve` function** uses backtracking to attempt to place queens in each column.
3. **Backtracking** occurs when a queen cannot be safely placed in a column, and the function "backtracks" to try different positions. 

The result is a list of all possible solutions, where each solution is represented as a list of strings, with each string representing a row on the board.
"""


def test_extract_python():
    python_code = extract_code_block(MOCK_CODE, "python")

    print(python_code[0])
    assert len(python_code) == 2
