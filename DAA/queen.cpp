#include <iostream>
using namespace std;

int N;
int board[20][20];

// Check if placing queen at board[row][col] is safe
bool isSafe(int row, int col)
{
    int i, j;

    // Check row on left side
    for (i = 0; i < col; i++)
        if (board[row][i])
            return false;

    // Check upper diagonal
    for (i = row, j = col; i >= 0 && j >= 0; i--, j--)
        if (board[i][j])
            return false;

    // Check lower diagonal
    for (i = row, j = col; j >= 0 && i < N; i++, j--)
        if (board[i][j])
            return false;

    return true;
}

// Backtracking function
bool solveNQueen(int col)
{
    if (col >= N)
        return true;

    // Skip column already having first queen
    bool colHasQueen = false;
    for (int r = 0; r < N; r++)
    {
        if (board[r][col] == 1)
        {
            colHasQueen = true;
            break;
        }
    }
    if (colHasQueen)
        return solveNQueen(col + 1);

    for (int row = 0; row < N; row++)
    {
        if (isSafe(row, col))
        {
            board[row][col] = 1;

            if (solveNQueen(col + 1))
                return true;

            board[row][col] = 0; // Backtrack
        }
    }
    return false;
}

// Print board
void printBoard()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << board[i][j] << " ";
        cout << endl;
    }
}

int main()
{
    cout << "Enter value of N: ";
    cin >> N;

    int r, c;
    cout << "Enter position of first queen (row and column, 0-indexed): ";
    cin >> r >> c;

    // Initialize board
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            board[i][j] = 0;

    // Place first queen
    board[r][c] = 1;

    if (solveNQueen(0))
        printBoard();
    else
        cout << "No solution possible.";

    return 0;
}
