#include <iostream>
#include <vector>
using namespace std;

vector<vector<int>> dp;

// Recursive + DP (Memoization)
int knapsackMemo(int index, int capacity, vector<int> &weight, vector<int> &profit)
{
    if (index == 0 || capacity == 0)
        return 0;

    if (dp[index][capacity] != -1)
        return dp[index][capacity];

    if (weight[index - 1] > capacity)
        dp[index][capacity] = knapsackMemo(index - 1, capacity, weight, profit);
    else
    {
        int include = profit[index - 1] + knapsackMemo(index - 1, capacity - weight[index - 1], weight, profit);
        int exclude = knapsackMemo(index - 1, capacity, weight, profit);
        dp[index][capacity] = max(include, exclude);
    }

    return dp[index][capacity];
}

int main()
{
    int n, capacity;
    cout << "Enter number of items: ";
    cin >> n;

    vector<int> weight(n), profit(n);

    cout << "Enter weight and profit of each item:\n";
    for (int i = 0; i < n; i++)
        cin >> weight[i] >> profit[i];

    cout << "Enter knapsack capacity: ";
    cin >> capacity;

    // DP Memoization table initialization
    dp.assign(n + 1, vector<int>(capacity + 1, -1));
    int ansMemo = knapsackMemo(n, capacity, weight, profit);

    cout << "\nMaximum Profit (DP + Memoization) = " << ansMemo << endl;

    return 0;
}
