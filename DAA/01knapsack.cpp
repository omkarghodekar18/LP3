#include <iostream>
#include <vector>
using namespace std;

int knapsackDP(int W, vector<int> weight, vector<int> profit, int n)
{
    vector<vector<int>> K(n + 1, vector<int>(W + 1, 0));

    for (int i = 0; i <= n; i++)
    {
        for (int w = 0; w <= W; w++)
        {
            if (i == 0 || w == 0)
                K[i][w] = 0; 
            else if (weight[i - 1] <= w)
                K[i][w] = max(profit[i - 1] + K[i - 1][w - weight[i - 1]], K[i - 1][w]);
            else
                K[i][w] = K[i - 1][w];
        }
    }

    return K[n][W];
}

int main()
{
    int n, W;
    cout << "Enter number of items: ";
    cin >> n;

    vector<int> profit(n), weight(n);

    cout << "Enter profits of items: ";
    for (int i = 0; i < n; i++)
        cin >> profit[i];

    cout << "Enter weights of items: ";
    for (int i = 0; i < n; i++)
        cin >> weight[i];

    cout << "Enter knapsack capacity: ";
    cin >> W;

    int maxProfit = knapsackDP(W, weight, profit, n);
    cout << "\nMaximum profit = " << maxProfit << endl;

    return 0;
}