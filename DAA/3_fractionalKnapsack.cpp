#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Item
{
    int weight;
    int profit;
    double ratio; // profit/weight
};

bool compare(Item a, Item b)
{
    return a.ratio > b.ratio; // Sort in descending order of ratio
}

int main()
{
    int n, capacity;
    cout << "Enter number of items: ";
    cin >> n;

    vector<Item> items(n);

    cout << "Enter weight and profit of each item:\n";
    for (int i = 0; i < n; i++)
    {
        cin >> items[i].weight >> items[i].profit;
        items[i].ratio = (double)items[i].profit / items[i].weight;
    }

    cout << "Enter capacity of knapsack: ";
    cin >> capacity;

    // Sort items based on profit/weight ratio
    sort(items.begin(), items.end(), compare);

    double maxProfit = 0.0;
    int remainingCapacity = capacity;

    for (int i = 0; i < n; i++)
    {
        if (items[i].weight <= remainingCapacity)
        {
            // Take whole item
            maxProfit += items[i].profit;
            remainingCapacity -= items[i].weight;
        }
        else
        {
            // Take fraction of item
            maxProfit += items[i].ratio * remainingCapacity;
            break; // Knapsack full
        }
    }

    cout << "\nMaximum Profit = " << maxProfit << endl;
    return 0;
}
