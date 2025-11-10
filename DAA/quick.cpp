#include <bits/stdc++.h>
using namespace std;

int partitionDeterministic(vector<int> &arr, int low, int high)
{
    int pivot = arr[low];
    int i = low;
    int j = high;

    while (i < j)
    {

        while (arr[i] <= pivot && i <= high)
            i++;

        while (arr[j] > pivot && j >= low)
            j--;

        if (i < j)
            swap(arr[i], arr[j]);
    }

    swap(arr[low], arr[j]); // place pivot correctly
    return j;
}

void quickSortDeterministic(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int p = partitionDeterministic(arr, low, high);
        quickSortDeterministic(arr, low, p - 1);
        quickSortDeterministic(arr, p + 1, high);
    }
}

// Randomized version uses the same partition but swaps pivot first
int partitionRandomized(vector<int> &arr, int low, int high)
{
    int randomIndex = low + rand() % (high - low + 1);
    swap(arr[low], arr[randomIndex]); // random pivot at low
    return partitionDeterministic(arr, low, high);
}

void quickSortRandomized(vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int p = partitionRandomized(arr, low, high);
        quickSortRandomized(arr, low, p - 1);
        quickSortRandomized(arr, p + 1, high);
    }
}

int main()
{
    srand(time(NULL));

    int n;
    cout << "Enter number of elements: ";
    cin >> n;

    vector<int> arr1(n), arr2(n);

    cout << "Enter elements: ";
    for (int i = 0; i < n; i++)
    {
        cin >> arr1[i];
        arr2[i] = arr1[i];
    }

    quickSortDeterministic(arr1, 0, n - 1);
    quickSortRandomized(arr2, 0, n - 1);

    cout << "\nSorted using Deterministic Quick Sort: ";
    for (int x : arr1)
        cout << x << " ";

    cout << "\nSorted using Randomized Quick Sort:    ";
    for (int x : arr2)
        cout << x << " ";

    cout << endl;
    return 0;
}
