#include <iostream>
using namespace std;

// Recursive Fibonacci Function
int fibonacciRecursive(int n)
{
    if (n <= 1)
        return n;
    return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
}

// Non-Recursive Fibonacci Function
int fibonacciNonRecursive(int n)
{
    int a = 0, b = 1, c;
    if (n == 0)
        return a;
    for (int i = 2; i <= n; i++)
    {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int main()
{
    int n;
    cout << "Enter number of terms: ";
    cin >> n;

    cout << "\nFibonacci using Recursive Approach:" << endl;
    for (int i = 0; i < n; i++)
        cout << fibonacciRecursive(i) << " ";

    cout << "\n\nFibonacci using Non-Recursive Approach:" << endl;
    for (int i = 0; i < n; i++)
        cout << fibonacciNonRecursive(i) << " ";

    cout << endl;
    return 0;
}
