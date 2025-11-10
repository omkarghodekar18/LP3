#include <iostream>
#include <queue>
#include <vector>
#include <map>
using namespace std;

// Node structure for Huffman tree
struct Node
{
    char ch;
    int freq;
    Node *left, *right;

    Node(char c, int f)
    {
        ch = c;
        freq = f;
        left = right = nullptr;
    }
};

// Comparison for min-heap (priority queue)
struct compare
{
    bool operator()(Node *l, Node *r)
    {
        return l->freq > r->freq; // smaller freq = higher priority
    }
};

// Recursive function to print Huffman codes
void printCodes(Node *root, string code)
{
    if (!root)
        return;

    // Leaf node → print character and code
    if (!root->left && !root->right)
        cout << root->ch << " : " << code << endl;

    // Left edge = '0', right edge = '1'
    printCodes(root->left, code + "0");
    printCodes(root->right, code + "1");
}

int main()
{
    int n;
    cout << "Enter number of characters: ";
    cin >> n;

    vector<char> chars(n);
    vector<int> freq(n);

    cout << "Enter characters:\n";
    for (int i = 0; i < n; i++)
        cin >> chars[i];

    cout << "Enter their frequencies:\n";
    for (int i = 0; i < n; i++)
        cin >> freq[i];

    // Min-heap priority queue
    priority_queue<Node *, vector<Node *>, compare> pq;

    // Step 1: Create leaf nodes and insert into priority queue
    for (int i = 0; i < n; i++)
        pq.push(new Node(chars[i], freq[i]));

    // Step 2: Build Huffman Tree
    while (pq.size() > 1)
    {
        Node *left = pq.top();
        pq.pop();
        Node *right = pq.top();
        pq.pop();

        Node *newNode = new Node('$', left->freq + right->freq);
        newNode->left = left;
        newNode->right = right;

        pq.push(newNode);
    }

    // Step 3: Root of Huffman Tree
    Node *root = pq.top();

    cout << "\nHuffman Codes are:\n";
    printCodes(root, "");

    return 0;
}