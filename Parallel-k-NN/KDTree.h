//
// Created by Zachary Halpern on 2019-03-06.
//

#ifndef PARALLEL_K_NN_KDTREE_H
#define PARALLEL_K_NN_KDTREE_H

#include <future>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

class KDTree
{

private: // Internal variables/structs
    struct Node
    {
        explicit Node(std::vector<float>, Node *, Node *);
        const std::vector<float> point;
        Node *lower_child, *higher_child;
    };

    typedef std::pair<float, Node *> queue_pair;
    Node *root_node;
    const unsigned long k_neighbors, max_threads;
    std::atomic<unsigned long> thread_count;

public: // External methods
    explicit KDTree(std::vector<std::vector<float>> *, unsigned long, unsigned long max_threads);
    ~KDTree();
    std::vector<std::vector<float>> getNearestNeighbors(std::vector<float>);
    Node *getRoot();
    static std::string vectorToString(const std::vector<float> &);

private: // Internal methods
    KDTree::Node *buildTree(std::vector<std::vector<float>>::iterator,
                            std::vector<std::vector<float>>::iterator,
                            unsigned long,
                            std::promise<Node *> *promise = nullptr);
    void getNearestNeighbors(KDTree::Node *input,
                             KDTree::Node *root,
                             unsigned long depth,
                             std::priority_queue<queue_pair, std::vector<queue_pair>, std::less<>> *priority_queue);
    float euclidianDistance(const std::vector<float> &, const std::vector<float> &);
    void deleteTree(Node *root);
    bool pruneAwayResults(KDTree::Node *input,
                          KDTree::Node *root,
                          unsigned long depth,
                          std::priority_queue<queue_pair, std::vector<queue_pair>, std::less<>> *priority_queue);
};
#endif // PARALLEL_K_NN_KDTREE_H
