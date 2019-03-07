//
// Created by Zachary Halpern on 2019-03-06.
//

#ifndef PARALLEL_K_NN_KDTREE_H
#define PARALLEL_K_NN_KDTREE_H

#include <queue>
#include <vector>

class KDTree
{
private:
    struct Node
    {
        explicit Node(std::vector<float>, Node *, Node *);
        std::vector<float> point;
        Node *lower_child, *higher_child;
    };

public:
    explicit KDTree(std::vector<std::vector<float>>, unsigned long);
    ~KDTree();
    std::vector<std::vector<float>> getNearestNeighbors(std::vector<float>);
    Node *getRoot();

private:
    KDTree::Node *buildTree(std::vector<std::vector<float>>, unsigned long);

    Node *getNearestNeighbors(KDTree::Node *, KDTree::Node *, unsigned long);

    float euclidianDistance(const std::vector<float> &, const std::vector<float> &);

    void deleteTree(Node *root);
    static std::string vectorToString(const std::vector<float> &);

private:
    Node *root_node;
    typedef std::pair<float, Node *> queue_pair;
    std::priority_queue<queue_pair, std::vector<queue_pair>, std::greater<>> priority_queue;
    unsigned long how_many_neighbors;
};
#endif // PARALLEL_K_NN_KDTREE_H
