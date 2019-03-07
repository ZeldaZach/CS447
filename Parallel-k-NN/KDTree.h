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
    void postorder(Node *p, int indent = 0);

private:
    KDTree::Node *buildTree(std::vector<std::vector<float>>, unsigned long);

    Node *getNearestNeighbors(KDTree::Node *, KDTree::Node *, unsigned long);

    float euclidianDistance(const std::vector<float> &, const std::vector<float> &);

    void deleteTree(Node *root);

private:
    Node *root_node;
    std::priority_queue<Node *> priority_queue;
    unsigned long how_many_neighbors;
};
#endif // PARALLEL_K_NN_KDTREE_H
