//
// Created by Zachary Halpern on 2019-03-06.
//

#ifndef PARALLEL_K_NN_KDTREE_H
#define PARALLEL_K_NN_KDTREE_H

#include <vector>

class KDTree
{
public:
    explicit KDTree(std::vector<std::vector<float>>);
    ~KDTree();
    std::vector<float> getNearestNeighbor(std::vector<float>);

private:
    struct Node
    {
        explicit Node(std::vector<float>, Node *, Node *);
        std::vector<float> point;
        Node *lower_child, *higher_child;
    };

    KDTree::Node *buildTree(std::vector<std::vector<float>>, unsigned long);
    Node *getRoot();
    Node *getNearestNeighbor(KDTree::Node *, KDTree::Node *, KDTree::Node *, unsigned long);
    float euclidianDistance(const std::vector<float> &, const std::vector<float> &);

private:
    Node *root_node;
};
#endif // PARALLEL_K_NN_KDTREE_H
