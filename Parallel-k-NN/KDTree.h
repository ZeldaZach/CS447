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
    struct KDNode
    {
        explicit KDNode(std::vector<float>, KDNode *, KDNode *);
        std::vector<float> point;
        KDNode *lower_child, *higher_child;
    };

    KDTree::KDNode *build_tree(std::vector<std::vector<float>>, unsigned long);
    KDNode *getRoot();
    KDNode *getNearestNeighbor(KDTree::KDNode *, KDTree::KDNode *, KDTree::KDNode *, unsigned long);
    float euclidianDistance(const std::vector<float> &, const std::vector<float> &);

private:
    KDNode *root_node;
};
#endif // PARALLEL_K_NN_KDTREE_H
