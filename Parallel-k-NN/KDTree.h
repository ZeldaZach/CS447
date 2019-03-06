//
// Created by Zachary Halpern on 2019-03-06.
//

#ifndef PARALLEL_K_NN_KDTREE_H
#define PARALLEL_K_NN_KDTREE_H

#include <vector>

class KDTree
{
public:
    struct KDNode
    {
        explicit KDNode(std::vector<float>, KDNode *, KDNode *);
        std::vector<float> point;
        KDNode *lower_child, *higher_child;
        float e_distance;
    };

    explicit KDTree(std::vector<std::vector<float>>);
    ~KDTree();
    KDTree::KDNode *treeify(std::vector<std::vector<float>>, unsigned long);

    KDNode *getRoot();
    KDNode *getNearestNeighbor(std::vector<float>);
    KDNode *getNearestNeighbor(KDTree::KDNode *, KDTree::KDNode *, KDTree::KDNode *, unsigned long);

    float getEuclidianDistance(const std::vector<float> &p1, const std::vector<float> &p2);

private:
    KDNode *root_node;
};
#endif // PARALLEL_K_NN_KDTREE_H
