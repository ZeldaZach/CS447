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
        explicit KDNode(std::vector<float> p, KDNode *lc, KDNode *hc);
        std::vector<float> point;
        KDNode *lower_child, *higher_child;
    };

    explicit KDTree(std::vector<std::vector<float>> points);
    ~KDTree();
    KDTree::KDNode *treeify(std::vector<std::vector<float>> points, int depth);

    KDNode *getRoot();

private:
    KDNode *root_node;
};
#endif // PARALLEL_K_NN_KDTREE_H
