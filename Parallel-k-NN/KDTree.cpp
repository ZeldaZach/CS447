//
// Created by Zachary Halpern on 2019-03-06.
//

#include "KDTree.h"

KDTree::KDTree(std::vector<std::vector<float>> points)
{
    root_node = treeify(std::move(points), 0);
}

KDTree::~KDTree() = default;

KDTree::KDNode::KDNode(std::vector<float> p, KDTree::KDNode *lc, KDTree::KDNode *hc)
    : point(std::move(p)), lower_child(lc), higher_child(hc)
{
}

KDTree::KDNode *KDTree::treeify(std::vector<std::vector<float>> points, int depth)
{
    if (points.empty()) {
        return nullptr;
    }

    // Which axis to treeify on
    auto axis = depth % points.at(0).size();

    std::vector<std::vector<float>> lower_than_points, greater_than_points;

    auto selected_point = points.at(0);
    points.erase(points.begin());

    for (const auto &point : points) {
        if (point.at(axis) >= selected_point.at(axis)) {
            greater_than_points.push_back(point);
        } else {
            lower_than_points.push_back(point);
        }
    }

    return new KDNode(selected_point, treeify(lower_than_points, depth + 1), treeify(greater_than_points, depth + 1));

    /*
     *     // Sort point list and choose median as pivot element
    select median by axis from pointList;

    // Create node and construct subtree
    node.location := median;
    node.leftChild := kdtree(points in pointList before median, depth+1);
    node.rightChild := kdtree(points in pointList after median, depth+1);
    return node;
     */
}

KDTree::KDNode *KDTree::getRoot()
{
    return root_node;
}
