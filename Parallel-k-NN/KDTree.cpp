//
// Created by Zachary Halpern on 2019-03-06.
//

#include "KDTree.h"
#include <algorithm>
#include <cmath>
#include <iostream>

KDTree::KDTree(std::vector<std::vector<float>> points) : root_node(build_tree(std::move(points), 0))
{
}

KDTree::~KDTree()
{
    delete root_node;
}

KDTree::KDNode::KDNode(std::vector<float> p, KDTree::KDNode *lc, KDTree::KDNode *hc)
    : point(std::move(p)), lower_child(lc), higher_child(hc)
{
}

KDTree::KDNode *KDTree::build_tree(std::vector<std::vector<float>> points, unsigned long depth)
{
    if (points.empty()) {
        return nullptr;
    }

    // Which axis to build tree on
    depth %= points.at(0).size();

    // Sort the points based on current axis
    std::sort(points.begin(), points.end(),
              [=](const std::vector<float> &v1, const std::vector<float> &v2) { return v1.at(depth) < v2.at(depth); });

    // Pluck out the middle element to balance our tree
    size_t middle_index = points.size() / 2;
    auto selected_point = points.at(middle_index);

    std::vector<std::vector<float>> lower_points(points.begin(), points.begin() + middle_index);
    std::vector<std::vector<float>> higher_points(points.begin() + middle_index + 1, points.end());

    return new KDNode(selected_point, build_tree(lower_points, depth + 1), build_tree(higher_points, depth + 1));
}

std::vector<float> KDTree::getNearestNeighbor(std::vector<float> input)
{
    KDTree::KDNode *input_node = new KDTree::KDNode(std::move(input), nullptr, nullptr);
    auto result = getNearestNeighbor(input_node, getRoot(), getRoot(), 0);

    delete input_node;
    return result->point;
}

KDTree::KDNode *
KDTree::getNearestNeighbor(KDTree::KDNode *input, KDTree::KDNode *root, KDTree::KDNode *best, unsigned long depth)
{
    // End of tree
    if (!root) {
        return best;
    }

    // Cycle of dimensions
    depth %= input->point.size();

    // Best on the appropriate side
    KDTree::KDNode *check_point;
    if (input->point.at(depth) < root->point.at(depth)) {
        check_point = getNearestNeighbor(input, root->lower_child, best, depth + 1);
    } else {
        check_point = getNearestNeighbor(input, root->higher_child, best, depth + 1);
    }

    // Distance between the root and the input
    auto e_dist = euclidianDistance(root->point, input->point);

    if (check_point) {
        // Distance between "best" found point and input
        auto checkpoint_e_dist = euclidianDistance(check_point->point, input->point);

        // Return whichever one is closer
        return checkpoint_e_dist < e_dist ? check_point : root;
    }

    // This should _never_ happen
    std::cerr << "This should never happen: Check_point is null" << std::endl;
    return nullptr;
}

KDTree::KDNode *KDTree::getRoot()
{
    return root_node;
}

float KDTree::euclidianDistance(const std::vector<float> &p1, const std::vector<float> &p2)
{
    float value = 0;
    for (unsigned long i = 0; i < p1.size(); i++) {
        value += std::powf(p1.at(i) - p2.at(i), 2.0);
    }

    return std::sqrt(value);
}
