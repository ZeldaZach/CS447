//
// Created by Zachary Halpern on 2019-03-06.
//

#include "KDTree.h"
#include <algorithm>
#include <cmath>
#include <iostream>

KDTree::KDTree(std::vector<std::vector<float>> points) : root_node(buildTree(std::move(points), 0))
{
}

KDTree::~KDTree()
{
    while (root_node != nullptr) {
        deleteNode(root_node, root_node->point, 0);
    }

    delete root_node;
}

std::vector<float> KDTree::getNearestNeighbor(std::vector<float> input)
{
    KDTree::Node *input_node = new KDTree::Node(std::move(input), nullptr, nullptr);
    auto result = getNearestNeighbor(input_node, getRoot(), getRoot(), 0);

    delete input_node;
    return result->point;
}

KDTree::Node::Node(std::vector<float> p, KDTree::Node *lc, KDTree::Node *hc)
    : point(std::move(p)), lower_child(lc), higher_child(hc)
{
}

KDTree::Node *KDTree::buildTree(std::vector<std::vector<float>> points, unsigned long depth)
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

    return new Node(selected_point, buildTree(lower_points, depth + 1), buildTree(higher_points, depth + 1));
}

KDTree::Node *KDTree::getRoot()
{
    return root_node;
}

KDTree::Node *
KDTree::getNearestNeighbor(KDTree::Node *input, KDTree::Node *root, KDTree::Node *best, unsigned long depth)
{
    // End of tree
    if (!root) {
        return best;
    }

    // Cycle of dimensions
    depth %= input->point.size();

    // Best on the appropriate side
    KDTree::Node *check_point;
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

float KDTree::euclidianDistance(const std::vector<float> &p1, const std::vector<float> &p2)
{
    float value = 0;
    for (unsigned long i = 0; i < p1.size(); i++) {
        value += std::pow(p1.at(i) - p2.at(i), 2.0);
    }

    return std::sqrt(value);
}

KDTree::Node *KDTree::deleteNode(KDTree::Node *root, std::vector<float> point, unsigned long depth)
{
    if (!root) {
        return nullptr;
    }

    depth %= point.size();

    if (root->point == point) {
        if (root->higher_child) {
            Node *minimum_node = findMinimum(root->higher_child, depth);
            root->point = minimum_node->point;
            root->higher_child = deleteNode(root->higher_child, minimum_node->point, depth + 1);
        } else if (root->lower_child) {
            Node *minimum_node = findMinimum(root->lower_child, depth);
            root->point = minimum_node->point;
            root->higher_child = deleteNode(root->lower_child, minimum_node->point, depth + 1);
        } else {
            delete root;
            return nullptr;
        }

        return root;
    } else {
        if (root->point.at(depth) < point.at(depth)) {
            return deleteNode(root->higher_child, point, depth + 1);
        } else {
            return deleteNode(root->lower_child, point, depth + 1);
        }
    }
}

KDTree::Node *KDTree::findMinimum(Node *root, unsigned long depth)
{
    return findMinimum(root, depth, 0);
}

KDTree::Node *KDTree::findMinimum(KDTree::Node *root, unsigned long dimension, unsigned long depth)
{
    if (!root) {
        return nullptr;
    }

    depth %= root->point.size();

    if (depth == dimension) {
        if (!root->lower_child) {
            return root;
        }

        return findMinimum(root->lower_child, dimension, depth + 1);
    }

    Node *min_lower = findMinimum(root->lower_child, dimension, depth + 1);
    Node *min_higher = findMinimum(root->higher_child, dimension, depth + 1);

    if (!min_lower) {
        return min_higher;
    } else if (!min_higher) {
        return min_lower;
    }

    for (unsigned long i = 0; i < root->point.size(); i++) {
        if (min_lower->point.at(i) < min_higher->point.at(i)) {
            return min_lower;
        }

        if (min_lower->point.at(i) > min_higher->point.at(i)) {
            return min_higher;
        }
    }

    return min_lower;
}
