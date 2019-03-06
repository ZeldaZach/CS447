//
// Created by Zachary Halpern on 2019-03-06.
//

#include "KDTree.h"
#include <algorithm>
#include <cmath>

KDTree::KDTree(std::vector<std::vector<float>> points)
{
    root_node = treeify(std::move(points), 0);
}

KDTree::~KDTree() = default;

KDTree::KDNode::KDNode(std::vector<float> p, KDTree::KDNode *lc, KDTree::KDNode *hc)
    : point(std::move(p)), lower_child(lc), higher_child(hc)
{
}

KDTree::KDNode *KDTree::treeify(std::vector<std::vector<float>> points, unsigned long depth)
{
    if (points.empty()) {
        return nullptr;
    }

    // Which axis to treeify on
    depth = depth % points.at(0).size();

    // Sort the points
    std::sort(points.begin(), points.end(),
              [=](const std::vector<float> &v1, const std::vector<float> &v2) { return v1.at(depth) < v2.at(depth); });

    // Pluck out the middle element to balance our tree
    size_t middle_index = points.size() / 2;
    auto selected_point = points.at(middle_index);

    std::vector<std::vector<float>> lower_than_points(points.begin(), points.begin() + middle_index);
    std::vector<std::vector<float>> greater_than_points(points.begin() + middle_index + 1, points.end());

    return new KDNode(selected_point, treeify(lower_than_points, depth + 1), treeify(greater_than_points, depth + 1));
}

KDTree::KDNode *KDTree::getNearestNeighbor(std::vector<float> input)
{
    KDTree::KDNode *input_node = new KDTree::KDNode(std::move(input), nullptr, nullptr);

    auto root = getRoot();
    root->e_distance = getEuclidianDistance(root->point, input_node->point);

    auto result = getNearestNeighbor(input_node, root, root, 0);

    delete input_node;
    return result;
}

KDTree::KDNode *
KDTree::getNearestNeighbor(KDTree::KDNode *input, KDTree::KDNode *root, KDTree::KDNode *best, unsigned long depth)
{
    if (!root) {
        return best;
    }

    auto new_e_distance = getEuclidianDistance(root->point, input->point);
    if (new_e_distance < best->e_distance) {
        // Reset old best
        best->e_distance = 0;

        // Mark root as best
        root->e_distance = new_e_distance;
        best = root;
    }

    depth = depth % input->point.size();

    auto lower_best = getNearestNeighbor(input, root->lower_child, best, depth + 1);
    auto higher_best = getNearestNeighbor(input, root->higher_child, best, depth + 1);

    auto lower_e_dist = getEuclidianDistance(lower_best->point, input->point);
    auto higher_e_dist = getEuclidianDistance(higher_best->point, input->point);

    if (lower_e_dist < higher_e_dist) {
        if (best->e_distance < lower_e_dist) {
            return best;
        } else {
            return lower_best;
        }
    } else {
        if (best->e_distance < higher_e_dist) {
            return best;
        } else {
            return higher_best;
        }
    }
}

KDTree::KDNode *KDTree::getRoot()
{
    return root_node;
}

float KDTree::getEuclidianDistance(const std::vector<float> &p1, const std::vector<float> &p2)
{
    float value = 0;
    for (unsigned long i = 0; i < p1.size(); i++) {
        value += std::powf(p1.at(i) - p2.at(i), 2.0);
    }

    return std::sqrt(value);
}
