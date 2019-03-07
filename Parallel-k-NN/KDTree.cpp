//
// Created by Zachary Halpern on 2019-03-06.
//

#include "KDTree.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

KDTree::KDTree(std::vector<std::vector<float>> points, unsigned long neighbors)
    : root_node(buildTree(std::move(points), 0)), how_many_neighbors(neighbors)
{
}

KDTree::~KDTree()
{
    deleteTree(root_node);
}

std::vector<std::vector<float>> KDTree::getNearestNeighbors(std::vector<float> input)
{
    // Empty the queue before starting
    while (!priority_queue.empty()) {
        priority_queue.pop();
    }

    // Get all nearest neighbors and put them into the queue
    KDTree::Node *input_node = new KDTree::Node(std::move(input), nullptr, nullptr);
    getNearestNeighbors(input_node, getRoot(), 0);
    delete input_node;

    std::vector<std::vector<float>> return_value;

    for (unsigned int i = 0; i < how_many_neighbors && !priority_queue.empty(); i++) {
        return_value.push_back(priority_queue.top()->point);
        priority_queue.pop();
    }

    return return_value;
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

KDTree::Node *KDTree::getNearestNeighbors(KDTree::Node *input, KDTree::Node *root, unsigned long depth)
{
    // End of tree
    if (!root) {
        return priority_queue.top();
    }

    priority_queue.push(root);

    // Cycle of dimensions
    depth %= input->point.size();

    // Best on the appropriate side
    KDTree::Node *check_point;
    if (input->point.at(depth) < root->point.at(depth)) {
        check_point = getNearestNeighbors(input, root->lower_child, depth + 1);
    } else {
        check_point = getNearestNeighbors(input, root->higher_child, depth + 1);
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
    std::cerr << "This should never happen: check_point is null" << std::endl;
    return nullptr;
}

float KDTree::euclidianDistance(const std::vector<float> &p1, const std::vector<float> &p2)
{
    if (p1.size() != p2.size()) {
        std::cerr << "Invalid calling of euclidianDistance: p1.size() = " << p1.size() << ", p2.size() = " << p2.size()
                  << std::endl;
        exit(1);
    }

    float value = 0;
    for (unsigned long i = 0; i < p1.size(); i++) {
        value += std::pow(p1.at(i) - p2.at(i), 2.0);
    }

    return std::sqrt(value);
}

/*
 * Post Order deletion of the tree
 */
void KDTree::deleteTree(KDTree::Node *root)
{
    if (!root) {
        return;
    }

    if (root->lower_child) {
        deleteTree(root->lower_child);
    }

    if (root->higher_child) {
        deleteTree(root->higher_child);
    }

    delete root;
}

std::string vectorize(std::vector<float> v)
{
    std::string ret("(");

    for (auto const &p : v) {
        ret += std::to_string(p) + ",";
    }
    ret.pop_back();
    ret += ")";
    return ret;
}

void KDTree::postorder(Node *p, int indent)
{
    if (p) {
        if (p->higher_child) {
            postorder(p->higher_child, indent + 4);
        }
        if (indent) {
            std::cout << std::setw(indent) << ' ';
        }
        if (p->higher_child)
            std::cout << " /\n" << std::setw(indent) << ' ';
        std::cout << vectorize(p->point) << "\n ";
        if (p->lower_child) {
            std::cout << std::setw(indent) << ' ' << " \\\n";
            postorder(p->lower_child, indent + 4);
        }
    }
}
