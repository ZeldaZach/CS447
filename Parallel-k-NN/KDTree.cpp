//
// Created by Zachary Halpern on 2019-03-06.
//

#include "KDTree.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>

KDTree::KDTree(std::vector<std::vector<float>> points, unsigned long neighbors, unsigned long max_threads)
    : root_node(nullptr), how_many_neighbors(neighbors), max_threads(max_threads)
{
    auto begin = std::chrono::steady_clock::now();
    root_node = buildTree(std::move(points), 0);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Time to build tree: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
              << "ns" << std::endl;

    exit(0);
}

KDTree::~KDTree()
{
    deleteTree(root_node);
}

std::vector<std::vector<float>> KDTree::getNearestNeighbors(std::vector<float> input)
{
    // Get all nearest neighbors and put them into the queue
    KDTree::Node *input_node = new KDTree::Node(std::move(input), nullptr, nullptr);
    getNearestNeighbors(input_node, getRoot(), 0);
    delete input_node;

    std::vector<std::vector<float>> return_value;

    for (unsigned int i = 0; i < how_many_neighbors && !priority_queue.empty(); i++) {
        return_value.push_back(priority_queue.top().second->point);
        priority_queue.pop();
    }

    while (!priority_queue.empty()) {
        priority_queue.pop();
    }

    // std::cout << "End" << std::endl;
    return return_value;
}

KDTree::Node::Node(std::vector<float> p, KDTree::Node *lc, KDTree::Node *hc)
    : point(std::move(p)), lower_child(lc), higher_child(hc)
{
}

KDTree::Node *
KDTree::buildTree(std::vector<std::vector<float>> points, unsigned long depth, std::promise<Node *> *promise)
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

    /*
    std::cout << "Using " << vectorToString(selected_point) << " with thread count: " << thread_count << "/"
              << max_threads << " with promise: " << promise << std::endl;
    */

    std::vector<std::vector<float>> lower_points(points.begin(), points.begin() + middle_index);
    std::vector<std::vector<float>> higher_points(points.begin() + middle_index + 1, points.end());

    Node *lower_node = nullptr, *higher_node = nullptr;
    std::promise<Node *> lower_promise, higher_promise;
    std::future<Node *> lower_future = lower_promise.get_future(), higher_future = higher_promise.get_future();
    // std::thread lower_thread, higher_thread;
    std::vector<std::thread> threads;
    bool lower_use_thread = false, higher_use_thread = false;

    if (thread_count < max_threads) {
        ++thread_count;
        threads.emplace_back(&KDTree::buildTree, this, lower_points, depth + 1, &lower_promise);
        lower_use_thread = true;
    } else {
        lower_node = buildTree(lower_points, depth + 1);
    }

    if (thread_count < max_threads) {
        ++thread_count;
        threads.emplace_back(std::thread(&KDTree::buildTree, this, higher_points, depth + 1, &higher_promise));
        higher_use_thread = true;
    } else {
        higher_node = buildTree(higher_points, depth + 1);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    if (lower_use_thread) {
        --thread_count;
        lower_node = lower_future.get();
    }

    if (higher_use_thread) {
        --thread_count;
        higher_node = higher_future.get();
    }

    if (promise) {
        promise->set_value(new Node(selected_point, lower_node, higher_node));
        return nullptr;
    }

    return new Node(selected_point, lower_node, higher_node);
}

KDTree::Node *KDTree::getRoot()
{
    return root_node;
}

bool KDTree::pruneAwayResults(KDTree::Node *input, KDTree::Node *root, unsigned long depth)
{
    // Check the furthest, best, point in the queue
    auto furthest_best_point = priority_queue.top();

    // If the distance perpendicular is further than the worst point, no points in
    // its tree can be any closer, so we can prune them away.
    auto root_perpendicular_dist = euclidianDistance({input->point.at(depth)}, {root->point.at(depth)});
    return (root_perpendicular_dist > furthest_best_point.first);
}

void KDTree::getNearestNeighbors(KDTree::Node *input, KDTree::Node *root, unsigned long depth)
{
    if (!root) {
        return;
    }

    // Cycle of dimensions
    depth %= input->point.size();

    // std::cout << vectorToString(root->point) << std::endl;
    auto e_dist = euclidianDistance(root->point, input->point);

    // Add the root to our queue, and remove the worst contender
    priority_queue.push(std::make_pair<>(e_dist, root));
    if (priority_queue.size() > how_many_neighbors) {
        priority_queue.pop();
    }

    if (input->point.at(depth) < root->point.at(depth)) {
        getNearestNeighbors(input, root->lower_child, depth + 1);
        if (!pruneAwayResults(input, root, depth)) {
            getNearestNeighbors(input, root->higher_child, depth + 1);
        }
    } else {
        getNearestNeighbors(input, root->higher_child, depth + 1);
        if (!pruneAwayResults(input, root, depth)) {
            getNearestNeighbors(input, root->lower_child, depth + 1);
        }
    }
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

    // return std::sqrt(value);
    return value;
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

std::string KDTree::vectorToString(const std::vector<float> &v)
{
    std::string ret("(");
    for (const float &p : v) {
        ret += std::to_string(p) + ", ";
    }
    ret.pop_back();
    ret.pop_back();
    ret += ")";
    return ret;
}
