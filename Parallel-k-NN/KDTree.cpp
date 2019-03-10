//
// Created by Zachary Halpern on 2019-03-06.
//

#include "KDTree.h"
#include "AtomicWriter.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>

KDTree::KDTree(std::vector<std::vector<float>> points, unsigned long neighbors, unsigned long max_threads)
    : root_node(nullptr), how_many_neighbors(neighbors), max_threads(max_threads - 1)
{
    auto begin = std::chrono::steady_clock::now();
    root_node = buildTree(std::move(points), 0);
    auto end = std::chrono::steady_clock::now();

    AtomicWriter() << "Time to build tree: "
                   << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;
}

KDTree::~KDTree()
{
    deleteTree(root_node);
}

std::vector<std::vector<float>> KDTree::getNearestNeighbors(std::vector<float> input)
{
    auto priority_queue = new std::priority_queue<KDTree::queue_pair, std::vector<KDTree::queue_pair>, std::less<>>();

    // Get all nearest neighbors and put them into the queue
    KDTree::Node *input_node = new KDTree::Node(std::move(input), nullptr, nullptr);
    getNearestNeighbors(input_node, getRoot(), 0, priority_queue);
    delete input_node;

    std::vector<std::vector<float>> return_value;
    for (unsigned int i = 0; i < how_many_neighbors && !priority_queue->empty(); i++) {
        return_value.push_back(priority_queue->top().second->point);
        priority_queue->pop();
    }

    delete priority_queue;
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
        // AtomicWriter() << "Points empty!" << std::endl;
        if (promise) {
            promise->set_value_at_thread_exit(nullptr);
        }
        return nullptr;
    }

    // AtomicWriter() << "buildTree called with promise " << promise << std::endl;

    // Which axis to build tree on
    depth %= points.at(0).size();

    // Sort the points based on current axis
    std::sort(points.begin(), points.end(),
              [=](const std::vector<float> &v1, const std::vector<float> &v2) { return v1.at(depth) < v2.at(depth); });

    // Pluck out the middle element to balance our tree
    const size_t middle_index = points.size() / 2;
    const auto selected_point = points.at(middle_index);

    // Split across the middle
    const std::vector<std::vector<float>> lower_points(points.begin(), points.begin() + middle_index);
    const std::vector<std::vector<float>> higher_points(points.begin() + middle_index + 1, points.end());

    std::promise<Node *> lower, higher;
    std::future<Node *> lower_future = lower.get_future(), higher_future = higher.get_future();
    std::vector<std::thread> threads;
    bool lower_active = false, higher_active = false;

    if (thread_count < max_threads) {
        ++thread_count;
        // AtomicWriter() << "Adding lower thread" << std::endl;
        threads.emplace_back(&KDTree::buildTree, this, lower_points, depth + 1, &lower);
        lower_active = true;
    }

    if (thread_count < max_threads) {
        ++thread_count;
        // AtomicWriter() << "Adding upper thread" << std::endl;
        threads.emplace_back(&KDTree::buildTree, this, higher_points, depth + 1, &higher);
        higher_active = true;
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            // AtomicWriter() << "Join Thread:" << thread.get_id() << std::endl;
            thread.join();
        }
    }

    Node *lower_node, *higher_node;
    // AtomicWriter() << "Checking if valid lower" << std::endl;
    if (lower_active) {
        lower_node = lower_future.get();
        --thread_count;
    } else {
        // AtomicWriter() << "Building lower in thread" << std::endl;
        lower_node = buildTree(lower_points, depth + 1, nullptr);
    }

    // AtomicWriter() << "Checking if valid higher" << std::endl;
    if (higher_active) {
        higher_node = higher_future.get();
        --thread_count;
    } else {
        // AtomicWriter() << "Building higher in thread" << std::endl;
        higher_node = buildTree(higher_points, depth + 1, nullptr);
    }

    if (promise) {
        // AtomicWriter() << "Promise found, setting value: " << lower_node << ":" << higher_node << std::endl;
        promise->set_value_at_thread_exit(new Node(selected_point, lower_node, higher_node));
        return nullptr;
    }

    // AtomicWriter() << "New Node made : " << vectorToString(selected_point) << std::endl;
    return new Node(selected_point, lower_node, higher_node);
}

KDTree::Node *KDTree::getRoot()
{
    return root_node;
}

bool KDTree::pruneAwayResults(KDTree::Node *input,
                              KDTree::Node *root,
                              unsigned long depth,
                              std::priority_queue<queue_pair, std::vector<queue_pair>, std::less<>> *priority_queue)
{
    // Check the furthest, best, point in the queue
    auto furthest_best_point = priority_queue->top();

    // If the distance perpendicular is further than the worst point, no points in
    // its tree can be any closer, so we can prune them away.
    auto root_perpendicular_dist = euclidianDistance({input->point.at(depth)}, {root->point.at(depth)});
    return (root_perpendicular_dist > furthest_best_point.first);
}

void KDTree::getNearestNeighbors(KDTree::Node *input,
                                 KDTree::Node *root,
                                 unsigned long depth,
                                 std::priority_queue<queue_pair, std::vector<queue_pair>, std::less<>> *priority_queue)
{
    if (!root) {
        return;
    }

    // Cycle of dimensions
    depth %= input->point.size();

    // AtomicWriter() << vectorToString(root->point) << std::endl;
    auto e_dist = euclidianDistance(root->point, input->point);

    // Add the root to our queue, and remove the worst contender
    priority_queue->push(std::make_pair<>(e_dist, root));
    if (priority_queue->size() > how_many_neighbors) {
        priority_queue->pop();
    }

    if (input->point.at(depth) < root->point.at(depth)) {
        getNearestNeighbors(input, root->lower_child, depth + 1, priority_queue);
        if (!pruneAwayResults(input, root, depth, priority_queue)) {
            getNearestNeighbors(input, root->higher_child, depth + 1, priority_queue);
        }
    } else {
        getNearestNeighbors(input, root->higher_child, depth + 1, priority_queue);
        if (!pruneAwayResults(input, root, depth, priority_queue)) {
            getNearestNeighbors(input, root->lower_child, depth + 1, priority_queue);
        }
    }
}

float KDTree::euclidianDistance(const std::vector<float> &p1, const std::vector<float> &p2)
{
    if (p1.size() != p2.size()) {
        AtomicWriter() << "Invalid calling of euclidianDistance: p1.size() = " << p1.size()
                       << ", p2.size() = " << p2.size() << std::endl;
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
