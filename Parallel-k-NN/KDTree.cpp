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
#include <utility>

/**
 * Constructor
 * @param points
 * @param neighbors
 * @param max_threads
 */
KDTree::KDTree(std::vector<std::vector<float>> *points, unsigned long neighbors, unsigned long max_threads)
    : root_node(nullptr), k_neighbors(neighbors), max_threads(2 * max_threads - 2)
{
    auto begin = std::chrono::steady_clock::now();
    root_node = buildTree(points->begin(), points->end(), 0);
    auto end = std::chrono::steady_clock::now();

    AtomicWriter() << "Time to build tree: "
                   << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
}

/**
 * Destructor
 */
KDTree::~KDTree()
{
    deleteTree(root_node);
}

/**
 * Public way to get the nearest neighbors that calls recursive operations
 * @param input
 * @return
 */
std::vector<std::vector<float>> KDTree::getNearestNeighbors(std::vector<float> input)
{
    auto priority_queue = new std::priority_queue<KDTree::queue_pair, std::vector<KDTree::queue_pair>, std::less<>>();

    // Get all nearest neighbors and put them into the queue
    KDTree::Node *input_node = new KDTree::Node(std::move(input), nullptr, nullptr);
    getNearestNeighbors(input_node, getRoot(), 0, priority_queue);
    delete input_node;

    std::vector<std::vector<float>> return_value;
    for (unsigned int i = 0; i < k_neighbors && !priority_queue->empty(); i++) {
        return_value.push_back(priority_queue->top().second->point);
        priority_queue->pop();
    }

    delete priority_queue;
    return return_value;
}

/**
 * Node Constructor
 * @param p
 * @param lc
 * @param hc
 */
KDTree::Node::Node(std::vector<float> p, KDTree::Node *lc, KDTree::Node *hc)
    : point(std::move(p)), lower_child(lc), higher_child(hc)
{
}

/**
 * Recurisve method to build the tree in parallel across N threads.
 * Passes iterators around for optimal speedups.
 * @param points_begin
 * @param points_end
 * @param depth
 * @param promise
 * @return
 */
KDTree::Node *KDTree::buildTree(std::vector<std::vector<float>>::iterator points_begin,
                                std::vector<std::vector<float>>::iterator points_end,
                                unsigned long depth,
                                std::promise<Node *> *promise)
{
    if (points_begin == points_end) {
        // Node has no children, so just let the parent know we're done
        if (promise) {
            promise->set_value_at_thread_exit(nullptr);
        }
        return nullptr;
    }

    // Which axis to build tree on
    depth %= (*points_begin).size();

    // Sort the points based on current axis
    std::sort(points_begin, points_end,
              [=](const std::vector<float> &v1, const std::vector<float> &v2) { return v1.at(depth) < v2.at(depth); });

    // Pluck out the middle element to balance our tree

    const size_t middle_index = (points_end - points_begin) / 2;
    const auto selected_point = points_begin + middle_index;

    /*
     * We can run up to max_threads at once, so we will spawn
     * 2*max_threads - 1 children to do build the tree in pieces
     * This ensures we never use more than max_threads * 100% cpu time
     */
    std::promise<Node *> lower, higher;
    std::future<Node *> lower_future = lower.get_future(), higher_future = higher.get_future();
    std::vector<std::thread> threads;
    bool lower_active = false, higher_active = false;

    if (thread_count < max_threads) {
        ++thread_count;
        threads.emplace_back(&KDTree::buildTree, this, points_begin, points_begin + middle_index, depth + 1, &lower);
        lower_active = true;
    }

    if (thread_count < max_threads) {
        ++thread_count;
        threads.emplace_back(&KDTree::buildTree, this, points_begin + middle_index + 1, points_end, depth + 1, &higher);
        higher_active = true;
    }

    Node *lower_node = nullptr, *higher_node = nullptr;

    // We want to build this node's subtree before we wait for its parallel parts to finish
    if (!lower_active) {
        lower_node = buildTree(points_begin, points_begin + middle_index, depth + 1, nullptr);
    }
    if (!higher_active) {
        higher_node = buildTree(points_begin + middle_index + 1, points_end, depth + 1, nullptr);
    }

    // Wait for the node's subtrees to finish before returning out
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.detach();
        }
    }
    threads.clear();

    // If we were waiting for a child, get its results
    if (lower_active) {
        lower_node = lower_future.get();
        --thread_count;
    }
    if (higher_active) {
        higher_node = higher_future.get();
        --thread_count;
    }

    // If we have a parent, let them know we're ready
    if (promise) {
        promise->set_value_at_thread_exit(new Node(*selected_point, lower_node, higher_node));
        return nullptr;
    }

    return new Node(*selected_point, lower_node, higher_node);
}

/**
 * Getter for the root of the tree
 * @return
 */
KDTree::Node *KDTree::getRoot()
{
    return root_node;
}

/**
 * See if a certain branch of the tree should be pruned as it cannot possibly
 * contain optimal nodes.
 * @param input
 * @param root
 * @param depth
 * @param priority_queue
 * @return
 */
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

/**
 * Recursive internal call to determine the K closest neighbors
 * to the input value. Incorporates pruning to avoid searching
 * bad branches.
 * @param input
 * @param root
 * @param depth
 * @param priority_queue
 */
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
    if (priority_queue->size() > k_neighbors) {
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

/**
 * Get the "Euclidian distance" for two points
 * NOTE: We do not square root for efficiency, and choose to work with the squares instead
 * @param p1
 * @param p2
 * @return
 */
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

/**
 * Post-order deletion of the tree
 * @param root
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

/**
 * Debug method to see the value of a node
 * @param v
 * @return
 */
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
