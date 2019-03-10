//
// Created by Zachary Halpern on 2019-03-06.
//

#ifndef PARALLEL_K_NN_KDTREE_H
#define PARALLEL_K_NN_KDTREE_H

#include <future>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

class KDTree
{
private:
    struct Node
    {
        explicit Node(std::vector<float>, Node *, Node *);
        std::vector<float> point;
        Node *lower_child, *higher_child;
    };

public:
    explicit KDTree(std::vector<std::vector<float>>, unsigned long, unsigned long max_threads);
    ~KDTree();
    std::vector<std::vector<float>> getNearestNeighbors(std::vector<float>);
    Node *getRoot();

private:
    KDTree::Node *buildTree(std::vector<std::vector<float>>, unsigned long, std::promise<Node *> *promise = nullptr);
    void getNearestNeighbors(KDTree::Node *input, KDTree::Node *root, unsigned long depth);
    float euclidianDistance(const std::vector<float> &, const std::vector<float> &);
    void deleteTree(Node *root);
    static std::string vectorToString(const std::vector<float> &);
    bool pruneAwayResults(KDTree::Node *input, KDTree::Node *root, unsigned long depth);
    typedef std::pair<float, Node *> queue_pair;

private:
    Node *root_node;
    std::priority_queue<queue_pair, std::vector<queue_pair>, std::less<>> priority_queue;
    unsigned long how_many_neighbors, max_threads;
    std::atomic<unsigned long> thread_count;
};

class AtomicWriter
{
    std::ostringstream st;
    std::ostream &stream;

public:
    AtomicWriter(std::ostream &s = std::cout) : stream(s)
    {
    }
    template <typename T> AtomicWriter &operator<<(T const &t)
    {
        st << t;
        return *this;
    }
    AtomicWriter &operator<<(std::ostream &(*f)(std::ostream &))
    {
        st << f;
        return *this;
    }
    ~AtomicWriter()
    {
        stream << st.str();
    }
};
#endif // PARALLEL_K_NN_KDTREE_H
