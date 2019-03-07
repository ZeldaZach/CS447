//
// Created by Zachary Halpern on 2019-03-04.
//

#ifndef PARALLEL_K_NN_KNEARESTNEIGHBORS_H
#define PARALLEL_K_NN_KNEARESTNEIGHBORS_H

#include "KDTree.h"
#include <assert.h>
#include <cstring>
#include <string>
#include <vector>

class KNearestNeighbors
{
public:
    explicit KNearestNeighbors();
    ~KNearestNeighbors();
    void readFile(char *);
    std::string generateAndWriteResults(char *);
    std::vector<float> getNearestNeighbor(std::vector<float>);

    inline std::vector<std::vector<float>> getPoints() const
    {
        return points;
    }

    inline std::vector<std::vector<float>> getQueries() const
    {
        return queries;
    }

    KDTree *getTree()
    {
        return tree;
    }

private:
    bool fileExists(const char *);
    void createTree();
    unsigned int getRandomData();

private:
    class Reader
    {
    public:
        explicit Reader(const char *p) : ptr{p}
        {
        }
        template <typename T> Reader &operator>>(T &o)
        {
            // Assert alignment.
            assert(uintptr_t(ptr) % sizeof(T) == 0);
            o = *(T *)ptr;
            ptr += sizeof(T);
            return *this;
        }

    private:
        const char *ptr;
    };

    struct Node
    {
        Node(std::string ft, uint64_t fid) : file_type(std::move(ft)), file_id(fid)
        {
        }

        std::string file_type;
        uint64_t file_id;
    };

    struct TrainingNode : public Node
    {
        TrainingNode(std::string ft, uint64_t fid, uint64_t p, uint64_t d)
            : Node(std::move(ft), fid), points(p), dimensions(d)
        {
        }

        uint64_t points;
        uint64_t dimensions;
    };

    struct QueryNode : public Node
    {
        QueryNode(std::string ft, uint64_t fid, uint64_t q, uint64_t n)
            : Node(std::move(ft), fid), queries(q), neighbors(n)
        {
        }

        uint64_t queries;
        uint64_t neighbors;
    };

private:
    std::vector<std::vector<float>> points;
    std::vector<std::vector<float>> queries;
    TrainingNode *points_node;
    QueryNode *query_node;
    KDTree *tree;
};

#endif // PARALLEL_K_NN_KNEARESTNEIGHBORS_H
