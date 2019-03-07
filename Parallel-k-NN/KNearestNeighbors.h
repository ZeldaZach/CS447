//
// Created by Zachary Halpern on 2019-03-04.
//

#ifndef PARALLEL_K_NN_KNEARESTNEIGHBORS_H
#define PARALLEL_K_NN_KNEARESTNEIGHBORS_H

#include "KDTree.h"
#include <string>
#include <vector>
#include <assert.h>
#include <cstring>

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
        Node(std::string ft, uint64_t fid) : file_type(std::move(ft)), file_id(static_cast<int>(fid))
        {
        }

        std::string file_type;
        int file_id;
    };

    struct TrainingNode : public Node
    {
        TrainingNode(std::string ft, uint64_t fid, uint64_t p, uint64_t d)
            : Node(std::move(ft), fid), points(static_cast<int>(p)), dimensions(static_cast<int>(d))
        {
        }

        int points;
        int dimensions;
    };

    struct QueryNode : public Node
    {
        QueryNode(std::string ft, uint64_t fid, uint64_t q, uint64_t n)
            : Node(std::move(ft), fid), queries(static_cast<int>(q)), neighbors(static_cast<int>(n))
        {
        }

        int queries;
        int neighbors;
    };

private:
    std::vector<std::vector<float>> points;
    std::vector<std::vector<float>> queries;
    TrainingNode *points_node;
    QueryNode *query_node;
    KDTree *tree;
};

#endif // PARALLEL_K_NN_KNEARESTNEIGHBORS_H
