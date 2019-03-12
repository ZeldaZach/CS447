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
public: // External methods
    explicit KNearestNeighbors(unsigned long, const char *, const char *, const char *);
    ~KNearestNeighbors();
    void readFile(const char *);
    void generateTree();
    void generateAndWriteResults(const char *);
    std::vector<std::vector<float>> getNearestNeighbors(std::vector<float>);
    void runQueries(const std::vector<std::vector<float>> &queries,
                    std::promise<std::vector<std::vector<float>>> *promise);

private: // Internal methods
    bool fileExists(const char *);
    unsigned int getRandomData();
    template <typename T> inline void binary_write(std::ostream &, const T &);

private: // Internal structs
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
        Node(std::string ft, unsigned long fid) : file_type(std::move(ft)), file_id(fid)
        {
        }

        std::string file_type;
        unsigned long file_id;
    };

    struct TrainingNode : public Node
    {
        TrainingNode(std::string ft, unsigned long fid, unsigned long p, unsigned long d)
            : Node(std::move(ft), fid), points(p), dimensions(d)
        {
        }

        unsigned long points;
        unsigned long dimensions;
    };

    struct QueryNode : public Node
    {
        QueryNode(std::string ft, unsigned long fid, unsigned long q, unsigned long n)
            : Node(std::move(ft), fid), queries(q), neighbors(n)
        {
        }

        unsigned long queries;
        unsigned long neighbors;
    };

private: // Internal variables
    std::vector<std::vector<float>> points;
    std::vector<std::vector<float>> queries;
    KDTree *tree;
    TrainingNode *points_node;
    QueryNode *query_node;
    unsigned long thread_count;
};

#endif // PARALLEL_K_NN_KNEARESTNEIGHBORS_H
