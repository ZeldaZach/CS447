//
// Created by Zachary Halpern on 2019-03-04.
//

#include "KNearestNeighbors.h"
#include "KDTree.h"
#include <assert.h>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

KNearestNeighbors::KNearestNeighbors() : tree(nullptr)
{
}

KNearestNeighbors::~KNearestNeighbors() = default;

void KNearestNeighbors::readFile(char *file_path)
{
    int fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        int en = errno;
        std::cerr << "Couldn't open " << file_path << ": " << strerror(en) << "." << std::endl;
        exit(2);
    }

    // Get the actual size of the file.
    struct stat sb
    {
    };
    int rv = fstat(fd, &sb);
    assert(rv == 0);

    // Use some flags that will hopefully improve performance.
    void *vp = mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (vp == MAP_FAILED) {
        int en = errno;
        fprintf(stderr, "mmap() failed: %s\n", strerror(en));
        exit(3);
    }
    char *file_mem = (char *)vp;

    // Tell the kernel that it should evict the pages as soon as possible.
    rv = madvise(vp, sb.st_size, MADV_SEQUENTIAL | MADV_WILLNEED);
    assert(rv == 0);

    rv = close(fd);
    assert(rv == 0);

    // Prefix to print before every line, to improve readability.
    std::string pref("    ");

    /*
     * Read file type string.
     */
    auto n = static_cast<unsigned long>(strnlen(file_mem, 8));
    std::string file_type(file_mem, n);

    // Start to read data, skip the file type string.
    Reader reader{file_mem + 8};

    if (file_type == "TRAINING") {
        uint64_t id;
        uint64_t n_points;
        uint64_t n_dims;

        reader >> id >> n_points >> n_dims;

        // std::vector<std::vector<float>> points;
        for (std::uint64_t i = 0; i < n_points; i++) {
            std::vector<float> dimensions;

            for (std::uint64_t j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                dimensions.push_back(f);
            }

            this->points.push_back(dimensions);
        }

        // Create the KD-Tree of training data
        this->points_node = new TrainingNode(std::string(file_type), id, n_points, n_dims);
        createTree();

    } else if (file_type == "QUERY") {
        uint64_t id;
        uint64_t n_queries;
        uint64_t n_dims;
        uint64_t n_neighbors;

        reader >> id >> n_queries >> n_dims >> n_neighbors;

        for (std::uint64_t i = 0; i < n_queries; i++) {
            std::vector<float> dimensions;

            for (std::uint64_t j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                dimensions.push_back(f);
            }

            this->queries.push_back(dimensions);
        }

        this->query_node = new QueryNode(std::string(file_type), id, n_queries, n_neighbors);

    } else if (file_type == "RESULT") {

        uint64_t training_id;
        uint64_t query_id;
        uint64_t result_id;
        uint64_t n_queries;
        uint64_t n_dims;
        uint64_t n_neighbors;

        reader >> training_id >> query_id >> result_id >> n_queries >> n_dims >> n_neighbors;

        std::cout << pref << "Training file ID: " << std::hex << std::setw(16) << std::setfill('0') << training_id
                  << std::dec << std::endl;
        std::cout << pref << "Query file ID: " << std::hex << std::setw(16) << std::setfill('0') << query_id << std::dec
                  << std::endl;
        std::cout << pref << "Result file ID: " << std::hex << std::setw(16) << std::setfill('0') << result_id
                  << std::dec << std::endl;
        std::cout << pref << "Number of queries: " << n_queries << std::endl;
        std::cout << pref << "Number of dimensions: " << n_dims << std::endl;
        std::cout << pref << "Number of neighbors returned for each query: " << n_neighbors << std::endl;
        return;
        for (std::uint64_t i = 0; i < n_queries; i++) {
            std::cout << pref << "Result " << i << ": ";
            for (std::uint64_t j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                std::cout << std::fixed << std::setprecision(6) << std::setw(15) << std::setfill(' ') << f;
                // Add comma.
                if (j < n_dims - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }

    } else {
        std::cerr << "Unknown file type: " << file_type << std::endl;
        exit(2);
    }

    rv = munmap(file_mem, sb.st_size);
    assert(rv == 0);
}

std::string KNearestNeighbors::generateAndWriteResults(char *file_path)
{
    const char *random_id = std::to_string(getRandomData()).c_str();

    std::string output_path = std::string(file_path) + "_" + std::string(random_id) + ".dat";

    if (fileExists(output_path.c_str())) {
        std::cerr << "Output file exists, renaming it" << std::endl;
        if (std::rename(output_path.c_str(), (output_path + std::to_string(std::time(nullptr))).c_str()) < 0) {
            std::cerr << "Failed to rename: " << strerror(errno) << std::endl;
            exit(1);
        }
    }

    std::cout << "FILE" << output_path << std::endl;
    std::ofstream out_file(output_path.c_str(), std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "Unable to open output file" << std::endl;
        exit(1);
    }

    // HEADER
    out_file.write("RESULT\0\0", 8);
    out_file.write(reinterpret_cast<char *>(&(this->points_node->file_id)), sizeof(this->points_node->file_id));
    out_file.write(reinterpret_cast<char *>(&(this->query_node->file_id)), sizeof(this->query_node->file_id));
    out_file.write(random_id, sizeof(random_id));
    out_file.write(reinterpret_cast<char *>(&(this->query_node->queries)), sizeof(this->query_node->queries));
    out_file.write(reinterpret_cast<char *>(&(this->points_node->dimensions)), sizeof(this->points_node->dimensions));
    out_file.write(reinterpret_cast<char *>(&(this->query_node->neighbors)), sizeof(this->query_node->neighbors));

    // BODY
    for (const auto &test_point : getQueries()) {
        auto result = getNearestNeighbor(test_point);
        for (auto dim : result) {
            out_file.write(reinterpret_cast<char *>(&dim), sizeof(dim));
        }
    }

    out_file.close();
    return output_path;
}

bool KNearestNeighbors::fileExists(const char *file_path)
{
    std::ifstream file(file_path);
    return static_cast<bool>(file);
}

void KNearestNeighbors::createTree()
{
    tree = new KDTree(points);
}

std::vector<float> KNearestNeighbors::getNearestNeighbor(std::vector<float> input)
{
    return tree->getNearestNeighbor(std::move(input));
}

unsigned int KNearestNeighbors::getRandomData()
{
    union
    {
        unsigned int value;
        char cs[sizeof(unsigned int)];
    } u{};

    std::ifstream random_file("/dev/urandom", O_RDONLY);
    random_file.read(u.cs, sizeof(u.cs));
    random_file.close();

    return u.value;
}