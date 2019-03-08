//
// Created by Zachary Halpern on 2019-03-04.
//

#include "KNearestNeighbors.h"
#include "KDTree.h"
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

KNearestNeighbors::KNearestNeighbors() : tree(nullptr), points_node(nullptr), query_node(nullptr)
{
}

KNearestNeighbors::~KNearestNeighbors()
{
    delete points_node;
    delete query_node;
    delete tree;
}

void KNearestNeighbors::readFile(char *file_path)
{
    int fd = open(file_path, O_RDONLY);
    if (fd < 0) {
        int en = errno;
        std::cerr << "Couldn't open " << file_path << ": " << strerror(en) << "." << std::endl;
        exit(2);
    }

    // Get the actual size of the file.
    struct stat sb;
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
        unsigned long id, n_points, n_dims;

        reader >> id >> n_points >> n_dims;

        // std::vector<std::vector<float>> points;
        for (unsigned long i = 0; i < n_points; i++) {
            std::vector<float> dimensions;

            for (unsigned long j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                dimensions.push_back(f);
            }

            this->points.push_back(dimensions);
        }

        // Create the KD-Tree of training data
        this->points_node = new TrainingNode(std::string(file_type), id, n_points, n_dims);
    } else if (file_type == "QUERY") {
        unsigned long id, n_queries, n_dims, n_neighbors;
        reader >> id >> n_queries >> n_dims >> n_neighbors;

        for (unsigned long i = 0; i < n_queries; i++) {
            std::vector<float> dimensions;

            for (unsigned long j = 0; j < n_dims; j++) {
                float f;
                reader >> f;
                dimensions.push_back(f);
            }

            this->queries.push_back(dimensions);
        }

        this->query_node = new QueryNode(std::string(file_type), id, n_queries, n_neighbors);
    } else if (file_type == "RESULT") {
        unsigned long training_id, query_id, result_id, n_queries, n_dims, n_neighbors;
        reader >> training_id >> query_id >> result_id >> n_queries >> n_dims >> n_neighbors;

        std::cout << pref << "Training file ID: " << std::setw(16) << std::setfill('0') << training_id << std::dec
                  << std::endl;
        std::cout << pref << "Query file ID: " << std::setw(16) << std::setfill('0') << query_id << std::dec
                  << std::endl;
        std::cout << pref << "Result file ID: " << std::setw(16) << std::setfill('0') << result_id << std::dec
                  << std::endl;
        std::cout << pref << "Number of queries: " << n_queries << std::endl;
        std::cout << pref << "Number of dimensions: " << n_dims << std::endl;
        std::cout << pref << "Number of neighbors returned for each query: " << n_neighbors << std::endl;

        for (unsigned long i = 0; i < n_queries; i++) {
            std::cout << pref << "Result " << i << ": ";
            for (unsigned long j = 0; j < n_dims; j++) {
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
    unsigned long random_id = getRandomData();

    std::string output_path = std::string(file_path) + "_" + std::to_string(random_id) + ".dat";

    if (fileExists(output_path.c_str())) {
        std::cerr << "Output file exists, renaming it" << std::endl;
        if (std::rename(output_path.c_str(), (output_path + std::to_string(std::time(nullptr))).c_str()) < 0) {
            std::cerr << "Failed to rename: " << strerror(errno) << std::endl;
            exit(1);
        }
    }

    std::ofstream out_file(output_path.c_str(), std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "Unable to open output file" << std::endl;
        exit(1);
    }

    // HEADER
    out_file.write("RESULT\0\0", 8);
    out_file.write(reinterpret_cast<char *>(&this->points_node->file_id), 8);
    out_file.write(reinterpret_cast<char *>(&this->query_node->file_id), 8);
    out_file.write(reinterpret_cast<char *>(&random_id), 8);
    out_file.write(reinterpret_cast<char *>(&this->query_node->queries), 8);
    out_file.write(reinterpret_cast<char *>(&this->points_node->dimensions), 8);
    out_file.write(reinterpret_cast<char *>(&this->query_node->neighbors), 8);

    // BODY
    for (const auto &query_point : getQueries()) {
        std::vector<std::vector<float>> neighbors = getNearestNeighbors(query_point);
        for (const auto &neighbor : neighbors) {
            for (float point : neighbor) {
                char *dim_data = reinterpret_cast<char *>(&point);
                std::cout << *reinterpret_cast<float *>(dim_data) << "\t(" << sizeof(dim_data) << ")" << std::endl;
                out_file.write(dim_data, 8);
            }
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

void KNearestNeighbors::generateTree()
{
    tree = new KDTree(points, query_node->neighbors);
}

std::vector<std::vector<float>> KNearestNeighbors::getNearestNeighbors(std::vector<float> input)
{
    return tree->getNearestNeighbors(std::move(input));
}

unsigned int KNearestNeighbors::getRandomData()
{
    unsigned char buffer[8];
    int fd = open("/dev/urandom", O_RDONLY);
    read(fd, buffer, 8);
    close(fd);

    return *reinterpret_cast<unsigned int *>(&buffer);
}
