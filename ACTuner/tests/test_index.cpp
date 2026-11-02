#include "hnswlib/hnswlib.h"
#include "hnswlib/utils.h"
#include <iostream>
#include <fstream>
#include <chrono>
int main(int argc, char **argv)
{
    if (argc != 6)
    {
        std::cout << "Usage:" << argv[0] << "data_file M ef_construction hnsw_path log_path" << std::endl;
        exit(-1);
    }
    float *data = nullptr;
    size_t dim, max_elements;
    utils::load_data(argv[1], data, max_elements, dim);

    auto start1 = std::chrono::high_resolution_clock::now();
    
    auto M = atoi(argv[2]);
    auto ef_construction = atoi(argv[3]);
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    alg_hnsw->addPoint(data, 0);

#pragma omp parallel for
    for (int i = 1; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }
    
    std::string hnsw_path = argv[4];
    alg_hnsw->saveIndex(hnsw_path);
    delete[] data;
    delete alg_hnsw;
    auto end1 = std::chrono::high_resolution_clock::now();
    double dura1 = std::chrono::duration<double>(end1 - start1).count();
    std::string filename = std::string(argv[5]) + "_indexTime.csv";
    std::ofstream outfile(filename);
    if(!outfile.is_open()){
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }
    outfile << "M,ef_construction,Time(s)\n";
    outfile << M << "," << ef_construction << "," << dura1 << "\n";
    outfile.close();
    return 0;
}
