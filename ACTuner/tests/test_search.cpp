#include "hnswlib/hnswlib.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include "hnswlib/utils.h"

int main(int argc, char** argv){
    
    if(argc != 6) {
        std::cout << "Usage: " << argv[0] << "datafile hnswfile queryfile gtthfile logfile" << std::endl;
        exit(-1);
    }
    
    size_t dim, max_elements;
    float* data = nullptr;
    utils::load_data(argv[1], data, max_elements, dim);
    auto hnsw_path = std::string(argv[2]);
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    size_t query_num = 0;
    float* query_data = nullptr;
    utils::load_data(argv[3], query_data, query_num, dim);
    std::vector<std::vector<unsigned>> gtrue;
    utils::load_ivecs(argv[4], gtrue);
    std::vector<unsigned> L_values = {10, 20, 30, 40, 50, 80, 100, 200, 300, 600, 1200, 2400}; 
    
    std::vector<utils::QueryResult> results = std::vector<utils::QueryResult>(L_values.size());
    utils::SearchMultiL_omp(alg_hnsw, query_data, query_num, gtrue, L_values, 10, dim, results);
    std::string filename = std::string(argv[5]) + "_performance.csv";
    std::ofstream log_file(filename);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << argv[5] << std::endl;
        return -1;
    }
    for (size_t i = 0; i < L_values.size(); ++i) {
        log_file << L_values[i] << "," 
                 << results[i].recall << "," 
                 << results[i].qps << std::endl;
    };
    log_file.close();
    return 0;
}