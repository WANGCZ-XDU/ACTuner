#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include "hnswlib.h"
namespace utils {
const int REPEAT_COUNT = 5;

void load_data(char* filename, float*& data, size_t& num, size_t& dim){
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){
        std::cerr << "open file error!" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)fsize / (dim + 1) / 4;
    #define DATA_ALIGN_FACTOR 16
    unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR * DATA_ALIGN_FACTOR;
    data = new float[(size_t)num * (size_t)new_dim];
    memset(data, 0, sizeof(float) * (size_t)num * (size_t)new_dim);
    in.seekg(0, std::ios::beg);
    for(size_t i = 0; i < num; i++){
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * new_dim), dim * 4);
    }
    in.close();
    dim = new_dim;
}

void load_ivecs(const char *filename, std::vector<std::vector<unsigned>> &result)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "open file error" << std::endl;
    }
    unsigned K;
    in.read((char *)&K, sizeof(unsigned));
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    unsigned num = (unsigned)(fsize / (K + 1) / 4);
    in.seekg(0, std::ios::beg);
    result.reserve(num);
    result.resize(num);
    unsigned kk = (K + 3) / 4 * 4;
    for (unsigned i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        result[i].resize(K);
        result[i].reserve(kk);
        in.read((char *)result[i].data(), K * sizeof(unsigned));
    }
    in.close();
}
struct QueryResult
{
    double qps;
    float recall;
    double avg_path_length;
    double recall_time;
};
float ComputeRecall(const std::vector<std::vector<unsigned>> &gtrue,
                    const std::vector<std::vector<unsigned>> &results,
                    size_t K)
{
    unsigned query_num = results.size();
    double total_recall = 0.0;
    for (unsigned i = 0; i < query_num; i++)
    {
        unsigned matches = 0;
        std::unordered_set<unsigned> gtrue_set(gtrue[i].begin(), gtrue[i].begin() + K);
        for (size_t j = 0; j < K; j++)
        {
            if (gtrue_set.find(results[i][j]) != gtrue_set.end())
            {
                matches++;
            }
        }
        total_recall += static_cast<double>(matches) / K;
    }
    return query_num > 0 ? total_recall / query_num : 0.0f;
}
void SearchMultiL(hnswlib::HierarchicalNSW<float> *hsnw_alg,
                         const float *query_data, size_t query_num,
                         const std::vector<std::vector<unsigned>> &gtrue,
                         const std::vector<unsigned> &L_values,
                         unsigned k,
                         unsigned dim,
                         std::vector<QueryResult> &results)
{
    if (results.size() != L_values.size())
    {
        results.resize(L_values.size());
    }
    for(int p = 0; p < L_values.size(); p++)
    {
        unsigned L = L_values[p];
        hsnw_alg->setEf(L);
        double total_time = 0.0;
        std::vector<std::vector<unsigned>> search_esults(query_num, std::vector<unsigned>(k));
        
        for (unsigned j = 0; j < query_num; j++)
        {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hsnw_alg->searchKnn(query_data + j * dim, k, nullptr);
            for (unsigned i = 0; i < k; i++)
            {
                if (!result.empty())
                {
                    search_esults[j][i] = result.top().second;
                    result.pop();
                }
                else
                {
                    search_esults[j][i] = 0; 
                }
            }
        }
        results[p].recall = ComputeRecall(gtrue, search_esults, k);
        auto s = std::chrono::high_resolution_clock::now();
        for (unsigned i = 0; i < REPEAT_COUNT; i++)
        {
            for (unsigned j = 0; j < query_num; j++)
            {
                hsnw_alg->searchKnn(query_data + j * dim, k, nullptr);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        total_time = std::chrono::duration<double>(e - s).count();
        results[p].qps = query_num / (total_time / (REPEAT_COUNT));
        results[p].avg_path_length = 0.0;
    }
}
void SearchMultiL_omp(hnswlib::HierarchicalNSW<float> *hsnw_alg,
                  const float *query_data, size_t query_num,
                  const std::vector<std::vector<unsigned>> &gtrue,
                  const std::vector<unsigned> &L_values,
                  unsigned k,
                  unsigned dim,
                  std::vector<QueryResult> &results)
{
    if (results.size() != L_values.size())
    {
        results.resize(L_values.size());
    }
    for (int p = 0; p < L_values.size(); p++)
    {
        unsigned L = L_values[p];
        hsnw_alg->setEf(L);
        double total_time = 0.0;
        std::vector<std::vector<unsigned>> search_results(query_num, std::vector<unsigned>(k));

#pragma omp parallel for schedule(static, 100)
            for (unsigned j = 0; j < query_num; j++)
            {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result = hsnw_alg->searchKnn(query_data + j * dim, k, nullptr);
                for (unsigned i = 0; i < k; i++)
                {
                    if (!result.empty())
                    {
                        search_results[j][i] = result.top().second;
                        result.pop();
                    }
                    else
                    {
                        search_results[j][i] = 0; 
                    }
                }
            }
            auto recall_start = std::chrono::high_resolution_clock::now();
            results[p].recall = ComputeRecall(gtrue, search_results, k);
            auto recall_end = std::chrono::high_resolution_clock::now();
            
            results[p].recall_time = std::chrono::duration<double>(recall_end - recall_start).count();

            auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel
            {
                for (unsigned i = 0; i < REPEAT_COUNT; i++)
                {
                #pragma omp for schedule(static, 100)
                    for (unsigned j = 0; j < query_num; j++)
                    {
                        hsnw_alg->searchKnn(query_data + j * dim, k, nullptr);
                    }
                }
            }
            auto e = std::chrono::high_resolution_clock::now();
            total_time = std::chrono::duration<double>(e - s).count();
            results[p].qps = query_num / (total_time / (REPEAT_COUNT));
            results[p].avg_path_length = 0.0;
        }
}
}
