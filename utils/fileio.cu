#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <queue>
#include <fstream>
#include <unordered_map>
#include <random>
#include <fileio.h>
#include <graph.h>
#include <node.h>

std::string op2String(Oper op) {
    switch (op) {
        case Oper::INPUT : return "INPUT";
        case Oper::MATMUL : return "MATMUL";
        case Oper::ADD : return "ADD";
        case Oper::ReLU : return "ReLU";
        case Oper::FUSED_MR : return "Fused Node (MatMul + ReLU)";
        case Oper::FUSED_AR : return "Fused Node (Add + ReLU)";
        case Oper::FUSED_MAR : return "Fused Node (MatMul + Add + ReLU)";
        case Oper::FUSED_MA : return "Fused Node (MatMul + Add)";
        default: return "Undefined";
    }
}

std::vector<float> readFloatsFromFile(std::string filename, size_t bytes_to_read) {
    std::vector<float> data;

    size_t num_floats = bytes_to_read / sizeof(float);

    if (num_floats == 0) {
        std::cerr << "Warning: Requested bytes (" << bytes_to_read 
                  << ") is smaller than the size of one float." << std::endl;
        return data;
    }

    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    data.resize(num_floats);

    file.read(reinterpret_cast<char*>(data.data()), num_floats * sizeof(float));

    std::streamsize bytes_read = file.gcount();
    size_t floats_read = bytes_read / sizeof(float);

    if (floats_read < num_floats) {
        data.resize(floats_read);
    }

    return data;
}

void writeVectorToFile(const float *data, int N, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "' for writing." << std::endl;
        return;
    }

    outFile.write(reinterpret_cast<const char*>(data), N * sizeof(float));

    outFile.close();
    std::cout << "Successfully wrote " << N << " elements to " << filename << std::endl;
}

void generateAndSaveInput(Node* node) {
    int size = 1;
    std::mt19937 rng(42);
    for (int d : node->shape) size *= d;

    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> cpuBuf(size);
    for (float& f : cpuBuf) f = dist(rng);

    std::string filename = node->name + ".bin";
    std::ofstream file(filename, std::ios::binary);
    file.write((char*)cpuBuf.data(), size * sizeof(float));
}

int getRandomInt(int min, int max) {
    // 'static' means these are created once and remembered across function calls
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // The distribution is cheap to create, so it can be made every time
    std::uniform_int_distribution<> distrib(min, max);
    
    return distrib(gen);
}