#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <queue>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <random>
#include <fileio.h>
#include <graph.h>
#include <node.h>
#include <kernel.h>
#include <unordered_set>
#include <utility>
#include <map>


float* Graph::bufferAlloc(Node* node) {
    float* address;
    size_t bytesNode = node->shape[0] * node->shape[1];

    cudaMalloc((void**)&address, bytesNode * sizeof(float));

    return address;
}

std::unordered_map<int, float*> Graph::nodeMem() {
    std::unordered_map<int, float*> mp;


    for (const std::unique_ptr<Node>& node : nodes) {
        Node* raw = node.get();
        mp[raw->id] = bufferAlloc(raw);
    }

    return mp;
}

void Graph::generator() {
    for (Node* item : sorted) {
        if (item->operation == Oper::INPUT) {
            generateAndSaveInput(item);
        }
    }
}

void Graph::execute() {
    if (nodeMemMap.size() < nodes.size()) {
        nodeMemMap = nodeMem();
    }
    int inputTill = 0;

    for (int i = 0; i < sorted.size(); i++) {
        if (sorted[i]->operation == Oper::INPUT) {
            size_t byteSize = sorted[i]->shape[0] * sorted[i]->shape[1] * sizeof(float);
            std::vector<float> cont = readFloatsFromFile(sorted[i]->name+".bin", byteSize);

            cudaMemcpy(nodeMemMap[sorted[i]->id], cont.data(), byteSize, cudaMemcpyHostToDevice);
            std::cout << "Inputing from " << sorted[i]->name << std::endl;
        } else {
            inputTill = i;
            break;
        }
    }

    for (int i = inputTill; i < sorted.size(); i++) {
        if (sorted[i]->operation == Oper::MATMUL) {
            int row_A = sorted[i]->inputs[0]->shape[0];
            int N     = sorted[i]->inputs[0]->shape[1];
            int col_B = sorted[i]->inputs[1]->shape[1];

            matMul(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->inputs[1]->id], nodeMemMap[sorted[i]->id], row_A, N, col_B);

        } else if (sorted[i]->operation == Oper::ADD) {
            int height = sorted[i]->shape[0];
            int width  = sorted[i]->shape[1];
            
            matAdd(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->inputs[1]->id], nodeMemMap[sorted[i]->id], height, width);
            
        } else if (sorted[i]->operation == Oper::ReLU) {
            int height = sorted[i]->shape[0];
            int width  = sorted[i]->shape[1];
            
            matReLU(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->id], height, width);
        }

        std::cout << "At " << sorted[i]->name << std::endl;
    }

    Node* node = getOutput();
    std::vector<float> output(node->shape[0] * node->shape[1]);
    std::cout << std::endl;
    std::cout << "Final Node: " << node->name << std::endl;
    cudaMemcpy(output.data(), nodeMemMap[node->id], node->shape[0] * node->shape[1] * sizeof(float), cudaMemcpyDeviceToHost);
    writeVectorToFile(output.data(), output.size(), node->name);
}

void Graph::execute(std::vector<Node*> fusedGraphs) {
    nodeMemMap.clear();
    nodeMemMap = nodeMem();
    
    int inputTill = 0;

    for (int i = 0; i < fusedGraphs.size(); i++) {
        if (fusedGraphs[i]->operation == Oper::INPUT) {
            size_t byteSize = fusedGraphs[i]->shape[0] * fusedGraphs[i]->shape[1] * sizeof(float);
            std::vector<float> cont = readFloatsFromFile(fusedGraphs[i]->name+".bin", byteSize);

            cudaMemcpy(nodeMemMap[fusedGraphs[i]->id], cont.data(), byteSize, cudaMemcpyHostToDevice);
            std::cout << "Inputing from " << fusedGraphs[i]->name << std::endl;
        } else {
            inputTill = i;
            break;
        }
    }

    for (int i = inputTill; i < fusedGraphs.size(); i++) {
        if (fusedGraphs[i]->operation == Oper::MATMUL) {
            int row_A = fusedGraphs[i]->inputs[0]->shape[0];
            int N     = fusedGraphs[i]->inputs[0]->shape[1];
            int col_B = fusedGraphs[i]->inputs[1]->shape[1];

            matMul(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);

        } else if (fusedGraphs[i]->operation == Oper::ADD) {
            int height = fusedGraphs[i]->shape[0];
            int width  = fusedGraphs[i]->shape[1];
            
            matAdd(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->id], height, width);
            
        } else if (fusedGraphs[i]->operation == Oper::ReLU) {
            int height = fusedGraphs[i]->shape[0];
            int width  = fusedGraphs[i]->shape[1];
            
            matReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->id], height, width);
            
        } else if (fusedGraphs[i]->operation == Oper::FUSED_MR) {
            int row_A = fusedGraphs[i]->inputs[0]->shape[0];
            int N     = fusedGraphs[i]->inputs[0]->shape[1];
            int col_B = fusedGraphs[i]->inputs[1]->shape[1];

            float* A = nodeMemMap[fusedGraphs[i]->inputs[0]->id];
            float* B = nodeMemMap[fusedGraphs[i]->inputs[1]->id];
            float* C = nodeMemMap[fusedGraphs[i]->id];

            matMulReLU(A, B, C, row_A, N, col_B);

        } else if (fusedGraphs[i]->operation == Oper::FUSED_AR) {
            int height = fusedGraphs[i]->shape[0];
            int width  = fusedGraphs[i]->shape[1];

            float* A = nodeMemMap[fusedGraphs[i]->inputs[0]->id];
            // Bias is guaranteed to be the last input node
            float* Bias = nodeMemMap[fusedGraphs[i]->inputs.back()->id]; 
            float* C = nodeMemMap[fusedGraphs[i]->id];

            matAddReLU(A, Bias, C, height, width);

        } else if (fusedGraphs[i]->operation == Oper::FUSED_MAR) {
            int row_A = fusedGraphs[i]->inputs[0]->shape[0];
            int N     = fusedGraphs[i]->inputs[0]->shape[1];
            int col_B = fusedGraphs[i]->inputs[1]->shape[1];

            float* A = nodeMemMap[fusedGraphs[i]->inputs[0]->id];
            float* B = nodeMemMap[fusedGraphs[i]->inputs[1]->id];
            // Bias is guaranteed to be the last input node
            float* Bias = nodeMemMap[fusedGraphs[i]->inputs.back()->id]; 
            float* C = nodeMemMap[fusedGraphs[i]->id];

            matMulAddReLU(A, B, Bias, C, row_A, N, col_B);

        } else if (fusedGraphs[i]->operation == Oper::FUSED_MA) {
            int row_A = fusedGraphs[i]->inputs[0]->shape[0];
            int N     = fusedGraphs[i]->inputs[0]->shape[1];
            int col_B = fusedGraphs[i]->inputs[1]->shape[1];

            float* A = nodeMemMap[fusedGraphs[i]->inputs[0]->id];
            float* B = nodeMemMap[fusedGraphs[i]->inputs[1]->id];
            // Bias is guaranteed to be the last input node
            float* Bias = nodeMemMap[fusedGraphs[i]->inputs.back()->id];
            float* C = nodeMemMap[fusedGraphs[i]->id];

            matMulAdd(A, B, Bias, C, row_A, N, col_B);
        }

        std::cout << "At " << fusedGraphs[i]->name << std::endl;
    }

    Node* node = fusedGraphs[fusedGraphs.size()-1];
    std::vector<float> output(node->shape[0] * node->shape[1]);
    std::cout << std::endl;
    std::cout << "Final Node: " << node->name << std::endl;
    cudaMemcpy(output.data(), nodeMemMap[node->id], node->shape[0] * node->shape[1] * sizeof(float), cudaMemcpyDeviceToHost);
    writeVectorToFile(output.data(), output.size(), node->name+"_fusedGraph");
}

float Graph::benchExecution() {
    if (nodeMemMap.size() < nodes.size()) {
        nodeMemMap = nodeMem();
    }

    // warmup
    for (int i = 0; i < sorted.size(); i++) {
        if (sorted[i]->operation == Oper::MATMUL) {
            int row_A = sorted[i]->inputs[0]->shape[0];
            int N     = sorted[i]->inputs[0]->shape[1];
            int col_B = sorted[i]->inputs[1]->shape[1];
            matMul(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->inputs[1]->id], nodeMemMap[sorted[i]->id], row_A, N, col_B);
        } else if (sorted[i]->operation == Oper::ADD) {
            int height = sorted[i]->shape[0];
            int width  = sorted[i]->shape[1];
            matAdd(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->inputs[1]->id], nodeMemMap[sorted[i]->id], height, width);
        } else if (sorted[i]->operation == Oper::ReLU) {
            int height = sorted[i]->shape[0];
            int width  = sorted[i]->shape[1];
            matReLU(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->id], height, width);
        }
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int run = 0; run < 100; run++) {
        for (int i = 0; i < sorted.size(); i++) {
            if (sorted[i]->operation == Oper::MATMUL) {
                int row_A = sorted[i]->inputs[0]->shape[0];
                int N     = sorted[i]->inputs[0]->shape[1];
                int col_B = sorted[i]->inputs[1]->shape[1];
                matMul(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->inputs[1]->id], nodeMemMap[sorted[i]->id], row_A, N, col_B);
            } else if (sorted[i]->operation == Oper::ADD) {
                int height = sorted[i]->shape[0];
                int width  = sorted[i]->shape[1];
                matAdd(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->inputs[1]->id], nodeMemMap[sorted[i]->id], height, width);
            } else if (sorted[i]->operation == Oper::ReLU) {
                int height = sorted[i]->shape[0];
                int width  = sorted[i]->shape[1];
                matReLU(nodeMemMap[sorted[i]->inputs[0]->id], nodeMemMap[sorted[i]->id], height, width);
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 100.0f;
}

float Graph::benchExecution(std::vector<Node*> fusedGraphs) {
    nodeMemMap.clear();
    nodeMemMap = nodeMem();

    // warmup
    for (int i = 0; i < fusedGraphs.size(); i++) {
        if (fusedGraphs[i]->operation == Oper::INPUT) continue;

        if (fusedGraphs[i]->operation == Oper::MATMUL) {
            int row_A = fusedGraphs[i]->inputs[0]->shape[0];
            int N     = fusedGraphs[i]->inputs[0]->shape[1];
            int col_B = fusedGraphs[i]->inputs[1]->shape[1];
            matMul(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);
        } else if (fusedGraphs[i]->operation == Oper::ADD) {
            int height = fusedGraphs[i]->shape[0];
            int width  = fusedGraphs[i]->shape[1];
            matAdd(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->id], height, width);
        } else if (fusedGraphs[i]->operation == Oper::ReLU) {
            int height = fusedGraphs[i]->shape[0];
            int width  = fusedGraphs[i]->shape[1];
            matReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->id], height, width);
        } else if (fusedGraphs[i]->operation == Oper::FUSED_MR) {
            int row_A = fusedGraphs[i]->inputs[0]->shape[0];
            int N     = fusedGraphs[i]->inputs[0]->shape[1];
            int col_B = fusedGraphs[i]->inputs[1]->shape[1];
            matMulReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);
        } else if (fusedGraphs[i]->operation == Oper::FUSED_AR) {
            int height = fusedGraphs[i]->shape[0];
            int width  = fusedGraphs[i]->shape[1];
            matAddReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs.back()->id], nodeMemMap[fusedGraphs[i]->id], height, width);
        } else if (fusedGraphs[i]->operation == Oper::FUSED_MAR) {
            int row_A = fusedGraphs[i]->inputs[0]->shape[0];
            int N     = fusedGraphs[i]->inputs[0]->shape[1];
            int col_B = fusedGraphs[i]->inputs[1]->shape[1];
            matMulAddReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->inputs.back()->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);
        } else if (fusedGraphs[i]->operation == Oper::FUSED_MA) {
            int row_A = fusedGraphs[i]->inputs[0]->shape[0];
            int N     = fusedGraphs[i]->inputs[0]->shape[1];
            int col_B = fusedGraphs[i]->inputs[1]->shape[1];
            matMulAdd(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->inputs.back()->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);
        }
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int run = 0; run < 100; run++) {
        for (int i = 0; i < fusedGraphs.size(); i++) {
            if (fusedGraphs[i]->operation == Oper::INPUT) continue;

            if (fusedGraphs[i]->operation == Oper::MATMUL) {
                int row_A = fusedGraphs[i]->inputs[0]->shape[0];
                int N     = fusedGraphs[i]->inputs[0]->shape[1];
                int col_B = fusedGraphs[i]->inputs[1]->shape[1];
                matMul(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);
            } else if (fusedGraphs[i]->operation == Oper::ADD) {
                int height = fusedGraphs[i]->shape[0];
                int width  = fusedGraphs[i]->shape[1];
                matAdd(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->id], height, width);
            } else if (fusedGraphs[i]->operation == Oper::ReLU) {
                int height = fusedGraphs[i]->shape[0];
                int width  = fusedGraphs[i]->shape[1];
                matReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->id], height, width);
            } else if (fusedGraphs[i]->operation == Oper::FUSED_MR) {
                int row_A = fusedGraphs[i]->inputs[0]->shape[0];
                int N     = fusedGraphs[i]->inputs[0]->shape[1];
                int col_B = fusedGraphs[i]->inputs[1]->shape[1];
                matMulReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);
            } else if (fusedGraphs[i]->operation == Oper::FUSED_AR) {
                int height = fusedGraphs[i]->shape[0];
                int width  = fusedGraphs[i]->shape[1];
                matAddReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs.back()->id], nodeMemMap[fusedGraphs[i]->id], height, width);
            } else if (fusedGraphs[i]->operation == Oper::FUSED_MAR) {
                int row_A = fusedGraphs[i]->inputs[0]->shape[0];
                int N     = fusedGraphs[i]->inputs[0]->shape[1];
                int col_B = fusedGraphs[i]->inputs[1]->shape[1];
                matMulAddReLU(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->inputs.back()->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);
            } else if (fusedGraphs[i]->operation == Oper::FUSED_MA) {
                int row_A = fusedGraphs[i]->inputs[0]->shape[0];
                int N     = fusedGraphs[i]->inputs[0]->shape[1];
                int col_B = fusedGraphs[i]->inputs[1]->shape[1];
                matMulAdd(nodeMemMap[fusedGraphs[i]->inputs[0]->id], nodeMemMap[fusedGraphs[i]->inputs[1]->id], nodeMemMap[fusedGraphs[i]->inputs.back()->id], nodeMemMap[fusedGraphs[i]->id], row_A, N, col_B);
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 100.0f;
}

Node* Graph::fuseNodes(std::vector<Node*> nodes2Fuse, std::vector<int>& fusedMap) {
    std::unordered_set<Node*> nodies(nodes2Fuse.begin(), nodes2Fuse.end());
    std::vector<Node*> inputting;
    for (Node* item : nodes2Fuse) {
        for (Node* val : item->inputs) {
              if (nodies.find(val) == nodies.end()) inputting.push_back(val);
        }
    }
    std::string identifier = "";
    for (Node* item : nodes2Fuse) {
        if (item->operation == Oper::MATMUL) identifier += "M";
        else if (item->operation == Oper::ADD) identifier += "A";
        else if (item->operation == Oper::ReLU) identifier += "R";
    }

    Node* fusedNode = nullptr;

    if (identifier == "MA") fusedNode = addNode("Fused_MA", Oper::FUSED_MA, inputting);
    else if (identifier == "MR") fusedNode = addNode("Fused_MR", Oper::FUSED_MR, inputting);
    else if (identifier == "AR") fusedNode = addNode("Fused_AR", Oper::FUSED_AR, inputting);
    else if (identifier == "MAR") fusedNode = addNode("Fused_MAR", Oper::FUSED_MAR, inputting);

    if (!fusedNode) {
        std::cout << "Error: unknown fusion pattern: " << identifier << "\n";
        return nullptr;
    }

    for (Node* item : outMap[nodes2Fuse[nodes2Fuse.size()-1]]) {
        auto it = std::find(item->inputs.begin(), item->inputs.end(), nodes2Fuse.back());
        if (it != item->inputs.end()) *it = fusedNode;
    }

    fusedNode->shape = nodes2Fuse[nodes2Fuse.size()-1]->shape;

    for (Node* item : nodes2Fuse) {
        fusedMap[item->id] = 1;
    }

    return fusedNode;
}

std::vector<Node*> Graph::fuseDFSMerger(Node* node, std::vector<int>& visited, bool matmul, bool add, bool relu) {
    if (outMap[node].size() > 1 || outMap[node].size() == 0) {
        return {node};
    }
    
    visited[node->id] = 1;
    std::vector<Node*> ans;
    
    if (!visited[outMap[node][0]->id]) {
        if (!matmul && node->operation == Oper::MATMUL) {
            ans = fuseDFS(outMap[node][0], visited, 1, add, relu);
        } else if (!add && node->operation == Oper::ADD) {
            ans = fuseDFS(outMap[node][0], visited, matmul, 1, relu);
        } else if (!relu && node->operation == Oper::ReLU) {
            ans = fuseDFS(outMap[node][0], visited, matmul, add, 1);
        }
    }


    ans.push_back(node);

    return ans;
} 

std::vector<Node*> Graph::fuseDFS(Node* node, std::vector<int>& visited, bool matmul, bool add, bool relu) {
    int count = 0;
    for (Node* item : node->inputs) {
        if (item->operation != Oper::INPUT) {
            count++;
        }
    }

    if (relu) {
        return {};
    }

    if (count > 1) {
        return {};
    }

    if (outMap[node].size() > 1 || outMap[node].size() == 0) {
        return {node};
    }

    visited[node->id] = 1;
    std::vector<Node*> ans;
    
    if (!visited[outMap[node][0]->id]) {
        if (!matmul && node->operation == Oper::MATMUL) {
            ans = fuseDFS(outMap[node][0], visited, 1, add, relu);
        } else if (!add && node->operation == Oper::ADD) {
            ans = fuseDFS(outMap[node][0], visited, matmul, 1, relu);
        } else if (!relu && node->operation == Oper::ReLU) {
            ans = fuseDFS(outMap[node][0], visited, matmul, add, 1);
        }
    }

    ans.push_back(node);

    return ans;
}

std::vector<Node*> Graph::fusionPass() {
    outMap.clear();
    std::vector<std::vector<Node*>> fusion;

    for (int i = 0; i < sorted.size(); i++) {
        if (!sorted[i]->inputs.empty()) {
            for (Node* item : sorted[i]->inputs) {
                if (outMap.find(item) != outMap.end()) {
                    outMap[item].push_back(sorted[i]);
                } else {
                    outMap[item] = {sorted[i]};
                }
            }
        }
    }
    std::vector<int> visited(sorted.size(), 0);
    std::vector<int> mergerMap(sorted.size(), 0);
    std::vector<std::pair<int, int>> fuseableMap(sorted.size(), std::pair<int, int>(0, 0));

    for (Node* item : sorted) {
        if (item->operation == Oper::INPUT) continue;

        int count = 0;
        if (item->inputs.size() > 1) {
          for (Node* val : item->inputs) {
            if (val->operation == Oper::INPUT) continue;

            count++;
          }
        }

        if (count > 1) {
          mergerMap[item->id] = 1;
        } 
    }

    int counter = -1;
    for (Node* item : sorted) {
        if (item->operation == Oper::INPUT) continue;

        std::vector<Node*> toFuse;

        if (!visited[item->id] && !mergerMap[item->id]) {
            toFuse = fuseDFS(item, visited, 0, 0, 0);
        }

        if (mergerMap[item->id]) {
            
            toFuse = fuseDFSMerger(item, visited, 0, 0, 0);
        }

        if (toFuse.size() > 1) {
            reverse(toFuse.begin(), toFuse.end());
            counter = fusion.size();
            for (Node* val : toFuse) {
                fuseableMap[val->id] = {1, counter};
            }
            fusion.push_back(toFuse);
        }
    }
    std::vector<int> fusedMap(sorted.size(), 0);
    std::vector<Node*> fusedSorted;

    for (Node* item : sorted) {
        if (fuseableMap[item->id].first) {
            if (!fusedMap[item->id]) {
                Node* newFusedNode = fuseNodes(fusion[fuseableMap[item->id].second], fusedMap);
                fusedSorted.push_back(newFusedNode);
            }
        } else {
            fusedSorted.push_back(item);
        }
    }

    return fusedSorted;

    // for (std::vector<Node*> toFuse : fusion) {
    //     std::cout << "----------List----------" << std::endl;
    //     for (Node* item : toFuse) {
    //         printNode(item);
    //         std::cout << std::endl;
    //     }

    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }
}


void Graph::setOutput(Node* node) {
    outputNode = node;
}

Node* Graph::getOutput() {
    return outputNode;
}


Node* Graph::addInput(std::string nm, std::vector<int> shp) {
    std::unique_ptr<Node> newNode = std::make_unique<Node>();
    newNode->id = nodes.size();
    newNode->name = nm;
    newNode->shape = shp;
    newNode->operation = Oper::INPUT;
    Node* raw = newNode.get();
    nodes.push_back(std::move(newNode));

    return raw;
}

Node* Graph::addNode(std::string nm, Oper op, std::vector<Node*> in) {
    std::vector<int> sp;
    if (op == Oper::MATMUL) {
        if (in.size() != 2) {
            std::cout << "Error: input for MatMul is only 1 node\n";
            return nullptr;
        }

        if (in[0]->shape[1] != in[1]->shape[0]) {
            std::cout << "Error: Dimensions don't match for Matrix Multiplication\n";
            return nullptr;
        }

        sp = std::vector<int>({in[0]->shape[0], in[1]->shape[1]});
    }

    if (op == Oper::ADD) {
        if (in.size() != 2) {
            std::cout << "Error: input for addition is given: " << in.size() << "\n";
            return nullptr;
        }
        if (in[0]->shape[0] != in[1]->shape[0] || in[0]->shape[1] != in[1]->shape[1]) {
            std::cout << "Error: ADD shape mismatch\n";
            return nullptr;
        }
        sp = in[0]->shape;
    }

    if (op == Oper::ReLU) {
        if (in.size() != 1) {
            std::cout << "Error: ReLU is given input: " << in.size() << "\n";
            return nullptr;
        }
        sp = std::vector<int>({in[0]->shape[0], in[0]->shape[1]});
    }

    std::unique_ptr<Node> newNode = std::make_unique<Node>();
    newNode->id = nodes.size();
    newNode->name = nm;
    newNode->operation = op;
    newNode->shape = sp;
    newNode->inputs = in;

    Node* raw = newNode.get();
    nodes.push_back(std::move(newNode));

    return raw;
}

void Graph::printGraph() {
    for (auto& item : nodes) {
        std::cout << "Node Name: " << item->name << "\n";
        std::cout << "ID: " << item->id << "\n";
        std::cout << "Operation: " << op2String(item->operation) << "\n";
        std::cout << "Shape: (";
        for (int i = 0; i < item->shape.size(); i++) {
            std::cout << item->shape[i];
            if (i < item->shape.size() - 1) std::cout << ", ";
        }
        std::cout << ")\n";
        if (!item->inputs.empty()) {
            std::cout << "Input Nodes: ";
            for (Node* node : item->inputs) {
                std::cout << node->name << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        std::cout << std::endl;
        std::cout << std::endl;
    }
}

void Graph::printNode(Node* item) {
    std::cout << "Node Name: " << item->name << "\n";
    std::cout << "ID: " << item->id << "\n";
    std::cout << "Operation: " << op2String(item->operation) << "\n";
    std::cout << "Shape: (";
    for (int i = 0; i < item->shape.size(); i++) {
        std::cout << item->shape[i];
        if (i < item->shape.size() - 1) std::cout << ", ";
    }
    std::cout << ")\n";
    if (!item->inputs.empty()) {
        std::cout << "Input Nodes: ";
        for (Node* node : item->inputs) {
            std::cout << node->name << "(id: " << node->id << ")" << " ";
        }
        std::cout << "\n";
    }
}

std::vector<Node*> Graph::topoSort() {
    int nodesNum = nodes.size();
    std::vector<int> indegree(nodesNum);
    std::unordered_map<int, std::vector<Node*>> adjList;

    for (int i = 0; i < nodesNum; i++) {
        Node* node = nodes[i].get();
        indegree[i] = node->inputs.size();
        for (Node* item : node->inputs) {
            adjList[item->id].push_back(node);
        }
    }

    std::queue<Node*> que;
    for (int i = 0; i < nodesNum; i++) {
        if (indegree[i] == 0) {
            Node* node = nodes[i].get();
            que.push(node);
        }
    }

    std::vector<Node*> topo;
    while (!que.empty()) {
        Node* node = que.front();
        que.pop();
        topo.push_back(node);

        for (Node* item : adjList[node->id]) {
            indegree[item->id]--;
            if (indegree[item->id] == 0) {
                que.push(item);
            }
        }
    }

    if (topo.size() != nodesNum) {
        std::cout << "Error: cycle detected in graph, topological sort incomplete\n";
        return {};
    }

    for (Node* item : topo) {
        sorted.push_back(item);
    }

    return topo;
}

std::vector<Node*> Graph::topoSort(std::vector<Node*> newNodes) {
    int nodesNum = newNodes.size();
    std::vector<int> indegree(nodesNum);
    std::unordered_map<int, std::vector<Node*>> adjList;

    for (int i = 0; i < nodesNum; i++) {
        Node* node = newNodes[i];
        indegree[i] = node->inputs.size();
        for (Node* item : node->inputs) {
            adjList[item->id].push_back(node);
        }
    }

    std::queue<Node*> que;
    for (int i = 0; i < nodesNum; i++) {
        if (indegree[i] == 0) {
            Node* node = newNodes[i];
            que.push(node);
        }
    }

    std::vector<Node*> topo;
    while (!que.empty()) {
        Node* node = que.front();
        que.pop();
        topo.push_back(node);

        for (Node* item : adjList[node->id]) {
            indegree[item->id]--;
            if (indegree[item->id] == 0) {
                que.push(item);
            }
        }
    }

    if (topo.size() != nodesNum) {
      std::cout << "Topological Sort size: " << topo.size() << std::endl;
        std::cout << "newNodes size: " << nodesNum << std::endl;
        std::cout << "Error: cycle detected in graph, topological sort incomplete\n";
        return {};
    }

    return topo;
}

