#include <iostream>
#include <vector>
#include <fileio.h>
#include <graph.h>
#include <node.h>

int main() {
    Graph graph;

    // Base Input (Simulating an embedding vector)
    Node* input_X = graph.addInput("Input_X", {256, 1});

    // =======================================================
    // Branch 1: Standard Fusable Block
    // =======================================================
    Node* w1 = graph.addInput("W1", {128, 256});
    Node* b1 = graph.addInput("B1", {128, 1});
    Node* matmul1 = graph.addNode("MatMul_1", Oper::MATMUL, {w1, input_X});
    Node* add1    = graph.addNode("Add_1", Oper::ADD, {matmul1, b1});
    Node* relu1   = graph.addNode("ReLU_1", Oper::ReLU, {add1});

    // =======================================================
    // Branch 2: Deep Fusable Pipeline (Sequential Fusions)
    // =======================================================
    Node* w2a = graph.addInput("W2a", {512, 256});
    Node* b2a = graph.addInput("B2a", {512, 1});
    Node* matmul2a = graph.addNode("MatMul_2a", Oper::MATMUL, {w2a, input_X});
    Node* add2a    = graph.addNode("Add_2a", Oper::ADD, {matmul2a, b2a});
    Node* relu2a   = graph.addNode("ReLU_2a", Oper::ReLU, {add2a});

    Node* w2b = graph.addInput("W2b", {128, 512});
    Node* b2b = graph.addInput("B2b", {128, 1});
    Node* matmul2b = graph.addNode("MatMul_2b", Oper::MATMUL, {w2b, relu2a});
    Node* add2b    = graph.addNode("Add_2b", Oper::ADD, {matmul2b, b2b});
    Node* relu2b   = graph.addNode("ReLU_2b", Oper::ReLU, {add2b});

    // =======================================================
    // Branch 3: The Unfusable Split (Fusion Trap)
    // =======================================================
    // MatMul_3 branches out to TWO different consumers. 
    // It must NOT fuse, or it will corrupt the data flow.
    Node* w3 = graph.addInput("W3", {128, 256});
    Node* matmul3 = graph.addNode("MatMul_3", Oper::MATMUL, {w3, input_X});
    
    Node* b3_A = graph.addInput("B3_A", {128, 1});
    Node* add3_A  = graph.addNode("Add_3A", Oper::ADD, {matmul3, b3_A});
    Node* relu3_A = graph.addNode("ReLU_3A", Oper::ReLU, {add3_A});

    Node* b3_B = graph.addInput("B3_B", {128, 1});
    Node* add3_B  = graph.addNode("Add_3B", Oper::ADD, {matmul3, b3_B});
    Node* relu3_B = graph.addNode("ReLU_3B", Oper::ReLU, {add3_B});

    // =======================================================
    // Merge Layer: Combine all paths back together
    // =======================================================
    Node* merge_1_2 = graph.addNode("Merge_1_2", Oper::ADD, {relu1, relu2b});
    Node* merge_3   = graph.addNode("Merge_3", Oper::ADD, {relu3_A, relu3_B});
    
    Node* final_add = graph.addNode("Final_Add", Oper::ADD, {merge_1_2, merge_3});
    Node* final_out = graph.addNode("Final_ReLU", Oper::ReLU, {final_add});

    graph.setOutput(final_out);
    std::vector<Node*> topoS = graph.topoSort();
    std::cout << "----------------------Topological Sort----------------------" << std::endl;
    for (Node* item : topoS) {
        graph.printNode(item);
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Running Input layers generation" << std::endl;
    graph.generator();

    std::cout << "Executing UNfused Graph" << std::endl;
    graph.execute();
    float unfusedMs = graph.benchExecution();

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "------------Fusion Pass------------" << std::endl;
    std::vector<Node*> fusionList = graph.fusionPass();
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Total Nodes Before Fusion: " << topoS.size() << std::endl;
    std::cout << "Total Nodes After Fusion: " << fusionList.size() << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "Executing FUSED Graph" << std::endl;
    graph.execute(fusionList);
    float fusedMs   = graph.benchExecution(fusedList);

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "\n=== Benchmark ===" << std::endl;
    std::cout << "Unfused: " << unfusedMs << " ms" << std::endl;
    std::cout << "Fused:   " << fusedMs   << " ms" << std::endl;
    std::cout << "Speedup: " << unfusedMs / fusedMs << "x" << std::endl;


    return 0;
}