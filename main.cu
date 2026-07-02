#include <fileio.h>
#include <graph.h>
#include <iostream>
#include <node.h>
#include <vector>

int main(int argc, char **argv) {
  Graph graph;

  bool codegen = false;
  bool standalone = false;
  std::string graphFile = "";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--codegen") {
      codegen = true;
    } else if (arg == "--standalone") {
      standalone = true;
    } else {
      graphFile = arg;
    }
  }

  if (graphFile != "") {
    std::cout << "[Graph Compiler] Loading graph from " << graphFile
              << std::endl;
    graph.loadFromFile(graphFile);
  } else {
    std::cout << "[Graph Compiler] No graph file provided, using hardcoded "
                 "benchmark graph."
              << std::endl;
    int DIM = 4096;

    // Base Input
    Node *input_X = graph.addInput("Input_X", {DIM, 1});

    // --- Block 1: Massive Projection ---
    Node *w1 = graph.addInput("W1", {DIM, DIM});
    Node *b1 = graph.addInput("B1", {DIM, 1});
    Node *matmul1 = graph.addNode("MatMul_1", Oper::MATMUL, {w1, input_X});
    Node *add1 = graph.addNode("Add_1", Oper::ADD, {matmul1, b1});
    Node *relu1 = graph.addNode("ReLU_1", Oper::ReLU, {add1});

    // --- Block 2: The Parallel Width Test ---
    // Branch A
    Node *w2a = graph.addInput("W2a", {DIM, DIM});
    Node *b2a = graph.addInput("B2a", {DIM, 1});
    Node *matmul2a = graph.addNode("MatMul_2a", Oper::MATMUL, {w2a, relu1});
    Node *add2a = graph.addNode("Add_2a", Oper::ADD, {matmul2a, b2a});
    Node *relu2a = graph.addNode("ReLU_2a", Oper::ReLU, {add2a});

    // Branch B
    Node *w2b = graph.addInput("W2b", {DIM, DIM});
    Node *b2b = graph.addInput("B2b", {DIM, 1});
    Node *matmul2b = graph.addNode("MatMul_2b", Oper::MATMUL, {w2b, relu1});
    Node *add2b = graph.addNode("Add_2b", Oper::ADD, {matmul2b, b2b});
    Node *relu2b = graph.addNode("ReLU_2b", Oper::ReLU, {add2b});

    // Branch C
    Node *w2c = graph.addInput("W2c", {DIM, DIM});
    Node *b2c = graph.addInput("B2c", {DIM, 1});
    Node *matmul2c = graph.addNode("MatMul_2c", Oper::MATMUL, {w2c, relu1});
    Node *add2c = graph.addNode("Add_2c", Oper::ADD, {matmul2c, b2c});
    Node *relu2c = graph.addNode("ReLU_2c", Oper::ReLU, {add2c});

    // --- Merge Parallel Branches ---
    Node *merge_AB = graph.addNode("Merge_AB", Oper::ADD, {relu2a, relu2b});
    Node *merge_ABC = graph.addNode("Merge_ABC", Oper::ADD, {merge_AB, relu2c});
    Node *relu_merge = graph.addNode("ReLU_Merge", Oper::ReLU, {merge_ABC});

    // --- Block 3: Deep Sequential Bottleneck ---
    Node *w3a = graph.addInput("W3a", {DIM / 2, DIM});
    Node *b3a = graph.addInput("B3a", {DIM / 2, 1});
    Node *matmul3a =
        graph.addNode("MatMul_3a", Oper::MATMUL, {w3a, relu_merge});
    Node *add3a = graph.addNode("Add_3a", Oper::ADD, {matmul3a, b3a});
    Node *relu3a = graph.addNode("ReLU_3a", Oper::ReLU, {add3a});

    Node *w3b = graph.addInput("W3b", {DIM, DIM / 2});
    Node *b3b = graph.addInput("B3b", {DIM, 1});
    Node *matmul3b = graph.addNode("MatMul_3b", Oper::MATMUL, {w3b, relu3a});
    Node *add3b = graph.addNode("Add_3b", Oper::ADD, {matmul3b, b3b});
    Node *relu3b = graph.addNode("ReLU_3b", Oper::ReLU, {add3b});

    // --- Final Residual Add (Skipping all of Block 2 and 3) ---
    Node *final_res =
        graph.addNode("Final_Residual", Oper::ADD, {relu1, relu3b});
    Node *final_out = graph.addNode("Final_Out", Oper::ReLU, {final_res});

    graph.setOutput(final_out);
  }
  std::vector<Node *> topoS = graph.topoSort();
  std::cout << "[Graph Compiler] Topological sort completed (" << topoS.size()
            << " nodes)." << std::endl;

  if (graphFile == "") {
    std::cout << "[Graph Compiler] Generating random fallback inputs..."
              << std::endl;
    graph.generator();
  } else {
    std::cout << "[Graph Compiler] Using loaded weights from binary files."
              << std::endl;
  }

  if (codegen) {
    std::cout << "\n--- Optimization Pass ---" << std::endl;
    std::vector<Node *> fusionList = graph.fusionPass();
    std::cout << "[Optimizer] Fused nodes from " << topoS.size() << " down to "
              << fusionList.size() << "!" << std::endl;

    std::cout << "\n--- Static Codegen Pass ---" << std::endl;
    graph.compileToFile("generated_model.cu", fusionList, standalone);
  } else {
    std::cout << "\n--- Baseline Execution (Unfused) ---" << std::endl;
    graph.execute();
    float unfusedMs = graph.benchExecution();

    std::cout << "\n--- Optimization Pass ---" << std::endl;
    std::vector<Node *> fusionList = graph.fusionPass();
    std::cout << "[Optimizer] Fused nodes from " << topoS.size() << " down to "
              << fusionList.size() << "!" << std::endl;

    std::cout << "\n--- Fused Execution ---" << std::endl;
    graph.execute(fusionList);
    float fusedMs = graph.benchExecution(fusionList);

    std::cout << "\n=====================================" << std::endl;
    std::cout << "          PERFORMANCE REPORT         " << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << " Baseline: " << unfusedMs << " ms" << std::endl;
    std::cout << " Fused:    " << fusedMs << " ms" << std::endl;
    std::cout << " Speedup:  " << unfusedMs / fusedMs << "x" << std::endl;
    std::cout << "=====================================\n" << std::endl;
  }

  return 0;
}