#pragma once

#include <cuda_runtime.h>
#include <fileio.h>
#include <map>
#include <memory>
#include <node.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Graph {
  std::vector<std::unique_ptr<Node>> nodes;
  Node *outputNode;
  std::vector<Node *> sorted;
  std::unordered_map<Node *, std::vector<Node *>> outMap;
  std::unordered_map<int, float *> nodeMemMap;

  // ── Stream infrastructure ──
  std::vector<cudaStream_t> streams;         // pool of CUDA streams
  std::unordered_map<int, int> nodeStreamId; // node id -> stream index
  std::unordered_map<int, cudaEvent_t>
      nodeEvents; // node id -> completion event
  int numStreams = 0;

public:
  float *bufferAlloc(Node *node);

  std::unordered_map<int, float *> nodeMem();

  void generator();

  void execute();

  void execute(std::vector<Node *> fusedGraphs);

  void setOutput(Node *node);

  Node *getOutput();

  Node *addInput(std::string nm, std::vector<int> shp);

  Node *addNode(std::string nm, Oper op, std::vector<Node *> in);

  void printGraph();

  std::vector<Node *> topoSort();

  Node *fuseNodes(std::vector<Node *> nodes2Fuse, std::vector<int> &fusedMap);

  std::vector<Node *> fuseDFSMerger(Node *node, std::vector<int> &visited,
                                    bool matmul, bool add, bool relu);

  std::vector<Node *> fuseDFS(Node *node, std::vector<int> &visited,
                              bool matmul, bool add, bool relu);

  std::vector<Node *> fusionPass();

  void printNode(Node *item);

  std::vector<Node *> topoSort(std::vector<Node *> newNodes);

  float benchExecution(std::vector<Node *> fusedGraphs);

  float benchExecution();

  // ── Stream management ──
  void assignStreams(const std::vector<Node *> &schedule);
  void destroyStreams();
  ~Graph();
};