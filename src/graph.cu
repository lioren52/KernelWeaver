#include <algorithm>
#include <fileio.h>
#include <fstream>
#include <graph.h>
#include <iostream>
#include <kernel.h>
#include <map>
#include <memory>
#include <node.h>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

float *Graph::bufferAlloc(Node *node) {
  float *address;
  size_t bytesNode = node->shape[0] * node->shape[1];

  cudaMalloc((void **)&address, bytesNode * sizeof(float));

  return address;
}

std::unordered_map<int, float *> Graph::nodeMem() {
  std::unordered_map<int, float *> mp;

  for (const std::unique_ptr<Node> &node : nodes) {
    Node *raw = node.get();
    mp[raw->id] = bufferAlloc(raw);
  }

  return mp;
}

void Graph::generator() {
  for (Node *item : sorted) {
    if (item->operation == Oper::INPUT) {
      generateAndSaveInput(item);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream assignment: analyse the node schedule and assign CUDA streams so that
// independent branches execute concurrently.
//
// Algorithm:
//   1. For each non-INPUT node, determine which streams its *input* nodes ran
//   on.
//   2. If all inputs are on the same stream (or are INPUT nodes), keep using
//      that stream — this is a simple linear chain.
//   3. If inputs span >1 stream, this is a merge point: assign it to stream 0
//      and record events on all incoming streams so the merge waits for them.
//   4. When a node has >1 consumer (fork point), each consumer gets a distinct
//      stream to enable parallelism.
// ─────────────────────────────────────────────────────────────────────────────
void Graph::assignStreams(const std::vector<Node *> &schedule) {
  destroyStreams();
  nodeStreamId.clear();
  nodeEvents.clear();

  // Build out-edge map for fork detection
  std::unordered_map<int, std::vector<Node *>> localOutMap;
  std::unordered_map<int, bool> isInSchedule;
  for (Node *n : schedule) {
    isInSchedule[n->id] = true;
  }
  for (Node *n : schedule) {
    for (Node *inp : n->inputs) {
      if (isInSchedule.count(inp->id)) {
        localOutMap[inp->id].push_back(n);
      }
    }
  }

  // First pass: identify how many distinct streams we need.
  // Count the max fan-out at any fork point — that's our stream count.
  int maxFanOut = 1;
  for (auto &kv : localOutMap) {
    if ((int)kv.second.size() > maxFanOut)
      maxFanOut = kv.second.size();
  }
  numStreams =
      std::max(2, maxFanOut + 1); // at least 2 streams, +1 for main thread

  streams.resize(numStreams);
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // Assign all INPUT nodes to stream 0
  for (Node *n : schedule) {
    if (n->operation == Oper::INPUT) {
      nodeStreamId[n->id] = 0;
    }
  }

  // Second pass: assign streams based on dependency structure
  int nextFreeStream = 1; // stream 0 is the "main" stream
  for (Node *n : schedule) {
    if (n->operation == Oper::INPUT)
      continue;

    // Gather the set of streams that my inputs live on
    std::set<int> inputStreams;
    for (Node *inp : n->inputs) {
      if (nodeStreamId.count(inp->id)) {
        inputStreams.insert(nodeStreamId[inp->id]);
      }
    }

    if (inputStreams.size() <= 1) {
      // Simple chain — inherit parent's stream
      int parentStream = inputStreams.empty() ? 0 : *inputStreams.begin();

      // Check if parent is a fork point (has multiple consumers)
      // If so, this node should get its own stream
      bool parentIsFork = false;
      for (Node *inp : n->inputs) {
        if (inp->operation == Oper::INPUT)
          continue;
        if (localOutMap[inp->id].size() > 1) {
          parentIsFork = true;
          break;
        }
      }

      if (parentIsFork) {
        // Find which child index we are among the fork's consumers
        for (Node *inp : n->inputs) {
          if (inp->operation == Oper::INPUT)
            continue;
          if (localOutMap[inp->id].size() > 1) {
            auto &consumers = localOutMap[inp->id];
            for (int ci = 0; ci < (int)consumers.size(); ci++) {
              if (consumers[ci] == n) {
                int assignedStream = (ci + 1) % numStreams;
                nodeStreamId[n->id] = assignedStream;
                break;
              }
            }
            break;
          }
        }
      } else {
        nodeStreamId[n->id] = parentStream;
      }
    } else {
      // Merge point: multiple streams converge. Use stream 0.
      nodeStreamId[n->id] = 0;
    }

    // Ensure we have an entry
    if (!nodeStreamId.count(n->id)) {
      nodeStreamId[n->id] = 0;
    }
  }
}

void Graph::destroyStreams() {
  for (auto &kv : nodeEvents) {
    cudaEventDestroy(kv.second);
  }
  nodeEvents.clear();
  for (auto &s : streams) {
    cudaStreamDestroy(s);
  }
  streams.clear();
  numStreams = 0;
}

Graph::~Graph() {
  destroyStreams();
  for (auto &kv : nodeMemMap) {
    cudaFree(kv.second);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: dispatch a single node on the given stream (used by execute & bench)
// ─────────────────────────────────────────────────────────────────────────────
static void dispatchNode(Node *node, std::unordered_map<int, float *> &mem,
                         cudaStream_t stream) {
  if (node->operation == Oper::MATMUL) {
    int row_A = node->inputs[0]->shape[0];
    int N = node->inputs[0]->shape[1];
    int col_B = node->inputs[1]->shape[1];
    matMul(mem[node->inputs[0]->id], mem[node->inputs[1]->id], mem[node->id],
           row_A, N, col_B, stream);

  } else if (node->operation == Oper::ADD) {
    int height = node->shape[0];
    int width = node->shape[1];
    matAdd(mem[node->inputs[0]->id], mem[node->inputs[1]->id], mem[node->id],
           height, width, stream);

  } else if (node->operation == Oper::ReLU) {
    int height = node->shape[0];
    int width = node->shape[1];
    matReLU(mem[node->inputs[0]->id], mem[node->id], height, width, stream);

  } else if (node->operation == Oper::FUSED_MR) {
    int row_A = node->inputs[0]->shape[0];
    int N = node->inputs[0]->shape[1];
    int col_B = node->inputs[1]->shape[1];
    matMulReLU(mem[node->inputs[0]->id], mem[node->inputs[1]->id],
               mem[node->id], row_A, N, col_B, stream);

  } else if (node->operation == Oper::FUSED_AR) {
    int height = node->shape[0];
    int width = node->shape[1];
    matAddReLU(mem[node->inputs[0]->id], mem[node->inputs.back()->id],
               mem[node->id], height, width, stream);

  } else if (node->operation == Oper::FUSED_MAR) {
    int row_A = node->inputs[0]->shape[0];
    int N = node->inputs[0]->shape[1];
    int col_B = node->inputs[1]->shape[1];
    matMulAddReLU(mem[node->inputs[0]->id], mem[node->inputs[1]->id],
                  mem[node->inputs.back()->id], mem[node->id], row_A, N, col_B,
                  stream);

  } else if (node->operation == Oper::FUSED_MA) {
    int row_A = node->inputs[0]->shape[0];
    int N = node->inputs[0]->shape[1];
    int col_B = node->inputs[1]->shape[1];
    matMulAdd(mem[node->inputs[0]->id], mem[node->inputs[1]->id],
              mem[node->inputs.back()->id], mem[node->id], row_A, N, col_B,
              stream);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: for stream-aware execution, synchronize merge points by waiting on
// events from all input streams before launching on the merge stream.
// ─────────────────────────────────────────────────────────────────────────────
static void syncMergePoint(Node *node, std::unordered_map<int, int> &streamMap,
                           std::unordered_map<int, cudaEvent_t> &events,
                           std::vector<cudaStream_t> &streamPool) {
  int myStream = streamMap.count(node->id) ? streamMap[node->id] : 0;

  // Collect distinct input streams
  std::set<int> inputStreams;
  for (Node *inp : node->inputs) {
    if (streamMap.count(inp->id)) {
      int s = streamMap[inp->id];
      if (s != myStream)
        inputStreams.insert(s);
    }
  }

  // For each distinct input stream, record an event (if not already) and wait
  for (int s : inputStreams) {
    // Find the latest node on stream s that is an input to this node
    for (Node *inp : node->inputs) {
      if (streamMap.count(inp->id) && streamMap[inp->id] == s) {
        if (!events.count(inp->id)) {
          cudaEvent_t ev;
          cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
          cudaEventRecord(ev, streamPool[s]);
          events[inp->id] = ev;
        }
        cudaStreamWaitEvent(streamPool[myStream], events[inp->id], 0);
      }
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
      size_t byteSize =
          sorted[i]->shape[0] * sorted[i]->shape[1] * sizeof(float);
      std::vector<float> cont =
          readFloatsFromFile(sorted[i]->name + ".bin", byteSize);

      cudaMemcpy(nodeMemMap[sorted[i]->id], cont.data(), byteSize,
                 cudaMemcpyHostToDevice);
    } else {
      inputTill = i;
      break;
    }
  }

  for (int i = inputTill; i < sorted.size(); i++) {
    dispatchNode(sorted[i], nodeMemMap, 0);
  }

  Node *node = getOutput();
  std::vector<float> output(node->shape[0] * node->shape[1]);
  std::cout << std::endl;
  std::cout << "Final Node: " << node->name << std::endl;
  cudaMemcpy(output.data(), nodeMemMap[node->id],
             node->shape[0] * node->shape[1] * sizeof(float),
             cudaMemcpyDeviceToHost);
  writeVectorToFile(output.data(), output.size(), node->name + ".bin");
}

void Graph::execute(std::vector<Node *> fusedGraphs) {
  nodeMemMap.clear();
  nodeMemMap = nodeMem();

  // Assign streams for parallel execution
  assignStreams(fusedGraphs);

  int inputTill = 0;

  for (int i = 0; i < fusedGraphs.size(); i++) {
    if (fusedGraphs[i]->operation == Oper::INPUT) {
      size_t byteSize =
          fusedGraphs[i]->shape[0] * fusedGraphs[i]->shape[1] * sizeof(float);
      std::vector<float> cont =
          readFloatsFromFile(fusedGraphs[i]->name + ".bin", byteSize);

      cudaMemcpy(nodeMemMap[fusedGraphs[i]->id], cont.data(), byteSize,
                 cudaMemcpyHostToDevice);
    } else {
      inputTill = i;
      break;
    }
  }

  for (int i = inputTill; i < fusedGraphs.size(); i++) {
    Node *n = fusedGraphs[i];
    int sid = nodeStreamId.count(n->id) ? nodeStreamId[n->id] : 0;
    cudaStream_t stream = (sid < (int)streams.size()) ? streams[sid] : 0;

    // Synchronize at merge points
    syncMergePoint(n, nodeStreamId, nodeEvents, streams);

    dispatchNode(n, nodeMemMap, stream);
  }

  // Synchronize all streams before reading output
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  Node *node = fusedGraphs[fusedGraphs.size() - 1];
  std::vector<float> output(node->shape[0] * node->shape[1]);
  std::cout << std::endl;
  std::cout << "Final Node: " << node->name << std::endl;
  cudaMemcpy(output.data(), nodeMemMap[node->id],
             node->shape[0] * node->shape[1] * sizeof(float),
             cudaMemcpyDeviceToHost);
  writeVectorToFile(output.data(), output.size(),
                    node->name + "_fusedGraph.bin");
}

float Graph::benchExecution() {
  if (nodeMemMap.size() < nodes.size()) {
    nodeMemMap = nodeMem();
  }

  // warmup
  for (int i = 0; i < sorted.size(); i++) {
    if (sorted[i]->operation == Oper::INPUT)
      continue;
    dispatchNode(sorted[i], nodeMemMap, 0);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int run = 0; run < 100; run++) {
    for (int i = 0; i < sorted.size(); i++) {
      if (sorted[i]->operation == Oper::INPUT)
        continue;
      dispatchNode(sorted[i], nodeMemMap, 0);
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

float Graph::benchExecution(std::vector<Node *> fusedGraphs) {
  nodeMemMap.clear();
  nodeMemMap = nodeMem();

  // Assign streams for parallel execution
  assignStreams(fusedGraphs);

  // Identify non-INPUT start index
  int inputTill = 0;
  for (int i = 0; i < fusedGraphs.size(); i++) {
    if (fusedGraphs[i]->operation != Oper::INPUT) {
      inputTill = i;
      break;
    }
  }

  // warmup — with streams
  for (int i = inputTill; i < fusedGraphs.size(); i++) {
    Node *n = fusedGraphs[i];
    int sid = nodeStreamId.count(n->id) ? nodeStreamId[n->id] : 0;
    cudaStream_t stream = (sid < (int)streams.size()) ? streams[sid] : 0;
    syncMergePoint(n, nodeStreamId, nodeEvents, streams);
    dispatchNode(n, nodeMemMap, stream);
  }
  cudaDeviceSynchronize();

  // Clear events from warmup
  for (auto &kv : nodeEvents) {
    cudaEventDestroy(kv.second);
  }
  nodeEvents.clear();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int run = 0; run < 100; run++) {
    // Clear events from previous iteration
    for (auto &kv : nodeEvents) {
      cudaEventDestroy(kv.second);
    }
    nodeEvents.clear();

    for (int i = inputTill; i < fusedGraphs.size(); i++) {
      Node *n = fusedGraphs[i];
      int sid = nodeStreamId.count(n->id) ? nodeStreamId[n->id] : 0;
      cudaStream_t stream = (sid < (int)streams.size()) ? streams[sid] : 0;
      syncMergePoint(n, nodeStreamId, nodeEvents, streams);
      dispatchNode(n, nodeMemMap, stream);
    }
    // Sync all streams at end of each iteration to ensure correct ordering
    for (int s = 0; s < numStreams; s++) {
      cudaStreamSynchronize(streams[s]);
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

Node *Graph::fuseNodes(std::vector<Node *> nodes2Fuse,
                       std::vector<int> &fusedMap) {
  std::unordered_set<Node *> nodies(nodes2Fuse.begin(), nodes2Fuse.end());
  std::vector<Node *> inputting;
  for (Node *item : nodes2Fuse) {
    for (Node *val : item->inputs) {
      if (nodies.find(val) == nodies.end())
        inputting.push_back(val);
    }
  }
  std::string identifier = "";
  for (Node *item : nodes2Fuse) {
    if (item->operation == Oper::MATMUL)
      identifier += "M";
    else if (item->operation == Oper::ADD)
      identifier += "A";
    else if (item->operation == Oper::ReLU)
      identifier += "R";
  }

  Node *fusedNode = nullptr;

  if (identifier == "MA")
    fusedNode = addNode("Fused_MA", Oper::FUSED_MA, inputting);
  else if (identifier == "MR")
    fusedNode = addNode("Fused_MR", Oper::FUSED_MR, inputting);
  else if (identifier == "AR")
    fusedNode = addNode("Fused_AR", Oper::FUSED_AR, inputting);
  else if (identifier == "MAR")
    fusedNode = addNode("Fused_MAR", Oper::FUSED_MAR, inputting);

  if (!fusedNode) {
    std::cout << "Error: unknown fusion pattern: " << identifier << "\n";
    return nullptr;
  }

  for (Node *item : outMap[nodes2Fuse[nodes2Fuse.size() - 1]]) {
    auto it =
        std::find(item->inputs.begin(), item->inputs.end(), nodes2Fuse.back());
    if (it != item->inputs.end())
      *it = fusedNode;
  }

  fusedNode->shape = nodes2Fuse[nodes2Fuse.size() - 1]->shape;

  for (Node *item : nodes2Fuse) {
    fusedMap[item->id] = 1;
  }

  return fusedNode;
}

std::vector<Node *> Graph::fuseDFSMerger(Node *node, std::vector<int> &visited,
                                         bool matmul, bool add, bool relu) {
  if (outMap[node].size() > 1 || outMap[node].size() == 0) {
    return {node};
  }

  visited[node->id] = 1;
  std::vector<Node *> ans;

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

std::vector<Node *> Graph::fuseDFS(Node *node, std::vector<int> &visited,
                                   bool matmul, bool add, bool relu) {
  int count = 0;
  for (Node *item : node->inputs) {
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
  std::vector<Node *> ans;

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

std::vector<Node *> Graph::fusionPass() {
  outMap.clear();
  std::vector<std::vector<Node *>> fusion;

  for (int i = 0; i < sorted.size(); i++) {
    if (!sorted[i]->inputs.empty()) {
      for (Node *item : sorted[i]->inputs) {
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
  std::vector<std::pair<int, int>> fuseableMap(sorted.size(),
                                               std::pair<int, int>(0, 0));

  for (Node *item : sorted) {
    if (item->operation == Oper::INPUT)
      continue;

    int count = 0;
    if (item->inputs.size() > 1) {
      for (Node *val : item->inputs) {
        if (val->operation == Oper::INPUT)
          continue;

        count++;
      }
    }

    if (count > 1) {
      mergerMap[item->id] = 1;
    }
  }

  int counter = -1;
  for (Node *item : sorted) {
    if (item->operation == Oper::INPUT)
      continue;

    std::vector<Node *> toFuse;

    if (!visited[item->id] && !mergerMap[item->id]) {
      toFuse = fuseDFS(item, visited, 0, 0, 0);
    }

    if (mergerMap[item->id]) {

      toFuse = fuseDFSMerger(item, visited, 0, 0, 0);
    }

    if (toFuse.size() > 1) {
      reverse(toFuse.begin(), toFuse.end());
      counter = fusion.size();
      for (Node *val : toFuse) {
        fuseableMap[val->id] = {1, counter};
      }
      fusion.push_back(toFuse);
    }
  }
  std::vector<int> fusedMap(sorted.size(), 0);
  std::vector<Node *> fusedSorted;

  for (Node *item : sorted) {
    if (fuseableMap[item->id].first) {
      if (!fusedMap[item->id]) {
        Node *newFusedNode =
            fuseNodes(fusion[fuseableMap[item->id].second], fusedMap);
        fusedSorted.push_back(newFusedNode);
      }
    } else {
      fusedSorted.push_back(item);
    }
  }

  return fusedSorted;
}

void Graph::setOutput(Node *node) { outputNode = node; }

Node *Graph::getOutput() { return outputNode; }

Node *Graph::addInput(std::string nm, std::vector<int> shp) {
  std::unique_ptr<Node> newNode = std::make_unique<Node>();
  newNode->id = nodes.size();
  newNode->name = nm;
  newNode->shape = shp;
  newNode->operation = Oper::INPUT;
  Node *raw = newNode.get();
  nodes.push_back(std::move(newNode));

  return raw;
}

Node *Graph::addNode(std::string nm, Oper op, std::vector<Node *> in) {
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
    if (in[0]->shape[0] != in[1]->shape[0] ||
        in[0]->shape[1] != in[1]->shape[1]) {
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

  Node *raw = newNode.get();
  nodes.push_back(std::move(newNode));

  return raw;
}

void Graph::printGraph() {
  for (auto &item : nodes) {
    std::cout << "Node Name: " << item->name << "\n";
    std::cout << "ID: " << item->id << "\n";
    std::cout << "Operation: " << op2String(item->operation) << "\n";
    std::cout << "Shape: (";
    for (int i = 0; i < item->shape.size(); i++) {
      std::cout << item->shape[i];
      if (i < item->shape.size() - 1)
        std::cout << ", ";
    }
    std::cout << ")\n";
    if (!item->inputs.empty()) {
      std::cout << "Input Nodes: ";
      for (Node *node : item->inputs) {
        std::cout << node->name << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
    std::cout << std::endl;
    std::cout << std::endl;
  }
}

void Graph::printNode(Node *item) {
  std::cout << "Node Name: " << item->name << "\n";
  std::cout << "ID: " << item->id << "\n";
  std::cout << "Operation: " << op2String(item->operation) << "\n";
  std::cout << "Shape: (";
  for (int i = 0; i < item->shape.size(); i++) {
    std::cout << item->shape[i];
    if (i < item->shape.size() - 1)
      std::cout << ", ";
  }
  std::cout << ")\n";
  if (!item->inputs.empty()) {
    std::cout << "Input Nodes: ";
    for (Node *node : item->inputs) {
      std::cout << node->name << "(id: " << node->id << ")" << " ";
    }
    std::cout << "\n";
  }
}

std::vector<Node *> Graph::topoSort() {
  int nodesNum = nodes.size();
  std::vector<int> indegree(nodesNum);
  std::unordered_map<int, std::vector<Node *>> adjList;

  for (int i = 0; i < nodesNum; i++) {
    Node *node = nodes[i].get();
    indegree[i] = node->inputs.size();
    for (Node *item : node->inputs) {
      adjList[item->id].push_back(node);
    }
  }

  std::queue<Node *> que;
  for (int i = 0; i < nodesNum; i++) {
    if (indegree[i] == 0) {
      Node *node = nodes[i].get();
      que.push(node);
    }
  }

  std::vector<Node *> topo;
  while (!que.empty()) {
    Node *node = que.front();
    que.pop();
    topo.push_back(node);

    for (Node *item : adjList[node->id]) {
      indegree[item->id]--;
      if (indegree[item->id] == 0) {
        que.push(item);
      }
    }
  }

  if (topo.size() != nodesNum) {
    std::cout
        << "Error: cycle detected in graph, topological sort incomplete\n";
    return {};
  }

  for (Node *item : topo) {
    sorted.push_back(item);
  }

  return topo;
}

std::vector<Node *> Graph::topoSort(std::vector<Node *> newNodes) {
  int nodesNum = newNodes.size();
  std::vector<int> indegree(nodesNum);
  std::unordered_map<int, std::vector<Node *>> adjList;

  for (int i = 0; i < nodesNum; i++) {
    Node *node = newNodes[i];
    indegree[i] = node->inputs.size();
    for (Node *item : node->inputs) {
      adjList[item->id].push_back(node);
    }
  }

  std::queue<Node *> que;
  for (int i = 0; i < nodesNum; i++) {
    if (indegree[i] == 0) {
      Node *node = newNodes[i];
      que.push(node);
    }
  }

  std::vector<Node *> topo;
  while (!que.empty()) {
    Node *node = que.front();
    que.pop();
    topo.push_back(node);

    for (Node *item : adjList[node->id]) {
      indegree[item->id]--;
      if (indegree[item->id] == 0) {
        que.push(item);
      }
    }
  }

  if (topo.size() != nodesNum) {
    std::cout << "Topological Sort size: " << topo.size() << std::endl;
    std::cout << "newNodes size: " << nodesNum << std::endl;
    std::cout
        << "Error: cycle detected in graph, topological sort incomplete\n";
    return {};
  }

  return topo;
}
