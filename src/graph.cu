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
#include <fstream>
#include <sstream>

float *Graph::bufferAlloc(Node *node) {
  float *address;
  size_t bytesNode = node->shape[0] * node->shape[1];

  cudaMalloc((void **)&address, bytesNode * sizeof(float));

  return address;
}

std::unordered_map<int, float *>
Graph::nodeMem(const std::vector<Node *> &schedule) {
  std::unordered_map<int, float *> mp;
  std::map<size_t, std::vector<float *>> freePool;
  std::unordered_map<int, int> outDegree;

  for (Node *n : schedule) {
    for (Node *inp : n->inputs) {
      outDegree[inp->id]++;
    }
  }

  for (Node *n : schedule) {
    size_t byteSize = n->shape[0] * n->shape[1] * sizeof(float);

    if (!freePool[byteSize].empty()) {
      mp[n->id] = freePool[byteSize].back();
      freePool[byteSize].pop_back();
    } else {
      mp[n->id] = bufferAlloc(n);
    }

    for (Node *inp : n->inputs) {
      outDegree[inp->id]--;
      if (outDegree[inp->id] == 0 && inp->operation != Oper::INPUT) {
        size_t inpSize = inp->shape[0] * inp->shape[1] * sizeof(float);
        freePool[inpSize].push_back(mp[inp->id]);
      }
    }
  }

  return mp;
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

void Graph::freeMem() {
  std::unordered_set<float *> uniquePtrs;
  for (auto &kv : nodeMemMap) {
    uniquePtrs.insert(kv.second);
  }
  for (float *ptr : uniquePtrs) {
    cudaFree(ptr);
  }
  nodeMemMap.clear();
}

Graph::~Graph() {
  destroyStreams();
  freeMem();
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
    int b_rows = node->inputs[1]->shape[0];
    int b_cols = node->inputs[1]->shape[1];
    matAdd(mem[node->inputs[0]->id], mem[node->inputs[1]->id], mem[node->id],
           height, width, b_rows, b_cols, stream);

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
    int b_rows = node->inputs[1]->shape[0];
    int b_cols = node->inputs[1]->shape[1];
    matAddReLU(mem[node->inputs[0]->id], mem[node->inputs[1]->id],
               mem[node->id], height, width, b_rows, b_cols, stream);

  } else if (node->operation == Oper::FUSED_MA) {
    int row_A = node->inputs[0]->shape[0];
    int N = node->inputs[0]->shape[1];
    int col_B = node->inputs[1]->shape[1];
    int b_rows = node->inputs[2]->shape[0];
    int b_cols = node->inputs[2]->shape[1];
    matMulAdd(mem[node->inputs[0]->id], mem[node->inputs[1]->id],
              mem[node->inputs[2]->id], mem[node->id], row_A, N, col_B, b_rows, b_cols,
              stream);

  } else if (node->operation == Oper::FUSED_MAR) {
    int row_A = node->inputs[0]->shape[0];
    int N = node->inputs[0]->shape[1];
    int col_B = node->inputs[1]->shape[1];
    int b_rows = node->inputs[2]->shape[0];
    int b_cols = node->inputs[2]->shape[1];
    matMulAddReLU(mem[node->inputs[0]->id], mem[node->inputs[1]->id],
                  mem[node->inputs[2]->id], mem[node->id], row_A, N, col_B, b_rows, b_cols,
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
    freeMem();
    nodeMemMap = nodeMem(sorted);
  }
  int inputTill = 0;

  for (int i = 0; i < sorted.size(); i++) {
    if (sorted[i]->operation == Oper::INPUT) {
      size_t byteSize =
          sorted[i]->shape[0] * sorted[i]->shape[1] * sizeof(float);
      
      std::string filename = sorted[i]->name + ".bin";
      std::vector<float> cont = readFloatsFromFile(filename, byteSize);
      
      // If the input file (like dynamic X.bin) doesn't exist, generate it on the fly!
      if (cont.empty()) {
          generateAndSaveInput(sorted[i]);
          cont = readFloatsFromFile(filename, byteSize);
      }

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
  freeMem();
  nodeMemMap = nodeMem(fusedGraphs);

  // Assign streams for parallel execution
  assignStreams(fusedGraphs);

  int inputTill = 0;

  for (int i = 0; i < fusedGraphs.size(); i++) {
    if (fusedGraphs[i]->operation == Oper::INPUT) {
      size_t byteSize =
          fusedGraphs[i]->shape[0] * fusedGraphs[i]->shape[1] * sizeof(float);
      
      std::string filename = fusedGraphs[i]->name + ".bin";
      std::vector<float> cont = readFloatsFromFile(filename, byteSize);
      
      if (cont.empty()) {
          generateAndSaveInput(fusedGraphs[i]);
          cont = readFloatsFromFile(filename, byteSize);
      }

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
    freeMem();
    nodeMemMap = nodeMem(sorted);
  }

  int inputTill = 0;
  for (int i = 0; i < sorted.size(); i++) {
    if (sorted[i]->operation != Oper::INPUT) {
      inputTill = i;
      break;
    }
  }

  cudaStream_t execStream;
  cudaStreamCreate(&execStream);

  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  cudaStreamBeginCapture(execStream, cudaStreamCaptureModeGlobal);
  for (int i = inputTill; i < sorted.size(); i++) {
    dispatchNode(sorted[i], nodeMemMap, execStream);
  }
  cudaStreamEndCapture(execStream, &graph);
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  // warmup
  cudaGraphLaunch(graphExec, execStream);
  cudaStreamSynchronize(execStream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, execStream);
  for (int run = 0; run < 100; run++) {
    cudaGraphLaunch(graphExec, execStream);
  }
  cudaEventRecord(stop, execStream);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(execStream);

  return ms / 100.0f;
}

float Graph::benchExecution(std::vector<Node *> fusedGraphs) {
  freeMem();
  nodeMemMap = nodeMem(fusedGraphs);

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

  // Pre-create events for syncMergePoint since we can't safely dynamically
  // create them during capture
  for (int i = inputTill; i < fusedGraphs.size(); i++) {
    Node *n = fusedGraphs[i];
    int myStream = nodeStreamId.count(n->id) ? nodeStreamId[n->id] : 0;
    std::set<int> inputStreams;
    for (Node *inp : n->inputs) {
      if (nodeStreamId.count(inp->id)) {
        int s = nodeStreamId[inp->id];
        if (s != myStream)
          inputStreams.insert(s);
      }
    }
    for (int s : inputStreams) {
      for (Node *inp : n->inputs) {
        if (nodeStreamId.count(inp->id) && nodeStreamId[inp->id] == s) {
          if (!nodeEvents.count(inp->id)) {
            cudaEvent_t ev;
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            nodeEvents[inp->id] = ev;
          }
        }
      }
    }
  }

  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);

  // Fork: make all other streams enter capture mode by waiting on an event recorded in stream 0
  cudaEvent_t forkEvent;
  cudaEventCreateWithFlags(&forkEvent, cudaEventDisableTiming);
  cudaEventRecord(forkEvent, streams[0]);
  for (int s = 1; s < numStreams; s++) {
    cudaStreamWaitEvent(streams[s], forkEvent, 0);
  }

  for (int i = inputTill; i < fusedGraphs.size(); i++) {
    Node *n = fusedGraphs[i];
    int sid = nodeStreamId.count(n->id) ? nodeStreamId[n->id] : 0;
    cudaStream_t stream = (sid < (int)streams.size()) ? streams[sid] : 0;

    // Instead of creating events, this will reuse the pre-created ones and
    // record/wait
    syncMergePoint(n, nodeStreamId, nodeEvents, streams);

    dispatchNode(n, nodeMemMap, stream);
  }

  // Fork-join: ensure all child streams finish before the main stream's graph
  // capture completes
  std::vector<cudaEvent_t> captureSyncEvents;
  for (int s = 1; s < numStreams; s++) {
    cudaEvent_t syncEvent;
    cudaEventCreateWithFlags(&syncEvent, cudaEventDisableTiming);
    cudaEventRecord(syncEvent, streams[s]);
    cudaStreamWaitEvent(streams[0], syncEvent, 0);
    captureSyncEvents.push_back(syncEvent);
  }

  cudaStreamEndCapture(streams[0], &graph);
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  // Cleanup the temporary fork-join events used during capture
  for (auto ev : captureSyncEvents) {
    cudaEventDestroy(ev);
  }
  cudaEventDestroy(forkEvent);

  // warmup
  cudaGraphLaunch(graphExec, streams[0]);
  cudaStreamSynchronize(streams[0]);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, streams[0]);
  for (int run = 0; run < 100; run++) {
    cudaGraphLaunch(graphExec, streams[0]);
  }
  cudaEventRecord(stop, streams[0]);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);

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
    bool exact_match = (in[0]->shape[0] == in[1]->shape[0] && in[0]->shape[1] == in[1]->shape[1]);
    bool broadcast_match = ((in[1]->shape[0] == 1 && in[1]->shape[1] == in[0]->shape[1]) || 
                            (in[1]->shape[1] == 1 && in[1]->shape[0] == in[0]->shape[0]));
    
    if (!exact_match && !broadcast_match) {
      std::cout << "Error: ADD shape mismatch (" << in[0]->shape[0] << "x" << in[0]->shape[1] << " + "
                << in[1]->shape[0] << "x" << in[1]->shape[1] << ")\n";
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

void Graph::loadFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "Failed to open IR file: " << filename << std::endl;
    return;
  }

  std::string line;
  std::unordered_map<std::string, Node*> nameToNode;

  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string type;
    ss >> type;

    if (type == "INPUT") {
      std::string name;
      int d1, d2;
      ss >> name >> d1 >> d2;
      Node* n = addInput(name, {d1, d2});
      nameToNode[name] = n;
    } else if (type == "OUTPUT") {
      std::string name;
      ss >> name;
      if (nameToNode.count(name)) {
        setOutput(nameToNode[name]);
      }
    } else {
      // Operations: MATMUL, ADD, RELU
      std::string outName;
      ss >> outName;
      
      std::vector<Node*> inputs;
      std::string inName;
      while (ss >> inName) {
        if (nameToNode.count(inName)) {
          inputs.push_back(nameToNode[inName]);
        }
      }

      Oper op;
      if (type == "MATMUL") op = Oper::MATMUL;
      else if (type == "ADD") op = Oper::ADD;
      else if (type == "RELU") op = Oper::ReLU;
      else continue;

      Node* n = addNode(outName, op, inputs);
      nameToNode[outName] = n;
    }
  }
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

#include <sstream>

std::string readFileContents(const std::string& path, bool stripIncludes) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    
    if (!stripIncludes) {
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }
    
    std::string line;
    std::stringstream ss;
    while (std::getline(file, line)) {
        if (line.find("#include <fileio.h>") != std::string::npos ||
            line.find("#include <kernel.h>") != std::string::npos ||
            line.find("#include <graph.h>") != std::string::npos ||
            line.find("#include <node.h>") != std::string::npos ||
            line.find("#include \"fileio.h\"") != std::string::npos ||
            line.find("#include \"kernel.h\"") != std::string::npos) {
            continue;
        }
        ss << line << "\n";
    }
    return ss.str();
}

void Graph::compileToFile(const std::string &filename, const std::vector<Node *> &targetGraphs, bool standalone) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }

    if (nodeMemMap.size() < nodes.size()) {
        freeMem();
        nodeMemMap = nodeMem(targetGraphs);
    }
    assignStreams(targetGraphs);

    // Build localOutMap for static stream dependency tracking
    std::unordered_map<int, std::vector<Node *>> localOutMap;
    std::unordered_map<int, bool> isInSchedule;
    for (Node *n : targetGraphs) isInSchedule[n->id] = true;
    for (Node *n : targetGraphs) {
        for (Node *inp : n->inputs) {
            if (isInSchedule.count(inp->id)) localOutMap[inp->id].push_back(n);
        }
    }

    out << "// Auto-generated by Graph-to-CUDA Compiler\n";
    out << "#include <iostream>\n";
    out << "#include <vector>\n";
    out << "#include <string>\n";
    
    if (standalone) {
        out << "\n// --- BEGIN INJECTED DEPENDENCIES ---\n";
        out << readFileContents("include/fileio.h", true) << "\n";
        out << readFileContents("include/kernel.h", true) << "\n";
        out << readFileContents("utils/fileio.cu", true) << "\n";
        out << readFileContents("src/kernel.cu", true) << "\n";
        out << "// --- END INJECTED DEPENDENCIES ---\n\n";
    } else {
        out << "#include <fileio.h>\n";
        out << "#include <kernel.h>\n";
    }

    out << "int main() {\n";
    
    // Allocations
    for (auto &kv : nodeMemMap) {
        Node *n = nodes[kv.first].get();
        size_t bytes = n->shape[0] * n->shape[1] * sizeof(float);
        out << "    float* mem_" << n->id << ";\n";
        out << "    cudaMalloc(&mem_" << n->id << ", " << bytes << ");\n";
    }

    out << "\n    // Load Inputs\n";
    int inputTill = 0;
    for (int i = 0; i < targetGraphs.size(); i++) {
        Node *n = targetGraphs[i];
        if (n->operation == Oper::INPUT) {
            size_t bytes = n->shape[0] * n->shape[1] * sizeof(float);
            out << "    {\n";
            out << "        std::vector<float> cont = readFloatsFromFile(\"" << n->name << ".bin\", " << bytes << ");\n";
            out << "        if (cont.empty()) {\n";
            out << "            std::cerr << \"Failed to load " << n->name << ".bin\\n\";\n";
            out << "            return 1;\n";
            out << "        }\n";
            out << "        cudaMemcpy(mem_" << n->id << ", cont.data(), " << bytes << ", cudaMemcpyHostToDevice);\n";
            out << "    }\n";
        } else {
            inputTill = i;
            break;
        }
    }

    out << "\n    // Create Streams\n";
    for (int i = 0; i < numStreams; i++) {
        out << "    cudaStream_t stream_" << i << ";\n";
        out << "    cudaStreamCreate(&stream_" << i << ");\n";
    }
    out << "\n    // Create Events\n";
    for (auto &kv : nodeEvents) {
        out << "    cudaEvent_t event_" << kv.first << ";\n";
        out << "    cudaEventCreate(&event_" << kv.first << ");\n";
    }

    out << "\n    // Execution\n";
    for (int i = inputTill; i < targetGraphs.size(); i++) {
        Node *n = targetGraphs[i];
        int sid = nodeStreamId.count(n->id) ? nodeStreamId[n->id] : 0;
        
        // Emulate syncMergePoint statically
        if (localOutMap[n->id].size() > 1) {
            out << "    cudaEventRecord(event_" << n->id << ", stream_" << sid << ");\n";
        }
        if (n->inputs.size() > 1 && n->operation != Oper::INPUT) {
            for (Node *inp : n->inputs) {
                if (inp->operation == Oper::INPUT) continue;
                if (localOutMap[inp->id].size() > 1) {
                    out << "    cudaStreamWaitEvent(stream_" << sid << ", event_" << inp->id << ", 0);\n";
                }
            }
        }
        
        // Emulate dispatchNode statically
        std::string s_arg = "stream_" + std::to_string(sid);
        if (n->operation == Oper::MATMUL) {
            out << "    matMul(mem_" << n->inputs[0]->id << ", mem_" << n->inputs[1]->id << ", mem_" << n->id << ", "
                << n->inputs[0]->shape[0] << ", " << n->inputs[0]->shape[1] << ", " << n->inputs[1]->shape[1] << ", " << s_arg << ");\n";
        } else if (n->operation == Oper::ADD) {
            out << "    matAdd(mem_" << n->inputs[0]->id << ", mem_" << n->inputs[1]->id << ", mem_" << n->id << ", "
                << n->shape[0] << ", " << n->shape[1] << ", " << n->inputs[1]->shape[0] << ", " << n->inputs[1]->shape[1] << ", " << s_arg << ");\n";
        } else if (n->operation == Oper::ReLU) {
            out << "    matReLU(mem_" << n->inputs[0]->id << ", mem_" << n->id << ", "
                << n->shape[0] << ", " << n->shape[1] << ", " << s_arg << ");\n";
        } else if (n->operation == Oper::FUSED_MR) {
            out << "    matMulReLU(mem_" << n->inputs[0]->id << ", mem_" << n->inputs[1]->id << ", mem_" << n->id << ", "
                << n->inputs[0]->shape[0] << ", " << n->inputs[0]->shape[1] << ", " << n->inputs[1]->shape[1] << ", " << s_arg << ");\n";
        } else if (n->operation == Oper::FUSED_AR) {
            out << "    matAddReLU(mem_" << n->inputs[0]->id << ", mem_" << n->inputs[1]->id << ", mem_" << n->id << ", "
                << n->shape[0] << ", " << n->shape[1] << ", " << n->inputs[1]->shape[0] << ", " << n->inputs[1]->shape[1] << ", " << s_arg << ");\n";
        } else if (n->operation == Oper::FUSED_MA) {
            out << "    matMulAdd(mem_" << n->inputs[0]->id << ", mem_" << n->inputs[1]->id << ", mem_" << n->inputs[2]->id << ", mem_" << n->id << ", "
                << n->inputs[0]->shape[0] << ", " << n->inputs[0]->shape[1] << ", " << n->inputs[1]->shape[1] << ", "
                << n->inputs[2]->shape[0] << ", " << n->inputs[2]->shape[1] << ", " << s_arg << ");\n";
        } else if (n->operation == Oper::FUSED_MAR) {
            out << "    matMulAddReLU(mem_" << n->inputs[0]->id << ", mem_" << n->inputs[1]->id << ", mem_" << n->inputs[2]->id << ", mem_" << n->id << ", "
                << n->inputs[0]->shape[0] << ", " << n->inputs[0]->shape[1] << ", " << n->inputs[1]->shape[1] << ", "
                << n->inputs[2]->shape[0] << ", " << n->inputs[2]->shape[1] << ", " << s_arg << ");\n";
        }
    }

    out << "\n    // Synchronize Streams\n";
    for (int i = 0; i < numStreams; i++) {
        out << "    cudaStreamSynchronize(stream_" << i << ");\n";
    }

    out << "\n    // Read Output\n";
    Node *outNode = targetGraphs.back();
    size_t outBytes = outNode->shape[0] * outNode->shape[1] * sizeof(float);
    out << "    std::vector<float> output(" << (outNode->shape[0] * outNode->shape[1]) << ");\n";
    out << "    cudaMemcpy(output.data(), mem_" << outNode->id << ", " << outBytes << ", cudaMemcpyDeviceToHost);\n";
    out << "    writeVectorToFile(output.data(), output.size(), \"" << outNode->name << "_static.bin\");\n";
    out << "    std::cout << \"Successfully executed static graph! Output written to " << outNode->name << "_static.bin\\n\";\n";
    
    out << "\n    return 0;\n";
    out << "}\n";
    out.close();
    std::cout << "Successfully generated " << filename << "\n";
}
