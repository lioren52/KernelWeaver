# KernelWeaver

A neural network graph compiler built from scratch in C++/CUDA. No frameworks, no cuBLAS, no cuDNN — just raw CUDA kernels, a custom IR, and a whole lot of fun building it.

Takes a computation graph (hardcoded or imported from ONNX), runs operator fusion, schedules it across multiple CUDA streams, and either executes it on the GPU or emits a standalone `.cu` file you can compile and run independently.

## Compiler Pipeline

```
             PyTorch Model
                    │
                    ▼
             ONNX Frontend
                    │
                    ▼
          Frontend Import Pass
                    │
                    ▼
        Custom Intermediate Representation
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
 Dependency Analysis      Shape Resolution
         │
         ▼
 Topological Scheduling
         │
         ▼
 Pattern-Based Fusion Pass
         │
         ▼
 GPU Memory Planning
         │
         ▼
 Stream Scheduling
         │
         ▼
 CUDA Backend
         │
         ├─────────────┐
         ▼             ▼
 Runtime Execute   CUDA Codegen
                    (.cu)
```

KernelWeaver follows a traditional compiler architecture adapted for neural network computation graphs.

### Frontend

Models can either be

- Constructed directly in C++
- Imported from ONNX via a Python frontend

The frontend lowers the network into a lightweight custom intermediate representation consisting of graph nodes representing tensor operations.

### Optimization Passes

Once imported, the compiler performs several graph-level optimization passes:

- Dependency analysis
- Topological scheduling
- Pattern-based operator fusion
- Multi-stream scheduling
- GPU memory planning through liveness analysis

### Backend

The optimized graph can then either

- Execute directly through the runtime
- Be lowered into standalone CUDA source code for independent compilation.

## Frontend (ONNX Import)

A lightweight Python frontend imports ONNX models and lowers supported operators into the compiler's custom intermediate representation.

Currently supported ONNX operators include

- Gemm
- MatMul
- Add
- ReLU

Model weights are exported as binary blobs while graph structure is emitted as a simple textual IR consumed by the compiler.

## Intermediate Representation

KernelWeaver uses a lightweight line-oriented intermediate representation describing

- Inputs
- Operators
- Data dependencies
- Outputs

Example

```text
INPUT X 128 64
INPUT W 64 32
MATMUL mm X W
ADD add mm bias
RELU out add
OUTPUT out
```

This representation serves as the compiler's internal graph description before optimization and backend lowering.

## Memory Planning

Rather than allocating a new GPU buffer for every intermediate tensor, the compiler performs liveness analysis to determine when tensors become dead.

Buffers whose final consumer has completed execution are recycled through a size-indexed free list, reducing GPU memory consumption and allocation overhead.

Input tensors remain resident throughout execution since they correspond to persistent model parameters.

## Runtime Scheduler

Independent branches within the computation graph are scheduled onto separate CUDA streams.

Merge points synchronize execution through CUDA events (`cudaEventRecord` / `cudaStreamWaitEvent`).

For benchmarking, the complete execution graph is captured using CUDA Graphs, allowing low-overhead replay while preserving the optimized execution schedule.

## Backend: CUDA Code Generation

The backend lowers optimized graph nodes into hand-written CUDA kernels.

The generated execution schedule invokes specialized kernels for

- MatMul
- Add
- ReLU
- MatMul + ReLU
- MatMul + Add
- Add + ReLU
- MatMul + Add + ReLU

The matrix multiplication backend implements

- Warp-level tiling
- Register blocking
- Shared-memory tiling
- Vectorized float4 memory operations
- Shared-memory bank-conflict avoidance

Fusion occurs at the backend by folding element-wise operations directly into the GEMM epilogue, eliminating intermediate global memory traffic and unnecessary kernel launches.

## Supported Operators

| Operation | Status |
|---|---|
| `FUSED_MR` | MatMul + ReLU (activation fused into matmul epilogue) |
| `FUSED_AR` | Add + ReLU (vectorized `float4` element-wise) |
| `FUSED_MA` | MatMul + Add (bias fused into matmul epilogue) |
| `FUSED_MAR` | MatMul + Add + ReLU (the full dense-layer fusion) |

The key insight: by fusing bias addition and ReLU into the matmul's register write-back, we avoid entire round-trips to global memory. This is the same optimization that production compilers like TensorRT perform.

## Memory Management

The compiler implements **liveness-based buffer reuse** — it tracks the out-degree of each node, and when a tensor's last consumer has been dispatched, its GPU buffer gets recycled into a free pool keyed by size. INPUT node buffers are kept alive since they hold the loaded weights.

## Multi-Stream Execution

Independent branches in the graph (e.g., parallel paths that fork from the same node) are assigned to separate CUDA streams for concurrent execution. Merge points synchronize using `cudaEventRecord` / `cudaStreamWaitEvent`. The benchmarking path captures the entire multi-stream execution as a CUDA Graph for accurate, low-overhead replay.

## Project Structure

```
KernelWeaver/
├── main.cu                  # Entry point, CLI args, hardcoded benchmark graph
├── include/
│   ├── node.h               # Node class & Oper enum (MATMUL, ADD, ReLU, fused ops)
│   ├── graph.h               # Graph class — topo sort, fusion, execution, codegen
│   ├── kernel.h              # CUDA kernel host-wrapper declarations
│   └── fileio.h              # File I/O, op-to-string, random generation
├── src/
│   ├── graph.cu              # Core compiler — fusion pass, stream assignment,
│   │                         #   memory planning, execute, benchmark, codegen
│   └── kernel.cu             # All CUDA kernels (base + 4 fused variants)
├── utils/
│   └── fileio.cu             # Binary file read/write, input generation
├── export_test_model.py      # Export a simple PyTorch model to ONNX
└── onnx_exporter.py          # Convert ONNX → text IR (.graph) for the compiler
```

## Building

**Requirements:** NVIDIA GPU, CUDA Toolkit (tested with CUDA 12.x), `nvcc`

```bash
nvcc -o graph_compiler main.cu src/graph.cu src/kernel.cu utils/fileio.cu \
     -Iinclude -O3 -arch=sm_75
```

> Adjust `-arch=sm_75` to match your GPU's compute capability (e.g., `sm_86` for Ampere, `sm_89` for Ada Lovelace).

## Usage

### Default Benchmark (Hardcoded Graph)

Runs a complex multi-branch graph (4096-dim, 3 parallel branches, residual skip connection) and compares unfused vs fused execution:

```bash
./graph_compiler
```

This will:
- Build the graph and run topological sort
- Execute **unfused** (baseline) and **fused** versions
- Print a performance comparison like:

```
=====================================
          PERFORMANCE REPORT
=====================================
 Baseline: X.XX ms
 Fused:    X.XX ms
 Speedup:  X.XXx
=====================================
```

### Code Generation Mode

Generate a standalone `.cu` file from the optimized graph:

```bash
./graph_compiler --codegen
```

This emits `generated_model.cu` with all kernel calls, stream creation, event synchronization, and memory management baked in.

### Running with ONNX Models

**Step 1:** Export your PyTorch model to ONNX:

```bash
python export_test_model.py
```

**Step 2:** Convert ONNX to the compiler's text IR:

```bash
python onnx_exporter.py test_model.onnx
```

This generates `model.graph` (the text IR) and `.bin` files for all weights.

**Step 3:** Run the compiler on it:

```bash
./graph_compiler model.graph
```

Or generate code from it:

```bash
./graph_compiler model.graph --codegen
```

### Text IR Format

The `.graph` file is a simple line-based format:

```
INPUT X 128 64
INPUT fc1_weight 32 64
INPUT fc1_bias 1 32
MATMUL matmul_out X fc1_weight
ADD add_out matmul_out fc1_bias
RELU relu_out add_out
OUTPUT relu_out
```

## Performance Notes

All testing was done on a **Tesla T4** GPU using Google Colab / Kaggle's free tier.

Results will vary depending on your GPU, the specific graph topology, and the dimensions involved. The speedup from operator fusion is highly graph-dependent — a deep sequential chain of MatMul→Add→ReLU layers will see the most benefit since each fusion eliminates two kernel launches and their associated global memory traffic. Graphs with operations that can't be fused (e.g., element-wise adds at merge points between branches) will see more modest gains.

In practice, expect anywhere from **~1.1x to ~2x+ speedup** from the fusion pass depending on the network architecture. The multi-stream parallelism adds further gains on graphs with independent branches.

> No cherry-picked benchmark numbers here — the results genuinely depend on what you throw at it. Try it on your own models and see.

## Currently Supported Operations

| Operation | Status |
|---|---|
| MatMul (Gemm) | ✅ Full support with arbitrary dimensions |
| Element-wise Add | ✅ With broadcasting (bias vectors) |
| ReLU | ✅ Vectorized `float4` |
| Fused MatMul+ReLU | ✅ |
| Fused Add+ReLU | ✅ |
| Fused MatMul+Add | ✅ |
| Fused MatMul+Add+ReLU | ✅ |
| Conv2D, Softmax, LayerNorm, etc. | ❌ Not yet |


## Known Issues & Limitations
- **ONNX import is limited** — The exporter (`onnx_exporter.py`) only handles `Gemm`, `MatMul`, `Add`, and `Relu` ops. Anything else (BatchNorm, Reshape, Flatten, Transpose, Concat, etc.) is silently skipped, which will produce a broken or incomplete graph. Stick to simple MLP/linear architectures. The `Gemm` handling also assumes specific `transB` conventions — non-standard ONNX attribute combinations may produce incorrectly transposed weights.
- **No `cudaFree` in generated code** — The `--codegen` output emits `cudaMalloc` for all buffers but never emits any `cudaFree` calls. The runtime compiler relies on the `Graph` destructor to clean up, but standalone generated programs will leak all GPU memory until process exit.
- **No CUDA error checking** — `cudaMalloc`, `cudaMemcpy`, stream/event creation, and kernel launches all return error codes that are currently never checked. A failed allocation (e.g., running out of VRAM on large graphs) will silently produce garbage output with no warning.
- **Shapes are 2D only** — Everything internally assumes `shape` is exactly `{rows, cols}`. There's no support for batched 3D tensors, 1D vectors, or higher-dimensional inputs without manually reshaping them to 2D first.
- **`--standalone` codegen is WIP** — The `--standalone` flag (which injects kernel source code into the generated `.cu` file) currently has bugs and may produce files that don't compile. Use `--codegen` without `--standalone` for now.
- **Fusion pass is single-use** — The `fusionPass()` method uses node-ID-indexed vectors sized to the current graph. Calling it a second time after new fused nodes have been created risks out-of-bounds access since fused nodes get IDs beyond the original array bounds.


## The Journey

This project started as a way to really understand what happens under the hood of ML compilers — not just the theory, but the actual GPU mechanics. Writing tiled matmul kernels from scratch, debugging shared memory bank conflicts, figuring out CUDA stream capture semantics for benchmarking, getting operator fusion to correctly rewire graph edges... every piece of it was a learning experience.

It's not a production compiler and it doesn't try to be one. But building it from the ground up — no cuBLAS, no cuDNN, no framework magic — was one heck of a journey, and genuinely one of the most fun projects I've worked on.

