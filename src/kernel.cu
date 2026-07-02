#define BM 64
#define BN 64
#define BK 32
#define WM 16
#define WN 32
#define TM 4
#define TN 4

// Padding constant to eliminate shared-memory bank conflicts.
// Adding 1 float to each row of As prevents stride-32 conflicts
// when threads in the same warp read consecutive rows at the same column.
#define AS_PAD 1

// ─────────────────────────────────────────────────────────────────────────────
// Fast warp-tiled + vectorized matmul.
// Handles arbitrary M, K, N (no multiple-of-tile requirement).
// Safe for column-vector outputs (N=1) and non-multiple K.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matrixMulFast(float *A, float *B, float *C, int M, int K,
                              int N) {
  __shared__ float As[BM][BK + AS_PAD];
  __shared__ float Bs[BK][BN];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;

  int warp_row = warp_id / 2;
  int warp_col = warp_id % 2;

  int lane_row = lane_id / 8;
  int lane_col = lane_id % 8;

  int out_row = blockIdx.y * BM + warp_row * WM + lane_row * TM;
  int out_col = blockIdx.x * BN + warp_col * WN + lane_col * TN;

  float reg_C[TM][TN] = {0.0f};
  float reg_A[TM];
  float reg_B[TN];

  int total_threads = blockDim.x * blockDim.y; // 256

  // Alignment flags: float4 loads require 16-byte aligned addresses.
  // If row stride (K for A, N for B) is not a multiple of 4, rows will be misaligned.
  bool a_aligned = (K % 4 == 0);
  bool b_aligned = (N % 4 == 0);

  int num_full_phases = K / BK; // phases where the whole BK tile fits
  int remainder = K % BK;       // leftover columns

  // ── Full BK-wide phases ───────────────────────────────────────────────────
  for (int ph = 0; ph < num_full_phases; ph++) {

    // Load As: each thread loads float4 chunks — 2 iters for 256 threads
    for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads;
      int row = idx / (BK / 4);
      int col4 = idx % (BK / 4);

      int global_row = blockIdx.y * BM + row;
      int global_col = ph * BK + col4 * 4;

      if (a_aligned && global_row < M && global_col + 3 < K) {
        // fully in-bounds: vectorized load
        float4 tmp =
            reinterpret_cast<float4 *>(&A[global_row * K + global_col])[0];
        As[row][col4 * 4 + 0] = tmp.x;
        As[row][col4 * 4 + 1] = tmp.y;
        As[row][col4 * 4 + 2] = tmp.z;
        As[row][col4 * 4 + 3] = tmp.w;
      } else {
        // boundary: scalar fallback
        for (int j = 0; j < 4; j++) {
          int gc = global_col + j;
          As[row][col4 * 4 + j] =
              (global_row < M && gc < K) ? A[global_row * K + gc] : 0.0f;
        }
      }
    }

    // Load Bs: same approach
    for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads;
      int row = idx / (BN / 4);
      int col4 = idx % (BN / 4);

      int global_row = ph * BK + row;
      int global_col = blockIdx.x * BN + col4 * 4;

      if (b_aligned && global_row < K && global_col + 3 < N) {
        float4 tmp =
            reinterpret_cast<float4 *>(&B[global_row * N + global_col])[0];
        Bs[row][col4 * 4 + 0] = tmp.x;
        Bs[row][col4 * 4 + 1] = tmp.y;
        Bs[row][col4 * 4 + 2] = tmp.z;
        Bs[row][col4 * 4 + 3] = tmp.w;
      } else {
        for (int j = 0; j < 4; j++) {
          int gc = global_col + j;
          Bs[row][col4 * 4 + j] =
              (global_row < K && gc < N) ? B[global_row * N + gc] : 0.0f;
        }
      }
    }

    __syncthreads();

    for (int k = 0; k < BK; k++) {
      for (int m = 0; m < TM; m++)
        reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];
      for (int n = 0; n < TN; n++)
        reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];
      for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
          reg_C[m][n] += reg_A[m] * reg_B[n];
    }

    __syncthreads();
  }

  // ── Remainder phase (K % BK leftover columns) ────────────────────────────
  if (remainder > 0) {
    int ph = num_full_phases;

    // Scalar load for remainder — not worth vectorizing partial chunks
    for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads;
      int row = idx / (BK / 4);
      int col4 = idx % (BK / 4);
      int global_row = blockIdx.y * BM + row;
      for (int j = 0; j < 4; j++) {
        int local_col = col4 * 4 + j;
        int global_col = ph * BK + local_col;
        As[row][local_col] = (global_row < M && local_col < remainder)
                                 ? A[global_row * K + global_col]
                                 : 0.0f;
      }
    }

    for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads;
      int row = idx / (BN / 4);
      int col4 = idx % (BN / 4);
      int global_row = ph * BK + row;
      for (int j = 0; j < 4; j++) {
        int local_col = col4 * 4 + j;
        int global_col = blockIdx.x * BN + local_col;
        Bs[row][local_col] = (row < remainder && global_col < N)
                                 ? B[global_row * N + global_col]
                                 : 0.0f;
      }
    }

    __syncthreads();

    for (int k = 0; k < remainder; k++) {
      for (int m = 0; m < TM; m++)
        reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];
      for (int n = 0; n < TN; n++)
        reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];
      for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
          reg_C[m][n] += reg_A[m] * reg_B[n];
    }

    __syncthreads();
  }

  // ── Write output ─────────────────────────────────────────────────────────
  for (int m = 0; m < TM; m++)
    for (int n = 0; n < TN; n++)
      if (out_row + m < M && out_col + n < N)
        C[(out_row + m) * N + out_col + n] = reg_C[m][n];
}

// ─────────────────────────────────────────────────────────────────────────────
// Vectorized element-wise Add kernel (float4).
// Falls back to scalar for the tail elements when total is not a multiple of 4.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matrixAdd(float *A, float *B, float *C, int height, int width, int b_rows, int b_cols) {
  int col4 = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= height || col4 * 4 >= width) return;

  int b_stride_y = (b_rows > 1) ? width : 0;
  int b_stride_x = (b_cols > 1) ? 1 : 0;

  int idx = row * width + col4 * 4;
  int b_idx = row * b_stride_y + col4 * 4 * b_stride_x;

  if (col4 * 4 + 3 < width) {
    float4 a = reinterpret_cast<float4 *>(&A[idx])[0];
    float4 b;
    if (b_cols > 1) {
      b = reinterpret_cast<float4 *>(&B[b_idx])[0];
    } else {
      float val = B[row * b_stride_y];
      b.x = val; b.y = val; b.z = val; b.w = val;
    }
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    reinterpret_cast<float4 *>(&C[idx])[0] = c;
  } else {
    for (int j = 0; j < 4 && col4 * 4 + j < width; j++) {
      int b_j = row * b_stride_y + (col4 * 4 + j) * b_stride_x;
      C[idx + j] = A[idx + j] + B[b_j];
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vectorized element-wise ReLU kernel (float4).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matrixReLU(const float *A, float *C, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int idx4 = idx * 4;
  if (idx4 + 3 < total) {
    float4 a = reinterpret_cast<const float4 *>(A)[idx];
    float4 c;
    c.x = fmaxf(0.0f, a.x);
    c.y = fmaxf(0.0f, a.y);
    c.z = fmaxf(0.0f, a.z);
    c.w = fmaxf(0.0f, a.w);
    reinterpret_cast<float4 *>(C)[idx] = c;
  } else {
    for (int j = 0; j < 4 && idx4 + j < total; j++) {
      C[idx4 + j] = fmaxf(0.0f, A[idx4 + j]);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host wrappers — all accept an optional cudaStream_t for stream-parallel
// dispatch
// ─────────────────────────────────────────────────────────────────────────────
void matMul(float *A, float *B, float *C, int row_A, int N, int col_B,
            cudaStream_t stream) {
  dim3 threadPerBlock(16, 16); // 256 threads = 8 warps, matches tile layout
  dim3 blocks((col_B + BN - 1) / BN, (row_A + BM - 1) / BM);
  matrixMulFast<<<blocks, threadPerBlock, 0, stream>>>(A, B, C, row_A, N,
                                                       col_B);
}

void matAdd(float *A, float *B, float *C, int height, int width, int b_rows, int b_cols,
            cudaStream_t stream) {
  dim3 numThreads(32, 8); // 256 threads
  dim3 numBlocks(((width + 3) / 4 + numThreads.x - 1) / numThreads.x, 
                 (height + numThreads.y - 1) / numThreads.y);

  matrixAdd<<<numBlocks, numThreads, 0, stream>>>(A, B, C, height, width, b_rows, b_cols);
}

void matReLU(float *A, float *C, int height, int width, cudaStream_t stream) {
  int total = height * width;
  int numThreads = 256;
  int numBlocks = (total / 4 + numThreads - 1) / numThreads;

  matrixReLU<<<numBlocks, numThreads, 0, stream>>>(A, C, total);
}

// =====================================================================
// 1. FUSED: Matrix Multiplication + ReLU (MR)
// =====================================================================
__global__ void matrixMulFastReLU(float *A, float *B, float *C, int M, int K,
                                  int N) {
  __shared__ float As[BM][BK + AS_PAD];
  __shared__ float Bs[BK][BN];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int warp_row = warp_id / 2;
  int warp_col = warp_id % 2;
  int lane_row = lane_id / 8;
  int lane_col = lane_id % 8;
  int out_row = blockIdx.y * BM + warp_row * WM + lane_row * TM;
  int out_col = blockIdx.x * BN + warp_col * WN + lane_col * TN;

  float reg_C[TM][TN] = {0.0f};
  float reg_A[TM], reg_B[TN];
  int total_threads = blockDim.x * blockDim.y;
  bool a_aligned = (K % 4 == 0);
  bool b_aligned = (N % 4 == 0);
  int num_full_phases = K / BK;
  int remainder = K % BK;

  for (int ph = 0; ph < num_full_phases; ph++) {
    for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BK / 4),
          col4 = idx % (BK / 4);
      int gr = blockIdx.y * BM + row, gc = ph * BK + col4 * 4;
      if (a_aligned && gr < M && gc + 3 < K) {
        float4 tmp = reinterpret_cast<float4 *>(&A[gr * K + gc])[0];
        As[row][col4 * 4 + 0] = tmp.x;
        As[row][col4 * 4 + 1] = tmp.y;
        As[row][col4 * 4 + 2] = tmp.z;
        As[row][col4 * 4 + 3] = tmp.w;
      } else {
        for (int j = 0; j < 4; j++) {
          int c = gc + j;
          As[row][col4 * 4 + j] = (gr < M && c < K) ? A[gr * K + c] : 0.0f;
        }
      }
    }
    for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BN / 4),
          col4 = idx % (BN / 4);
      int gr = ph * BK + row, gc = blockIdx.x * BN + col4 * 4;
      if (b_aligned && gr < K && gc + 3 < N) {
        float4 tmp = reinterpret_cast<float4 *>(&B[gr * N + gc])[0];
        Bs[row][col4 * 4 + 0] = tmp.x;
        Bs[row][col4 * 4 + 1] = tmp.y;
        Bs[row][col4 * 4 + 2] = tmp.z;
        Bs[row][col4 * 4 + 3] = tmp.w;
      } else {
        for (int j = 0; j < 4; j++) {
          int c = gc + j;
          Bs[row][col4 * 4 + j] = (gr < K && c < N) ? B[gr * N + c] : 0.0f;
        }
      }
    }
    __syncthreads();
    for (int k = 0; k < BK; k++) {
      for (int m = 0; m < TM; m++)
        reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];
      for (int n = 0; n < TN; n++)
        reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];
      for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
          reg_C[m][n] += reg_A[m] * reg_B[n];
    }
    __syncthreads();
  }
  if (remainder > 0) {
    int ph = num_full_phases;
    for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BK / 4),
          col4 = idx % (BK / 4), gr = blockIdx.y * BM + row;
      for (int j = 0; j < 4; j++) {
        int lc = col4 * 4 + j, gc = ph * BK + lc;
        As[row][lc] = (gr < M && lc < remainder) ? A[gr * K + gc] : 0.0f;
      }
    }
    for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BN / 4),
          col4 = idx % (BN / 4), gr = ph * BK + row;
      for (int j = 0; j < 4; j++) {
        int lc = col4 * 4 + j, gc = blockIdx.x * BN + lc;
        Bs[row][lc] = (row < remainder && gc < N) ? B[gr * N + gc] : 0.0f;
      }
    }
    __syncthreads();
    for (int k = 0; k < remainder; k++) {
      for (int m = 0; m < TM; m++)
        reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];
      for (int n = 0; n < TN; n++)
        reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];
      for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
          reg_C[m][n] += reg_A[m] * reg_B[n];
    }
    __syncthreads();
  }
  for (int m = 0; m < TM; m++)
    for (int n = 0; n < TN; n++)
      if (out_row + m < M && out_col + n < N)
        C[(out_row + m) * N + out_col + n] = fmaxf(0.0f, reg_C[m][n]);
}

void matMulReLU(float *A, float *B, float *C, int row_A, int N, int col_B,
                cudaStream_t stream) {
  dim3 threadPerBlock(16, 16);
  dim3 blocks((col_B + BN - 1) / BN, (row_A + BM - 1) / BM);
  matrixMulFastReLU<<<blocks, threadPerBlock, 0, stream>>>(A, B, C, row_A, N,
                                                           col_B);
}

// =====================================================================
// 2. FUSED: Matrix Addition + ReLU (AR) — vectorized with float4
// =====================================================================
__global__ void matrixAddReLU(float *A, float *B, float *C, int height, int width, int b_rows, int b_cols) {
  int col4 = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= height || col4 * 4 >= width) return;

  int b_stride_y = (b_rows > 1) ? width : 0;
  int b_stride_x = (b_cols > 1) ? 1 : 0;

  int idx = row * width + col4 * 4;
  int b_idx = row * b_stride_y + col4 * 4 * b_stride_x;

  if (col4 * 4 + 3 < width) {
    float4 a = reinterpret_cast<float4 *>(&A[idx])[0];
    float4 b;
    if (b_cols > 1) {
      b = reinterpret_cast<float4 *>(&B[b_idx])[0];
    } else {
      float val = B[row * b_stride_y];
      b.x = val; b.y = val; b.z = val; b.w = val;
    }
    float4 c;
    c.x = fmaxf(0.0f, a.x + b.x);
    c.y = fmaxf(0.0f, a.y + b.y);
    c.z = fmaxf(0.0f, a.z + b.z);
    c.w = fmaxf(0.0f, a.w + b.w);
    reinterpret_cast<float4 *>(&C[idx])[0] = c;
  } else {
    for (int j = 0; j < 4 && col4 * 4 + j < width; j++) {
      int b_j = row * b_stride_y + (col4 * 4 + j) * b_stride_x;
      float val = A[idx + j] + B[b_j];
      C[idx + j] = fmaxf(0.0f, val);
    }
  }
}

void matAddReLU(float *A, float *B, float *C, int height, int width, int b_rows, int b_cols,
                cudaStream_t stream) {
  dim3 numThreads(32, 8); // 256 threads
  dim3 numBlocks(((width + 3) / 4 + numThreads.x - 1) / numThreads.x, 
                 (height + numThreads.y - 1) / numThreads.y);

  matrixAddReLU<<<numBlocks, numThreads, 0, stream>>>(A, B, C, height, width, b_rows, b_cols);
}

// =====================================================================
// 3. FUSED: Matrix Multiplication + Addition (MA)
// =====================================================================
__global__ void matrixMulFastAdd(float *A, float *B, float *Bias, float *C,
                                 int M, int K, int N, int b_rows, int b_cols) {
  __shared__ float As[BM][BK + AS_PAD];
  __shared__ float Bs[BK][BN];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int warp_row = warp_id / 2;
  int warp_col = warp_id % 2;
  int lane_row = lane_id / 8;
  int lane_col = lane_id % 8;
  int out_row = blockIdx.y * BM + warp_row * WM + lane_row * TM;
  int out_col = blockIdx.x * BN + warp_col * WN + lane_col * TN;

  float reg_C[TM][TN] = {0.0f};
  float reg_A[TM], reg_B[TN];
  int total_threads = blockDim.x * blockDim.y;
  bool a_aligned = (K % 4 == 0);
  bool b_aligned = (N % 4 == 0);
  int num_full_phases = K / BK;
  int remainder = K % BK;

  for (int ph = 0; ph < num_full_phases; ph++) {
    for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BK / 4),
          col4 = idx % (BK / 4);
      int gr = blockIdx.y * BM + row, gc = ph * BK + col4 * 4;
      if (a_aligned && gr < M && gc + 3 < K) {
        float4 tmp = reinterpret_cast<float4 *>(&A[gr * K + gc])[0];
        As[row][col4 * 4 + 0] = tmp.x;
        As[row][col4 * 4 + 1] = tmp.y;
        As[row][col4 * 4 + 2] = tmp.z;
        As[row][col4 * 4 + 3] = tmp.w;
      } else {
        for (int j = 0; j < 4; j++) {
          int c = gc + j;
          As[row][col4 * 4 + j] = (gr < M && c < K) ? A[gr * K + c] : 0.0f;
        }
      }
    }
    for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BN / 4),
          col4 = idx % (BN / 4);
      int gr = ph * BK + row, gc = blockIdx.x * BN + col4 * 4;
      if (b_aligned && gr < K && gc + 3 < N) {
        float4 tmp = reinterpret_cast<float4 *>(&B[gr * N + gc])[0];
        Bs[row][col4 * 4 + 0] = tmp.x;
        Bs[row][col4 * 4 + 1] = tmp.y;
        Bs[row][col4 * 4 + 2] = tmp.z;
        Bs[row][col4 * 4 + 3] = tmp.w;
      } else {
        for (int j = 0; j < 4; j++) {
          int c = gc + j;
          Bs[row][col4 * 4 + j] = (gr < K && c < N) ? B[gr * N + c] : 0.0f;
        }
      }
    }
    __syncthreads();
    for (int k = 0; k < BK; k++) {
      for (int m = 0; m < TM; m++)
        reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];
      for (int n = 0; n < TN; n++)
        reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];
      for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
          reg_C[m][n] += reg_A[m] * reg_B[n];
    }
    __syncthreads();
  }
  if (remainder > 0) {
    int ph = num_full_phases;
    for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BK / 4),
          col4 = idx % (BK / 4), gr = blockIdx.y * BM + row;
      for (int j = 0; j < 4; j++) {
        int lc = col4 * 4 + j, gc = ph * BK + lc;
        As[row][lc] = (gr < M && lc < remainder) ? A[gr * K + gc] : 0.0f;
      }
    }
    for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BN / 4),
          col4 = idx % (BN / 4), gr = ph * BK + row;
      for (int j = 0; j < 4; j++) {
        int lc = col4 * 4 + j, gc = blockIdx.x * BN + lc;
        Bs[row][lc] = (row < remainder && gc < N) ? B[gr * N + gc] : 0.0f;
      }
    }
    __syncthreads();
    for (int k = 0; k < remainder; k++) {
      for (int m = 0; m < TM; m++)
        reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];
      for (int n = 0; n < TN; n++)
        reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];
      for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
          reg_C[m][n] += reg_A[m] * reg_B[n];
    }
    __syncthreads();
  }
  // Epilogue: add bias
  int b_stride_y = (b_rows > 1) ? N : 0;
  int b_stride_x = (b_cols > 1) ? 1 : 0;

  for (int m = 0; m < TM; m++)
    for (int n = 0; n < TN; n++)
      if (out_row + m < M && out_col + n < N) {
        C[(out_row + m) * N + out_col + n] =
            reg_C[m][n] + Bias[(out_row + m) * b_stride_y + (out_col + n) * b_stride_x];
      }
}

void matMulAdd(float *A, float *B, float *Bias, float *C, int row_A, int N,
               int col_B, int b_rows, int b_cols, cudaStream_t stream) {
  dim3 threadPerBlock(16, 16);
  dim3 blocks((col_B + BN - 1) / BN, (row_A + BM - 1) / BM);
  matrixMulFastAdd<<<blocks, threadPerBlock, 0, stream>>>(A, B, Bias, C, row_A,
                                                          N, col_B, b_rows, b_cols);
}

// =====================================================================
// 4. FUSED: Matrix Multiplication + Addition + ReLU (MAR)
// =====================================================================
__global__ void matrixMulFastAddReLU(float *A, float *B, float *Bias, float *C,
                                     int M, int K, int N, int b_rows, int b_cols) {
  __shared__ float As[BM][BK + AS_PAD];
  __shared__ float Bs[BK][BN];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane_id = tid % 32;
  int warp_row = warp_id / 2;
  int warp_col = warp_id % 2;
  int lane_row = lane_id / 8;
  int lane_col = lane_id % 8;
  int out_row = blockIdx.y * BM + warp_row * WM + lane_row * TM;
  int out_col = blockIdx.x * BN + warp_col * WN + lane_col * TN;

  float reg_C[TM][TN] = {0.0f};
  float reg_A[TM], reg_B[TN];
  int total_threads = blockDim.x * blockDim.y;
  bool a_aligned = (K % 4 == 0);
  bool b_aligned = (N % 4 == 0);
  int num_full_phases = K / BK;
  int remainder = K % BK;

  for (int ph = 0; ph < num_full_phases; ph++) {
    for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BK / 4),
          col4 = idx % (BK / 4);
      int gr = blockIdx.y * BM + row, gc = ph * BK + col4 * 4;
      if (a_aligned && gr < M && gc + 3 < K) {
        float4 tmp = reinterpret_cast<float4 *>(&A[gr * K + gc])[0];
        As[row][col4 * 4 + 0] = tmp.x;
        As[row][col4 * 4 + 1] = tmp.y;
        As[row][col4 * 4 + 2] = tmp.z;
        As[row][col4 * 4 + 3] = tmp.w;
      } else {
        for (int j = 0; j < 4; j++) {
          int c = gc + j;
          As[row][col4 * 4 + j] = (gr < M && c < K) ? A[gr * K + c] : 0.0f;
        }
      }
    }
    for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BN / 4),
          col4 = idx % (BN / 4);
      int gr = ph * BK + row, gc = blockIdx.x * BN + col4 * 4;
      if (b_aligned && gr < K && gc + 3 < N) {
        float4 tmp = reinterpret_cast<float4 *>(&B[gr * N + gc])[0];
        Bs[row][col4 * 4 + 0] = tmp.x;
        Bs[row][col4 * 4 + 1] = tmp.y;
        Bs[row][col4 * 4 + 2] = tmp.z;
        Bs[row][col4 * 4 + 3] = tmp.w;
      } else {
        for (int j = 0; j < 4; j++) {
          int c = gc + j;
          Bs[row][col4 * 4 + j] = (gr < K && c < N) ? B[gr * N + c] : 0.0f;
        }
      }
    }
    __syncthreads();
    for (int k = 0; k < BK; k++) {
      for (int m = 0; m < TM; m++)
        reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];
      for (int n = 0; n < TN; n++)
        reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];
      for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
          reg_C[m][n] += reg_A[m] * reg_B[n];
    }
    __syncthreads();
  }
  if (remainder > 0) {
    int ph = num_full_phases;
    for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BK / 4),
          col4 = idx % (BK / 4), gr = blockIdx.y * BM + row;
      for (int j = 0; j < 4; j++) {
        int lc = col4 * 4 + j, gc = ph * BK + lc;
        As[row][lc] = (gr < M && lc < remainder) ? A[gr * K + gc] : 0.0f;
      }
    }
    for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
      int idx = tid + i * total_threads, row = idx / (BN / 4),
          col4 = idx % (BN / 4), gr = ph * BK + row;
      for (int j = 0; j < 4; j++) {
        int lc = col4 * 4 + j, gc = blockIdx.x * BN + lc;
        Bs[row][lc] = (row < remainder && gc < N) ? B[gr * N + gc] : 0.0f;
      }
    }
    __syncthreads();
    for (int k = 0; k < remainder; k++) {
      for (int m = 0; m < TM; m++)
        reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];
      for (int n = 0; n < TN; n++)
        reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];
      for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
          reg_C[m][n] += reg_A[m] * reg_B[n];
    }
    __syncthreads();
  }
  // Epilogue: add bias and apply ReLU
  int b_stride_y = (b_rows > 1) ? N : 0;
  int b_stride_x = (b_cols > 1) ? 1 : 0;

  for (int m = 0; m < TM; m++)
    for (int n = 0; n < TN; n++)
      if (out_row + m < M && out_col + n < N) {
        float val = reg_C[m][n] + Bias[(out_row + m) * b_stride_y + (out_col + n) * b_stride_x];
        C[(out_row + m) * N + out_col + n] = fmaxf(0.0f, val);
      }
}

void matMulAddReLU(float *A, float *B, float *Bias, float *C, int row_A, int N,
                   int col_B, int b_rows, int b_cols, cudaStream_t stream) {
  dim3 threadPerBlock(16, 16);
  dim3 blocks((col_B + BN - 1) / BN, (row_A + BM - 1) / BM);
  matrixMulFastAddReLU<<<blocks, threadPerBlock, 0, stream>>>(A, B, Bias, C,
                                                              row_A, N, col_B, b_rows, b_cols);
}