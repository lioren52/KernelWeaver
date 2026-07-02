#pragma once
#include <cuda_runtime.h>

void matMul(float *A, float *B, float *C, int row_A, int N, int col_B,
            cudaStream_t stream = 0);

void matAdd(float *A, float *B, float *C, int height, int width, int b_rows,
            int b_cols, cudaStream_t stream = 0);

void matReLU(float *A, float *C, int height, int width,
             cudaStream_t stream = 0);

void matMulReLU(float *A, float *B, float *C, int row_A, int N, int col_B,
                cudaStream_t stream = 0);

void matAddReLU(float *A, float *B, float *C, int height, int width, int b_rows,
                int b_cols, cudaStream_t stream = 0);

void matMulAdd(float *A, float *B, float *Bias, float *C, int row_A, int N,
               int col_B, int b_rows, int b_cols, cudaStream_t stream = 0);

void matMulAddReLU(float *A, float *B, float *Bias, float *C, int row_A, int N,
                   int col_B, int b_rows, int b_cols, cudaStream_t stream = 0);