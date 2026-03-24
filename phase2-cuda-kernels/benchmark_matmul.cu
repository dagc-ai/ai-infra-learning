// benchmark_matmul.cu
// Compile: nvcc -O3 -lcublas benchmark_matmul.cu -o benchmark_matmul

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

// ---- Naive kernel ----
// One thread computes one output element.
// Every float access goes directly to HBM.
__global__ void naiveMatMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ---- Tiled kernel ----
// Threads cooperatively load tiles into shared memory (SRAM).
// Each HBM value is loaded once and reused TILE_SIZE times.
__global__ void tiledMatMul(float* A, float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative load: each thread loads one element into shared memory
        sA[threadIdx.y][threadIdx.x] = (row < N && t*TILE_SIZE+threadIdx.x < N)
            ? A[row * N + t * TILE_SIZE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (col < N && t*TILE_SIZE+threadIdx.y < N)
            ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;

        // Wait for ALL threads to finish loading before anyone computes
        __syncthreads();

        // Compute from SRAM (~5 cycles) instead of HBM (~500 cycles)
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        // Wait for ALL threads to finish computing before anyone loads next tile
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}

// ---- GPU-side timer using CUDA events ----
// More accurate than CPU wall clock for GPU kernels.
// cudaEventRecord() plants a timestamp in the GPU stream.
// cudaEventElapsedTime() returns GPU time between two timestamps in ms.
typedef void (*KernelFn)(float*, float*, float*, int, int);

float timeKernel(KernelFn launch, float* A, float* B, float* C, int N, int repeats) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up: first launch has JIT overhead, don't count it
    launch(A, B, C, N, 1);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    launch(A, B, C, N, repeats);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeats;
}

void runNaive(float* A, float* B, float* C, int N, int reps) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
    for (int i = 0; i < reps; i++)
        naiveMatMul<<<grid, block>>>(A, B, C, N);
}

void runTiled(float* A, float* B, float* C, int N, int reps) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
    for (int i = 0; i < reps; i++)
        tiledMatMul<<<grid, block>>>(A, B, C, N);
}

int main() {
    int sizes[] = {1024, 2048, 4096};
    int num_sizes = 3;
    int repeats = 20;

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    printf("\n%-6s  %-10s %-12s  %-10s %-12s  %-10s %-12s\n",
           "N", "Naive(ms)", "Naive(TFLOPS)", "Tiled(ms)",
           "Tiled(TFLOPS)", "cuBLAS(ms)", "cuBLAS(TFLOPS)");
    printf("------------------------------------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        size_t bytes = (size_t)N * N * sizeof(float);
        double flops = 2.0 * N * N * N;

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        // Initialize with random data on host, copy to device
        float *h_A = (float*)malloc(bytes);
        for (int i = 0; i < N*N; i++) h_A[i] = (float)rand() / RAND_MAX;
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_A, bytes, cudaMemcpyHostToDevice);
        free(h_A);

        // Naive (skip N=4096 — takes several minutes)
        float naive_ms = -1.0f;
        if (N <= 2048)
            naive_ms = timeKernel(runNaive, d_A, d_B, d_C, N, repeats);

        // Tiled
        float tiled_ms = timeKernel(runTiled, d_A, d_B, d_C, N, repeats);

        // cuBLAS
        // Note: cuBLAS expects column-major. Passing B,A instead of A,B
        // is the standard trick to get correct row-major results.
        cudaEvent_t c_start, c_stop;
        cudaEventCreate(&c_start); cudaEventCreate(&c_stop);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        cudaDeviceSynchronize();
        cudaEventRecord(c_start);
        for (int r = 0; r < repeats; r++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        cudaEventRecord(c_stop);
        cudaEventSynchronize(c_stop);
        float cublas_ms = 0;
        cudaEventElapsedTime(&cublas_ms, c_start, c_stop);
        cublas_ms /= repeats;
        cudaEventDestroy(c_start); cudaEventDestroy(c_stop);

        double tiled_tflops  = flops / tiled_ms  / 1e9;
        double cublas_tflops = flops / cublas_ms / 1e9;

        if (naive_ms > 0) {
            double naive_tflops = flops / naive_ms / 1e9;
            printf("%-6d  %-10.2f %-12.2f  %-10.2f %-12.2f  %-10.2f %-12.2f\n",
                   N, naive_ms, naive_tflops,
                   tiled_ms, tiled_tflops,
                   cublas_ms, cublas_tflops);
        } else {
            printf("%-6d  %-10s %-12s  %-10.2f %-12.2f  %-10.2f %-12.2f\n",
                   N, "skipped", "skipped",
                   tiled_ms, tiled_tflops,
                   cublas_ms, cublas_tflops);
        }

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    cublasDestroy(handle);
    return 0;
}
