// vec_add.cu
// Purpose: Add two float arrays element-wise on the GPU.
// This is the minimal CUDA program — everything here appears in every CUDA kernel.

#include <cuda_runtime.h>
#include <stdio.h>

// __global__ = "launch this on the GPU, call it from the CPU"
__global__ void vecAdd(float* a, float* b, float* c, int n) {

    // Each thread computes its global index in the 1D grid.
    // blockDim.x  = threads per block (set at launch time, e.g. 256)
    // blockIdx.x  = which block this thread belongs to (0, 1, 2, ...)
    // threadIdx.x = this thread's position within its block (0..255)
    //
    // Example with 256 threads/block:
    //   Thread 0 of block 0:  i = 0 * 256 + 0 = 0
    //   Thread 1 of block 0:  i = 0 * 256 + 1 = 1
    //   Thread 0 of block 1:  i = 1 * 256 + 0 = 256
    //   Thread 0 of block 4:  i = 4 * 256 + 0 = 1024
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Guard: our grid might have more threads than elements.
    // If n=1000001 and blockDim=256, we launch ceil(1000001/256)*256 threads
    // but only 1000001 of them should do work.
    if (i < n) {
        c[i] = a[i] + b[i];  // The actual work: one thread, one addition
    }
}

int main() {
    int n = 1 << 20;                  // 2^20 = 1,048,576 elements
    size_t size = n * sizeof(float);  // bytes: 4MB per array

    // ---- Host (CPU) memory ----
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    for (int i = 0; i < n; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // ---- Device (GPU) memory ----
    // cudaMalloc(&ptr, bytes) — allocates in HBM, returns pointer in GPU address space
    // You cannot dereference d_a from CPU code. It's an opaque GPU address.
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Transfer: CPU -> GPU (host -> device)
    // cudaMemcpy(dst, src, bytes, direction)
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // ---- Kernel Launch ----
    int threadsPerBlock = 256;   // = 8 warps per block
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    // For n=1M: blocksPerGrid = (1048576 + 255) / 256 = 4096 blocks
    // Total threads launched: 4096 * 256 = 1,048,576 exactly

    // <<<blocks, threads>>> is the CUDA launch syntax
    // The CPU continues immediately after this — the call is asynchronous
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Block CPU until all GPU work is complete
    cudaDeviceSynchronize();

    // Transfer: GPU -> CPU (device -> host)
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify: every element should be 1.0 + 2.0 = 3.0
    printf("h_c[0]    = %.1f (expected 3.0)\n", h_c[0]);
    printf("h_c[%d] = %.1f (expected 3.0)\n", n-1, h_c[n-1]);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    else
        printf("No CUDA errors.\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
