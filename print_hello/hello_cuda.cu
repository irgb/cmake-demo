#include <stdio.h>
#include <iostream>

#include "print_hello.h"

using std::cout;
using std::endl;

__global__ void hello_cuda_kernel() {
    printf("Hello World from GPU!\n");
}

// __host__ is default identifier, can be omitted.
__host__ void printHello() {
    cout << "CUDA is enabled. hello from GPU: " << endl;
    hello_cuda_kernel<<<2,3>>>();
    cout << "hello_cuda_kernel calling end, which is asynchronous." << endl;
    cudaDeviceReset();
}

__global__ void add_vector_kernel(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void addVector(int *a, int *b, int *c, int n) {
    cout << "add vector on GPU:" << endl;
    int *device_a, *device_b, *device_c;
    cudaMalloc((void **)(&device_a), sizeof(int) * n);
    cudaMalloc((void **)(&device_b), sizeof(int) * n);
    cudaMalloc((void **)(&device_c), sizeof(int) * n);

    cudaMemcpy(device_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, sizeof(int) * n, cudaMemcpyHostToDevice);

    add_vector_kernel<<<2, 3>>>(device_a, device_b, device_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, device_c, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}
