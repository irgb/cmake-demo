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

void addVector(int *A, int *B, int *C, int n) {
    for(int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}
