#include <stdio.h>
#include <iostream>

#include "print_hello.h"

using std::cout;
using std::endl;

__global__ void hello_cuda_kernel() {
    printf("Hello World from GPU!\n");
}

void printHello() {
    cout << "CUDA is enabled. hello from GPU: " << endl;
    hello_cuda_kernel<<<2,3>>>();
    cudaDeviceReset();
}
