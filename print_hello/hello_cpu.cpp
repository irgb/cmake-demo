#include <stdio.h>
#include <iostream>

#include "print_hello.h"

using std::cout;
using std::endl;

void hello_cpu() {
    printf("Hello World from CPU!\n");
}

void printHello() {
    cout << "CUDA is disabled(Execute cmake with -DWITH_CUDA=ON to enable it.). hello from CPU: " << endl;
    hello_cpu();
}

void addVector(int *A, int *B, int *C, int n) {
    cout << "add vector on CPU." << endl;
    for(int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}
