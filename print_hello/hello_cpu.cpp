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
