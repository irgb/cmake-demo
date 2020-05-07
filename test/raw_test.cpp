#include <iostream>
#include <stdlib.h>

#include "add.h"

using namespace std;

int main(int argc, char *argv[]) {
    if(argc < 3) {
        cout << "Usage: ./raw_test <number1> <number2>" << endl;
        return 1;
    }
    int i1 = atoi(argv[1]);
    int i2 = atoi(argv[2]);
    cout << add(i1, i2) << endl;
    return 0;
}
