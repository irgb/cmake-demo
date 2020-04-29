#include <iostream>

#include "gflags/gflags.h"

#include "config.h"
#include "add.h"
#include "sub.h"
#include "reverse.h"
#include "print_hello.h"

using namespace std;

DEFINE_bool(cpp_version, false, "print cpp version");

void testAddVector(int& n, int *a, int *b, int *c);

int main(int argc, char* argv[])
{
    // parse gflags
    gflags::SetVersionString(PROJECT_VERSION);
    gflags::SetUsageMessage("Usage:");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    #ifdef HELLO_TOGGLE
    cout << "Hello, Cmake!" << endl;
    #endif

    if(FLAGS_cpp_version) {
        cout << "__cplusplus: " << __cplusplus << endl;
    }

    cout << "------------" << endl;
    int i1 = 0, i2 = 0;
    cout << "input 1th number: ";
    cin >> i1;
    cout << "input 2th number: ";
    cin >> i2;
    cout<< "add: " << add(i1, i2) << endl;
    cout<< "sub: " << sub(i1, i2) << endl;

    cout << "-------------" << endl;
    string str;
    cout << "input origin string: ";
    cin >> str;
    reverse(str);
    cout << "reversed string: " << str << endl;

    cout << "--------------" << endl;
    cout << "print hello from cpu/gpu:" << endl;
    printHello();

    cout << "--------------" << endl;
    cout << "test add vector:" << endl;
    int n = 0, *a = NULL, *b = NULL, *c = NULL;
    testAddVector(n, a, b, c);
    return 0;
}

void testAddVector(int& n, int *a, int *b, int *c) {
    n = 0;
    a = (int *)malloc(sizeof(int) * n);
    b = (int *)malloc(sizeof(int) * n);
    c = (int *)malloc(sizeof(int) * n);
    cout << "input length of vector: ";
    cin >> n;

    cout << "input the vector a: ";
    for(int i = 0; i < n; ++i) cin >> a[i];
    cout << "input the vector b: ";
    for(int i = 0; i < n; ++i) cin >> b[i];

    addVector(a, b, c, n);
    cout << "a + b = [";
    for(int i = 0; i < n; ++i) {
        cout << c[i] << ", ";
    }
    cout << "]" << endl;
}
