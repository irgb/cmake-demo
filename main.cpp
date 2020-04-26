#include <iostream>

#include "gflags/gflags.h"

#include "config.h"
#include "add.h"
#include "sub.h"
#include "reverse.h"
#include "print_hello.h"

using namespace std;

DEFINE_bool(cpp_version, false, "print cpp version");

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

    int i1 = 0, i2 = 0;
    cout << "input 1th number: ";
    cin >> i1;
    cout << "input 2th number: ";
    cin >> i2;
    cout << "------" << endl;
    cout<< "add: " << add(i1, i2) << endl;
    cout<< "sub: " << sub(i1, i2) << endl;

    string str;
    cout << "input origin string: ";
    cin >> str;
    reverse(str);
    cout << "reversed string: " << str << endl;

    printHello();
    return 0;
}
