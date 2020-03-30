#include <algorithm>

#include "reverse.h"

void reverse(std::string& str) {
    int l = 0, r = str.length() - 1;
    while(l < r) {
        std::swap(str[l], str[r]);
        ++l;
        --r;
    }
}
