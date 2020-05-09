#include "gtest/gtest.h"

#include "add.h"

TEST(TestAdd, ArithmeticAddInt) {
    EXPECT_EQ(1, 1);
    EXPECT_EQ(3, add(1, 2));
}

