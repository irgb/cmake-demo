aux_source_directory(. DIR_MATH_SRC_LIST)
list(FILTER DIR_MATH_SRC_LIST EXCLUDE REGEX "_test.c")

add_library(math ${DIR_MATH_SRC_LIST})

INSTALL(TARGETS math DESTINATION lib)
INSTALL(DIRECTORY ./ DESTINATION include/math FILES_MATCHING PATTERN *.h)

link_directories(${CMAKE_BINARY_DIR}/third-party/gtest/install/lib)
include_directories(${CMAKE_BINARY_DIR}/third-party/gtest/install/include)

add_executable(add_test add_test.cpp)
add_dependencies(add_test external_gtest)
target_link_libraries(add_test math)
target_link_libraries(add_test Threads::Threads gtest gtest_main)

add_test(NAME add_test COMMAND $<TARGET_FILE:add_test>)
