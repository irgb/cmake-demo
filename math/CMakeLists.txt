AUX_SOURCE_DIRECTORY(. DIR_MATH_SRC_LIST)

add_library(math ${DIR_MATH_SRC_LIST})

INSTALL(TARGETS math DESTINATION lib)
INSTALL(DIRECTORY ./ DESTINATION include/math FILES_MATCHING PATTERN *.h)

# user find_library and target_link_libraries is better than include_directories
# https://stackoverflow.com/a/31471772/5432806
find_library(GTEST_LIBRARY gtest PATHS ${CMAKE_BINARY_DIR}/third-party/gtest/install/lib)
find_library(GTEST_MAIN_LIBRARY gtest_main PATHS ${CMAKE_BINARY_DIR}/third-party/gtest/install/lib)
#link_directories(${CMAKE_BINARY_DIR}/third-party/gtest/install/lib)

include_directories(${CMAKE_BINARY_DIR}/third-party/gtest/install/include)

add_executable(add_test add_test.cpp)
target_link_libraries(add_test math)
target_link_libraries(add_test Threads::Threads ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})

add_test(NAME add_test COMMAND $<TARGET_FILE:add_test>)
