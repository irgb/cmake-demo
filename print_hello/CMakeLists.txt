AUX_SOURCE_DIRECTORY(./*.cpp DIR_PRINT_HELLO_CPU_SRC_LIST)
AUX_SOURCE_DIRECTORY(./*.cu DIR_PRINT_HELLO_CUDA_SRC_LIST)

FILE(GLOB CUDA_SRCS ./*.cu)
FILE(GLOB CPU_SRCS ./*.cpp ./*.c)

include_directories("${CMAKE_BINARY_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

if(WITH_CUDA)
    ADD_LIBRARY(print_hello ${CUDA_SRCS})
else(WITH_CUDA)
    ADD_LIBRARY(print_hello ${CPU_SRCS})
endif()
target_link_libraries(print_hello gcov)

INSTALL(TARGETS print_hello DESTINATION lib)
INSTALL(DIRECTORY ./ DESTINATION include/print_hello FILES_MATCHING PATTERN *.h)
