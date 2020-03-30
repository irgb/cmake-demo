cmake_minimum_required(VERSION 3.1.0)
message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# set CXX standard: 98, 11, 14, 17
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g -ggdb" CACHE STRING "Flags used by the compiler during debug builds.")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall" CACHE STRING "Flags used by the compiler during release builds.")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
RelWithDebInfo MinSizeRel.")
ENDIF()

message(STATUS "BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

project(hello)
set(PROJECT_VERSION 2.0.0)

find_package(Threads REQUIRED)

enable_testing()

# search module in ${CMAKE_MODULE_PATH}
include(gflags)

# set compile options
option(HELLO_TOGGLE "this is hello toggle" ON)

configure_file(config.h.in config.h @ONLY)

add_subdirectory(math)
add_subdirectory(string)
add_subdirectory(test)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}/math")
include_directories("${PROJECT_SOURCE_DIR}/string")
include_directories(${CMAKE_BINARY_DIR}/third-party/gflag-2.2.2/include)
link_directories(${CMAKE_BINARY_DIR}/third-party/gflag-2.2.2/lib)

aux_source_directory(. SRC_LIST)
add_executable(main ${SRC_LIST})

message(STATUS "THREADS_FOUND: ${THREADS_FOUND}")
message(STATUS "CMAKE_THREAD_LIBS_INIT: ${CMAKE_THREAD_LIBS_INIT}")
target_link_libraries(main math)
target_link_libraries(main string)
target_link_libraries(main Threads::Threads)
target_link_libraries(main gflags)
add_dependencies(main gflag-2.2.2)
