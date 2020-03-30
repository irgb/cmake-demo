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
set(EXECUTABLE_NAME ${PROJECT_NAME} CACHE STRING "")

find_package(Threads REQUIRED)

enable_testing()


# set compile options
option(HELLO_TOGGLE "this is hello toggle" ON)

configure_file(config.h.in config.h @ONLY)

add_subdirectory(math)
add_subdirectory(string)
add_subdirectory(test)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories("${PROJECT_SOURCE_DIR}/math")
include_directories("${PROJECT_SOURCE_DIR}/string")
include_directories(${CMAKE_BINARY_DIR}/third-party/gflags/include)
link_directories(${CMAKE_BINARY_DIR}/third-party/gflags/lib)

aux_source_directory(. SRC_LIST)
add_executable(${EXECUTABLE_NAME} ${SRC_LIST})
install(TARGETS ${EXECUTABLE_NAME} DESTINATION bin)

# search module in ${CMAKE_MODULE_PATH}
include(gflags)

target_link_libraries(${EXECUTABLE_NAME} math)
target_link_libraries(${EXECUTABLE_NAME} string)
target_link_libraries(${EXECUTABLE_NAME} Threads::Threads)
target_link_libraries(${EXECUTABLE_NAME} gflags)

# reference: https://yngzmiao.blog.csdn.net/article/details/102797448
message(STATUS "CMAKE_COMMAND: ${CMAKE_COMMAND}")
add_custom_target(echo
    COMMAND ${CMAKE_COMMAND} -E echo "echo target: ${PROJECT_NAME}:${PROJECT_VERSION}"
    COMMENT "this is echo from add_custom_target"
)

add_custom_command(OUTPUT echo_cmd
    COMMAND ${CMAKE_COMMAND} -E echo "echo target: ${PROJECT_NAME}:${PROJECT_VERSION}"
    COMMENT "this is echo from add_custom_command"
)

add_custom_target(echo_cmd ALL
    DEPENDS echo_cmd
)
