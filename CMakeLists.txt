cmake_minimum_required(VERSION 3.1.0)
# disable default -rdynamic option. https://tinyurl.com/ycg93bjq
cmake_policy(SET CMP0065 NEW)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache) # Less useful to do it for linking, see edit2
    message(STATUS "CCACHE FOUND")
endif(CCACHE_FOUND)

message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

# set CXX standard: 98, 11, 14, 17
#set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(init)
project(hello C CXX)


if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)



set(PROJECT_VERSION 2.0.0)
set(EXECUTABLE_NAME ${PROJECT_NAME} CACHE STRING "")

find_package(Threads REQUIRED)

enable_testing()


# set compile options
option(HELLO_TOGGLE "this is hello toggle" ON)
option(WITH_CUDA "this is cuda toggle" OFF)

if(WITH_CUDA)
    enable_language(CUDA)
endif()

configure_file(config.h.in ${CMAKE_BINARY_DIR}/include/config.h @ONLY)

add_subdirectory(math)
add_subdirectory(string)
add_subdirectory(print_hello)
add_subdirectory(test)

include_directories("${CMAKE_BINARY_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/math")
include_directories("${PROJECT_SOURCE_DIR}/string")
include_directories("${PROJECT_SOURCE_DIR}/print_hello")
include_directories(${CMAKE_BINARY_DIR}/third-party/gflags/include)
link_directories(${CMAKE_BINARY_DIR}/third-party/gflags/lib)

aux_source_directory(. SRC_LIST)
add_executable(${EXECUTABLE_NAME} ${SRC_LIST})

install(TARGETS ${EXECUTABLE_NAME} DESTINATION bin)

# search module in ${CMAKE_MODULE_PATH}
include(gflags)

target_link_libraries(${EXECUTABLE_NAME} math)
target_link_libraries(${EXECUTABLE_NAME} string)
target_link_libraries(${EXECUTABLE_NAME} print_hello)
target_link_libraries(${EXECUTABLE_NAME} Threads::Threads)
target_link_libraries(${EXECUTABLE_NAME} gflags)

# reference: https://yngzmiao.blog.csdn.net/article/details/102797448
message(STATUS "CMAKE_COMMAND: ${CMAKE_COMMAND}")
add_custom_target(echo
    COMMAND ${CMAKE_COMMAND} -E echo "echo target: ${PROJECT_NAME}:${PROJECT_VERSION}"
    COMMENT "this is echo from add_custom_target"
)

add_custom_target(echo_all ALL
    COMMAND ${CMAKE_COMMAND} -E echo "echo target: ${PROJECT_NAME}:${PROJECT_VERSION}"
    COMMENT "this is echo_all from add_custom_target"
)

add_custom_command(OUTPUT echo_cmd
    COMMAND ${CMAKE_COMMAND} -E echo "echo target: ${PROJECT_NAME}:${PROJECT_VERSION}"
    COMMENT "this is echo from add_custom_command"
)

add_custom_target(echo_cmd ALL
    DEPENDS echo_cmd
)

add_custom_command(OUTPUT echo_cmd_new
    COMMAND ${CMAKE_COMMAND} -E echo "echo target: ${PROJECT_NAME}:${PROJECT_VERSION}"
    COMMAND cmake -E copy /hello/build/lib/libmath.a /tmp/libmath.a
    COMMENT "this is echo_cmd_new"
    DEPENDS math
)
add_custom_target(copy ALL DEPENDS echo_cmd_new)

