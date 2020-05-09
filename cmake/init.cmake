# default: "-g"
set(CMAKE_C_FLAGS_DEBUG "-g")
# default: "-O3 -DNDEBUG"
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
# default: "-O2 -g -DNDEBUG"
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
# default: "-Os -DNDEBUG"
set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# default: "-g"
set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb -Wall")
# default: "-O3 -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")
# default: "-O2 -g -DNDEBUG"
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
# default: "-Os -DNDEBUG"
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# default: "-g"
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
# default: "-O3 -DNDEBUG"
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
# default: "-O2 -g -DNDEBUG"
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
# default: "-O1 -DNDEBUG"
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG")

# init CMAKE_EXE_LINKER_FLAGS
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")

# check if cxx compiler support c++11
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
#
## set CMAKE_CXX_FLAGS
#if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#else()
#    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
#endif()
#
## set CMAKE_CUDA_FLAGS
#if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
#else()
#    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
#endif()
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")

# add compiler option to CXX Compiler
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-function>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>")

# add compiler option to CUDA Compiler
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-omit-frame-pointer>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-parameter>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-function>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fdata-sections>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections>")
