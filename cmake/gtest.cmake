include(ExternalProject)
set(PROJ_GTEST external_gtest)
set(GTEST_FILENAME "release-1.8.1.zip")

set(GTEST_ROOT ${CMAKE_BINARY_DIR}/third-party/gtest)
message(STATUS "GTEST_ROOT: ${GTEST_ROOT}")

set(GTEST_URL           https://github.com/google/googletest/archive/${GTEST_FILENAME})

#set(GTEST_CONFIGURE     cd ${GTEST_ROOT}/src/${PROJ_GTEST} && cmake -DCMAKE_INSTALL_PREFIX=${GTEST_ROOT}/install -Dgtest_disable_pthreads=OFF .)
set(GTEST_CONFIGURE     cd ${GTEST_ROOT}/src/${PROJ_GTEST} && cmake .)
set(GTEST_MAKE          cd ${GTEST_ROOT}/src/${PROJ_GTEST} && make)
set(GTEST_INSTALL       cd ${GTEST_ROOT}/src/${PROJ_GTEST} && make install)

include_directories(${GTEST_ROOT}/install/include)
link_directories(${GTEST_ROOT}/install/lib)

ExternalProject_Add(${PROJ_GTEST} 
    URL                   ${GTEST_URL}
    DOWNLOAD_NAME         ${GTEST_FILENAME}
    PREFIX                ${GTEST_ROOT}
    CONFIGURE_COMMAND     ${GTEST_CONFIGURE}
    BUILD_COMMAND         ${GTEST_MAKE}
    INSTALL_COMMAND       ${GTEST_INSTALL}
    CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=${GTEST_ROOT}/install
                          -Dgtest_disable_pthreads=OFF
)

# build ${PROJ_GFLAGS} before ${EXECUTABLE_NAME}
ADD_DEPENDENCIES(${EXECUTABLE_NAME} ${PROJ_GTEST})
