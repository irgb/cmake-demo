include(ExternalProject)
set(PROJ_GFLAGS external_gflags)

set(GFLAG_ROOT          ${CMAKE_BINARY_DIR}/third-party/gflags)
set(GFLAG_LIB_DIR       ${GFLAG_ROOT}/lib)
set(GFLAG_INCLUDE_DIR   ${GFLAG_ROOT}/include)

message(STATUS "GFLAG_ROOT: ${GFLAG_ROOT}")
set(GFLAG_URL           https://github.com/gflags/gflags/archive/v2.2.1.zip)
set(GFLAG_DOWNLOAD_CMD  wget ${GFLAG_URL})
set(GFLAG_CONFIGURE     cd ${GFLAG_ROOT}/src/${PROJ_GFLAGS} && cmake -D CMAKE_INSTALL_PREFIX=${GFLAG_ROOT} .)
set(GFLAG_MAKE          cd ${GFLAG_ROOT}/src/${PROJ_GFLAGS} && make)
set(GFLAG_INSTALL       cd ${GFLAG_ROOT}/src/${PROJ_GFLAGS} && make install)

ExternalProject_Add(${PROJ_GFLAGS}
        URL                   ${GFLAG_URL}
        #DOWNLOAD_COMMAND      ${GFLAG_DOWNLOAD_CMD}
        DOWNLOAD_NAME         gflags-2.2.2.zip
        PREFIX                ${GFLAG_ROOT}
        CONFIGURE_COMMAND     ${GFLAG_CONFIGURE}
        BUILD_COMMAND         ${GFLAG_MAKE}
        INSTALL_COMMAND       ${GFLAG_INSTALL}
)

# build ${PROJ_GFLAGS} before ${EXECUTABLE_NAME}
add_dependencies(${EXECUTABLE_NAME} ${PROJ_GFLAGS})
