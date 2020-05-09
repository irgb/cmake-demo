include(ExternalProject)
set(PROJ_GFLAGS external_gflags)
set(GFLAGS_FILENAME "v2.2.1.zip")

set(GFLAG_ROOT          ${CMAKE_BINARY_DIR}/third-party/gflags)

message(STATUS "GFLAG_ROOT: ${GFLAG_ROOT}")
set(GFLAG_URL           https://github.com/gflags/gflags/archive/${GFLAGS_FILENAME})

set(GFLAG_CONFIGURE     cd ${GFLAG_ROOT}/src/${PROJ_GFLAGS} && cmake -D CMAKE_INSTALL_PREFIX=${GFLAG_ROOT} .)
set(GFLAG_MAKE          cd ${GFLAG_ROOT}/src/${PROJ_GFLAGS} && make)
set(GFLAG_INSTALL       cd ${GFLAG_ROOT}/src/${PROJ_GFLAGS} && make install)

ExternalProject_Add(${PROJ_GFLAGS}
        URL                   ${GFLAG_URL}
	DOWNLOAD_NAME         ${GFLAGS_FILENAME}
        PREFIX                ${GFLAG_ROOT}
        CONFIGURE_COMMAND     ${GFLAG_CONFIGURE}
        BUILD_COMMAND         ${GFLAG_MAKE}
        INSTALL_COMMAND       ${GFLAG_INSTALL}
)
