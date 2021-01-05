set(LLVM_TARGET_DEFINITIONS test.td)
mlir_tablegen(test.h.inc -gen-op-decls)
mlir_tablegen(test.cpp.inc -gen-op-defs)
mlir_tablegen(test_dialect.h.inc --gen-dialect-decls)
mlir_tablegen(test_rewriters.inc -gen-rewriters)

add_public_tablegen_target(TestIncGen)

add_custom_target(test_inc_gen ALL DEPENDS TestIncGen)

aux_source_directory(. DIR_SRC_LIST)
list(FILTER DIR_SRC_LIST EXCLUDE REGEX "_test.c")

add_library(dialect ${DIR_SRC_LIST})
add_dependencies(dialect TestIncGen)
INSTALL(TARGETS dialect DESTINATION lib)

include_directories(${CMAKE_BINARY_DIR})
