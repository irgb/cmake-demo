#include "mlir/Support/MlirOptMain.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "dialect/test.h"
#include "dialect/types.h"

int main(int argc, char** argv) {
    //std::cerr << "main" << std::endl;
    mlir::MLIRContext *context = Global::getMLIRContext();
    //context->getOrLoadDialect<test::TestDialect>();
    //context->getOrLoadDialect<mlir::StandardOpsDialect>();
    //std::cerr << "main allowUnregisteredDialects" << std::endl;
    context->allowUnregisteredDialects();
    //std::cerr << "main insert<test::TestDialect>" << std::endl;
    context->getDialectRegistry().insert<test::TestDialect>();
    //std::cerr << "main insert<mlir::StandardOpsDialect>" << std::endl;
    context->getDialectRegistry().insert<mlir::StandardOpsDialect>();
    //context->getOrLoadDialect(test::TestDialect::getDialectNamespace());
    //context->getDialectRegistry().loadByName(test::TestDialect::getDialectNamespace(), context);
    context->getOrLoadDialect<test::TestDialect>();
    //std::cerr << "main MlirOptMain" << std::endl;
    return failed(mlir::MlirOptMain(argc, argv, "mlir pass driver\n", context->getDialectRegistry()));
}
