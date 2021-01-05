#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "dialect/test.h"
#include "dialect/types.h"

int main(int argc, char** argv) {
    mlir::MLIRContext *context = Global::getMLIRContext();
    // 在顶层 ModuleOp上创建一个 PassManager
    mlir::PassManager passManager(context);
    // 在 ModuleOp/FuncOp 上创建一个 PassManager
    //mlir::OpPassManager &optPm = passManager.nest<mlir::FuncOp>();
    //optPm.addPass(hello::createLowerToAffinePass());
    passManager.addPass(mlir::createCanonicalizerPass());
    //mlir::registerAllPasses();

    //std::cerr << "main" << std::endl;
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
