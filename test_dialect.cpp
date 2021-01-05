#include <iostream>

#include "mlir/Parser.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "dialect/test.h"
#include "dialect/types.h"

using namespace std;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

llvm::raw_ostream & printIndent(int indent) {
    for (int i = 0; i < indent; ++i)
        llvm::outs() << "    ";
    return llvm::outs();
}

void printOperation(mlir::Operation *op, int indent);
void printRegion(mlir::Region &region, int indent);
void printBlock(mlir::Block &block, int indent);

void printOperation(mlir::Operation *op, int indent) {
    printIndent(indent) << "op: '" << op->getName()
        << "' with " << op->getNumOperands() << " operands"
        << ", " << op->getNumResults() << " results"
        << ", " << op->getAttrs().size() << " attributes"
        << ", " << op->getNumRegions() << " regions\n";
    if (!op->getAttrs().empty()) {
        printIndent(indent) << op->getAttrs().size() << " attributes:\n";
        for (mlir::NamedAttribute attr : op->getAttrs()) {
            printIndent(indent + 1) << "- {" << attr.first << " : " << attr.second << "}\n";
        }
    }
    
    if (op->getNumRegions() > 0) {
        printIndent(indent) << op->getNumRegions() << " nested regions:\n";
        for (mlir::Region &region : op->getRegions()) {
            printRegion(region, indent + 1);
        }
    }
}

void printRegion(mlir::Region &region, int indent) {
    printIndent(indent) << "Region with " << region.getBlocks().size() << " blocks:\n";
    for (mlir::Block &block : region.getBlocks()) {
        printBlock(block, indent + 1);
    }
}

void printBlock(mlir::Block &block, int indent) {
    printIndent(indent) << "Block with " << block.getNumArguments() << " arguments"
        << ", " << block.getNumSuccessors() << " successors"
        << ", " << block.getOperations().size() << " operations\n";

    for (mlir::Operation &operation : block.getOperations()) {
        printOperation(&operation, indent + 1);
    }
}

int main(int argc, char** argv) {
    cl::ParseCommandLineOptions(argc, argv, "mlir demo");

    //llvm::raw_ostream &out = llvm::outs();
    //out << "hello world\n";
    string mlir_source = R"src(
module @my$module {
func @func() {
    %1:1 = "test.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    //%2 = test.constant {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> } : () -> tensor<2x3xf64>
    test.return
}
func @main() {
    %1 = "test.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    //%2 = test.constant {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> } : () -> tensor<2x3xf64>
    return
}
}
)src";

    string mlir_source_2 = R"src(
// define a type alias
!i32_alias = type i32

// ModuleOp contains a single region containing a single block.
module @module_symbol attributes {module.attr="this is attr"} {
  // A function is a FuncOp containing a single region, i.e. function body.
  func @add(%arg0 : i32, %arg1 : i32) -> i32 {
      // "%res:N" means %res contains N results.
      %res:1 = "test.add"(%arg0, %arg1) {} : (i32, i32) -> i32
      // "%res#N" means get the N-th result in %res.
      std.return %res#0 : i32
  }
  
  func @sub(%arg0 : i32, %arg1 : i32) -> i32 {
      %res = "test.sub"(%arg0, %arg1) {} : (i32, i32) -> i32
      // op "return" use default "std" namespace.
      return %res : i32
  }

  // use std.constant to define value.
  // "std.constant" is a custom operation with a custom assembly form.
  %1 = constant 2 : i32
  %2 = constant 3 : !i32_alias

  // "get_string" is generic operation has no custom assembly form.
  //%op = "test.compute_constant"() {value="+"}: () ->!test.compute_type
  %op = test.compute_constant "+" : !test.compute_type<1>

  %res = "do_async"(%1, %2, %op) ({
  //^bb0
    %is_add = "is_add"(%op) : (!test.compute_type<1>) -> i1
    //"cond_br"(%is_add)[^bb1, ^bb2] : (i1) -> ()
    cond_br %is_add, ^bb1(%1, %2: i32, i32), ^bb2(%1, %2: i32, i32)

  ^bb1(%arg00 : i32, %arg01 : i32):
    %br1_res = call @add(%arg00, %arg01) : (i32, i32) -> i32
    "dialect.innerop7"(%1, %2) : (i32, i32) -> ()
    br ^bb3(%1, %2: i32, i32)

  ^bb2(%arg10 : i32, %arg11 : i32):
    %br2_res = call @sub(%arg10, %arg11) : (i32, i32) -> i32
    "dialect.innerop7"(%br2_res) : (i32) -> ()

  ^bb3(%arg20 : i32, %arg21 : i32):
    "dialect.innerop7"() : () -> ()

  }, {}) : (i32, i32, !test.compute_type<1>) -> (i32)
  
  // module_terminator will be add implicitly to the end of a module.  
  "module_terminator"() : () -> ()
}
)src";

    mlir::MLIRContext *context = Global::getMLIRContext();
    context->allowUnregisteredDialects();
    context->getDialectRegistry().insert<test::TestDialect>();
    context->getDialectRegistry().insert<mlir::StandardOpsDialect>();

    //mlir::OwningModuleRef module_ref = mlir::parseSourceString(mlir_source_2, context);
    mlir::OwningModuleRef module_ref = mlir::parseSourceFile(inputFilename, context);
    std::cout << "----------print operation, region, block begin----------" << std::endl;
    printOperation(module_ref->getOperation(), 0);
    std::cout << "----------print operation, region, block end----------" << std::endl;

    mlir::ModuleOp module_op = module_ref.get();

    if (failed(mlir::verify(module_op))) {
        module_op.emitError("module verification error");
        return 1;
    } else {
        cout << "verify module \"" << module_op.getOperationName().data() << "\" success" << endl;
    }
    mlir::Block *block = module_ref->getBody();
    mlir::Region &region = module_ref->getBodyRegion();
    for (auto iter = module_ref->begin(); iter != module_ref->end(); ++iter) {
        mlir::Operation &op = *iter;
        mlir::Dialect *dialect = op.getDialect();
        string dialect_name = dialect ? dialect->getNamespace().str() : "null";
        cout << op.getName().getStringRef().str() << ", namespace: " << dialect_name << endl;
        if (llvm::isa<mlir::FuncOp>(op)) {
            mlir::FuncOp func_op = llvm::dyn_cast<mlir::FuncOp>(op);
            cout << "func_op: " << func_op.getName().str() << endl;
        } else {
            cout << "non func_op:" << op.getName().getStringRef().str() << endl;
        }
    }

    module_ref->dump();

    // ==========test PassManager begin====================
    std::cout << "==========test PassManager begin==============" << std::endl;
    mlir::PassManager passManager(context);
    // addPass 会调用 getCanonicalizationPatterns
    passManager.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    if (mlir::failed(passManager.run(*module_ref))) {
        std::cerr << "PassManager run failed" << std::endl;
    } else {
        std::cout << "PassManager run succeeded." << std::endl;
    }
    module_ref->dump();
    std::cout << "==========test PassManager end==============" << std::endl;
    // ==========test PassManager end===================
    return 0;
}
