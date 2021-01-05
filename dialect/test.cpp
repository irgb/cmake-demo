#include <iostream>
#include "dialect/test.h"
#include "dialect/types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/TableGen/OpTrait.h"

using namespace std;
using namespace test;

Global::Global() {};

mlir::MLIRContext* Global::context = nullptr;

mlir::MLIRContext* Global::getMLIRContext() {
    if(context == nullptr) {
        context = new mlir::MLIRContext();
        //std::cerr << "new mlir::MLIRContext " << context << std::endl;
    }
    //std::cerr << "getMLIRContext: " << context << std::endl;
    return context;
}

void TestDialect::initialize() {
    allowUnknownTypes();
    allowUnknownOperations();

    addTypes<ComputeType>();

    addOperations<
#define GET_OP_LIST
#include "dialect/test.cpp.inc"
      >();
}


static mlir::LogicalResult verify(ConstantOp op) {
    cout << "verify ConstantOp" << endl;
    return mlir::success();
}

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return mlir::failure();

  result.addTypes(value.getType());
  return mlir::success();
}

static void print(mlir::OpAsmPrinter &printer, ConstantOp op) {
  printer << "toy.constant ";
  printer.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});
  printer << op.value();
}

// verifier for ComputeConstantOp.
static mlir::LogicalResult verify(ComputeConstantOp op) {
    auto resultType = op.getResult().getType().dyn_cast<test::ComputeType>();
    //std::cerr << "result type:" << typeid(resultType).name() << std::endl;
    mlir::StringRef value = op.value();
    //std::cerr << "value size:" << value.size() << std::endl;
    if (resultType.getWidth() != value.size()) {
        return op.emitOpError("return type width must match the size of"
                              " the attached value attribute: ")
            << resultType.getWidth() << " != " << value.size();
    }
    return mlir::success();
}

mlir::Type TestDialect::parseType(mlir::DialectAsmParser &parser) const {
    if (parser.parseKeyword("compute_type") || parser.parseLess()) {
        return mlir::Type();
    }
    unsigned int width = 0;
    if(parser.parseInteger(width)) {
        return mlir::Type();
    }
    //std::cerr << "parseType charType context: " << charType.getContext() << std::endl;
    if(parser.parseGreater()) {
        return mlir::Type();
    }
    return ComputeType::get(width);
    //return ComputeType::get(width, Global::getMLIRContext());
}

void TestDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
    ComputeType computeType = type.cast<ComputeType>();
    printer << "compute_type<" << computeType.getWidth() << ">";
}

//===----------------------------------------------------------------------===//
// TransposeOp

void TransposeOp::build(mlir::OpBuilder &builder,
                        mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

#define GET_OP_CLASSES
#include "dialect/test.cpp.inc"
