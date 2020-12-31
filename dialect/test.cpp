#include <iostream>
#include "dialect/test.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/TableGen/OpTrait.h"

using namespace std;
using namespace test;

Global::Global() {};

mlir::MLIRContext* Global::context = nullptr;

MLIRContext* Global::getMLIRContext() {
    if(context == nullptr) {
        context = new mlir::MLIRContext();
    }
    return context;
}

void TestDialect::initialize() {
    allowUnknownTypes();
    allowUnknownOperations();

    addOperations<
#define GET_OP_LIST
#include "dialect/test.cpp.inc"
      >();
    addTypes<ComputeType>();
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

namespace test {
namespace detail {

struct ComputeTypeStorage : public mlir::TypeStorage {
    ComputeTypeStorage(int width): width(width) {}
    using KeyTy = char;

    bool operator==(const KeyTy &key) const {
        return key == KeyTy(width);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key);
    }

    static ComputeTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy key) {
        return new(allocator.allocate<ComputeTypeStorage>())
            ComputeTypeStorage(key);
    }

    int width;
};

} // end detail namespace
} // end test namespace


ComputeType ComputeType::get(int width, mlir::MLIRContext *context) {
    return Base::get(context, width);
}

int ComputeType::getWidth() {
    return getImpl()->width;
}

mlir::Type TestDialect::parseType(mlir::DialectAsmParser &parser) const {
    if (parser.parseKeyword("compute_type") || parser.parseLess()) {
        return mlir::Type();
    }
    int width = 0;
    if(parser.parseInteger(width)) {
        return mlir::Type();
    }
    if(parser.parseGreater()) {
        return mlir::Type();
    }
    return ComputeType::get(width, Global::getMLIRContext());
}

void TestDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const {
    ComputeType computeType = type.cast<ComputeType>();
    printer << "compute_type<" << computeType.getWidth() << ">";
}

#define GET_OP_CLASSES
#include "dialect/test.cpp.inc"
