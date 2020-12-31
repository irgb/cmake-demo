#ifndef TEST_DIALECT_H_
#define TEST_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/TableGen/OpTrait.h"

using namespace mlir;

// global variables
class Global {
private:
    static mlir::MLIRContext *context;
    Global();
public:
    static MLIRContext *getMLIRContext();
}; // class Global

namespace test {
namespace detail {
struct ComputeTypeStorage; 
} // end namespace detail

class TestDialect : public ::mlir::Dialect {
  explicit TestDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<TestDialect>()) {
    
    initialize();
  }

  // parse an instance of a type registered to the dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  // print an instance of a type registered to the dialect.
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  static ::llvm::StringRef getDialectNamespace() { return "test"; }
};

/// Include the auto-generated header file containing the declarations of the
/// toy operations.

#define GET_OP_CLASSES
#include "dialect/test.h.inc"

class ComputeType : public Type::TypeBase<ComputeType, Type, detail::ComputeTypeStorage> {
public:
    using Base::Base;

    static ComputeType get(int width, MLIRContext *context);
    int getWidth();
};

} // end namespace 

#endif // TEST_DIALECT_H_
