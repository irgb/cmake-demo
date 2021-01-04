#ifndef DIALECT_TYPES_H_
#define DIALECT_TYPES_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace test {

namespace detail {
struct ComputeTypeStorage; 
} // end namespace detail

class ComputeType : public mlir::Type::TypeBase<ComputeType, mlir::Type, detail::ComputeTypeStorage> {
public:
    using Base::Base;

    static ComputeType get(unsigned int width);
    unsigned int getWidth();
};

} // end namespace test

#endif // DIALECT_TYPES_H_

