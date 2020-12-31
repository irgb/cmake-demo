#include "llvm/ADT/bit.h"
#include "mlir/IR/Types.h"

namespace test {
namespace detail {
struct ComputeTypeStorage : public TypeStorage {
    ComputeTypeStorage(char op): op(op) {}
    using KeyTy = char;

    bool operator==(const KeyTy &key) {
        return key == KeyTy(op);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key);
    }

    static ComputeTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy key) {
        return new(allocator.allocat<ComputeTypeStorage>())
            ComputeTypeStorage(key);
    }

    char op;
};

} // namespace detail

class ComputeType : public Type::TypeBase<ComputeType, Type, detail::ComputeTypeStorage> {
public:
    using Base::Base;

    static ComputeType get(char op, MLIRContext *context) {
        return Base::get(context, op);
    }

    char getOp() {
        return getImpl() -> op;
    }

    bool isAdd() { return getOp() == '+'; }
    bool isSub() { return getOp() == '-'; }
};

} // namespace test
