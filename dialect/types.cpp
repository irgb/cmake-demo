#include <iostream>

#include "dialect/types.h"

namespace test {
namespace detail {

// =============================
// ComputeTypeStorage definition
// =============================
struct ComputeTypeStorage : public mlir::TypeStorage {
    ComputeTypeStorage(unsigned int width, mlir::Type charType): width(width), charType(charType) {
        //std::cerr << "ComputeTypeStorage::ComputeTypeStorage" << std::endl;
    }
    using KeyTy = std::pair<unsigned int, mlir::Type>;

    bool operator==(const KeyTy &key) const {
        return key == KeyTy(width, charType);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key);
    }

    static ComputeTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
        return new(allocator.allocate<ComputeTypeStorage>())
            ComputeTypeStorage(key.first, key.second);
    }

    unsigned int width;
    mlir::Type charType;
};

} // end detail namespace

// =======================
// ComputeType definition
// =======================
ComputeType ComputeType::get(unsigned int width, Type type) {
    //std::cerr << "ComputeType::get begin" << std::endl;
    //std::cerr << "ComputeType::get type context: " << type.getContext() << std::endl;
    auto ret = Base::get(type.getContext(), width, type);
    //return Base::get(type.getContext(), width, type);
    //std::cerr << "ComputeType::get end" << std::endl;
    return ret;
}

unsigned int ComputeType::getWidth() {
    return getImpl()->width;
}

} // end namespace test
