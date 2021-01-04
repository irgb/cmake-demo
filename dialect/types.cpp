#include <iostream>

#include "dialect/types.h"
#include "dialect/test.h"

namespace test {
namespace detail {

// =============================
// ComputeTypeStorage definition
// =============================
struct ComputeTypeStorage : public mlir::TypeStorage {
    ComputeTypeStorage(unsigned int width): width(width) {
        //std::cerr << "ComputeTypeStorage::ComputeTypeStorage" << std::endl;
    }
    using KeyTy = unsigned int;

    bool operator==(const KeyTy &key) const {
        return key == width;
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return llvm::hash_value(key);
    }

    static ComputeTypeStorage *construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
        return new(allocator.allocate<ComputeTypeStorage>())
            ComputeTypeStorage(key);
    }

    unsigned int width;
};

} // end detail namespace

// =======================
// ComputeType definition
// =======================
ComputeType ComputeType::get(unsigned int width) {
    //std::cerr << "ComputeType::get begin" << std::endl;
    //std::cerr << "ComputeType::get type context: " << type.getContext() << std::endl;
    auto ret = Base::get(Global::getMLIRContext(), width);
    //return Base::get(type.getContext(), width, type);
    //std::cerr << "ComputeType::get end" << std::endl;
    return ret;
}

unsigned int ComputeType::getWidth() {
    return getImpl()->width;
}

} // end namespace test
