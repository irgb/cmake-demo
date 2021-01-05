#include "mlir/IR/PatternMatch.h"

#include "dialect/test.h"

#include <iostream>

using namespace mlir;

namespace {
#include "dialect/test_rewriters.inc"
}

namespace test {

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
/// 等价于 DRR 格式: "def SimplifyRedundantTranspose : Pat<(TransposeOp(TransposeOp $arg)), (replaceWithValue $arg)>;"
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};


/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList &results,
                                              mlir::MLIRContext *context) {
  results.insert<SimplifyRedundantTranspose>(context);
}


void ReshapeOp::getCanonicalizationPatterns(mlir::OwningRewritePatternList& results, mlir::MLIRContext* context) {
    std::cout << "ReshapeOp::getCanonicalizationPatterns" << std::endl;
    results.insert<ReshapeReshapeOptPattern>(context);
}

} // namespace test
