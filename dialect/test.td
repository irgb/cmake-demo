#ifndef TEST_OPS
#define TEST_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//def StringType : OpaqueType<"demo", "string", "!demo.string type">;

// Provide a definition of the 'test' dialect in the ODS framework so that we
// can define our operations.
def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

def ComputeType :
    Type<CPred<"$_self.isa<ComputeType>()">, "test compute type">,
    BuildableType<"$_builder.getType<test::ComputeType>()">;

class Test_Op<string mnemonic, list<OpTrait> traits = []> : Op<Test_Dialect, mnemonic, traits>;

def ConstantOp : Test_Op<"constant", [ConstantLike, NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor:$result);

  //let parser = [{ return ::parseConstantOp(parser, result); }];
  //let printer = [{ return ::print(p, *this); }];

  //let assemblyFormat = "attr-dict `:` type($result)";
  //let assemblyFormat = "attr-dict `:` functional-type(operands, $result)";
  //let assemblyFormat = "operands attr-dict `:` type($result)";
  //let verifier = [{ return ::verify(*this); }];
}

def ConstantTensorOp : Test_Op<"constant_tensor", [NoSideEffect]> {
  let summary = "constant tensor";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor:$result);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseConstantOp(parser, result); }];
  //let printer = [{ return ::print(p, *this); }];

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  //let builders = [
  //  // Build a constant with a given constant tensor value.
  //  OpBuilder<"DenseElementsAttr value", [{
  //    build($_builder, $_state, value.getType(), value);
  //  }]>,

  //  // Build a constant with a given constant floating-point value.
  //  OpBuilder<"double value">
  //];

  let assemblyFormat = "attr-dict `:` type(results)";

  // Invoke a static verify method to verify this constant operation.
  //let verifier = [{ return ::verify(*this); }];
}

def AddOp : Test_Op<"add"> {
  let summary = "i32 addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];

  // Allow building an AddOp with from the two input operands.
  //let builders = [
  //  OpBuilder<"Value lhs, Value rhs">
  //];
}

def SubOp : Test_Op<"sub"> {
  let summary = "i32 subtract operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32);

  // Specify a parser and printer method.
  //let parser = [{ return ::parseBinaryOp(parser, result); }];
  //let printer = [{ return ::printBinaryOp(p, *this); }];

  // Allow building an AddOp with from the two input operands.
  //let builders = [
  //  OpBuilder<"Value lhs, Value rhs">
  //];
}

def ComputeConstantOp : Test_Op<"compute_constant", [ConstantLike, NoSideEffect]> {
  let summary = "compute constant";
  let description = [{
    Constant operation turns a literal struct value into an SSA value. The data
    is attached to the operation as an attribute. The struct constant is encoded
    as an array of other constant values. For example:

    ```mlir
       %op = test.compute_constant '+' : !test.compute_type<1>
    ```
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs ComputeType:$output);

  let assemblyFormat = "$value attr-dict `:` type($output)";

  let verifier = [{ return ::verify(*this); }];
  //let hasFolder = 1;
}

def ReshapeOp : Test_Op<"reshape", [NoSideEffect]> {
    let summary = "reshape operation";

    let description = [{
        reshape operation turns a literal into an SSA value. The data is attached
        to the operation as an attribute. For example:

        ```
          %0 = test.reshape dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
        ```
    }];

    let arguments = (ins F64Tensor:$input);

    let results = (outs StaticShapeTensorOf<[F64]>);

    //let parser = [{ return ::parser$cppClass(parser, result); }];
    //let printer = [{ return ::print(p, *this); }];
    //let assemblyFormat = [{ $attr0 attr-dict `:` $operands }];

    //let builders = [
    //    // build a reshape with a given reshape tensor value.
    //    OpBuilder<"DenseElementAttr value", [{
    //        build($_builder, $_state, value.getType(), value);
    //    }]>,
    //    // build a reshape with a given reshape floating-point value.
    //    //OpBuilder<"double value">
    //];

    //let verifier = [{ return ::verify(*this); }];

    let assemblyFormat = "`(` $input `:` type($input) `)` attr-dict `to` type(results)";
    let hasCanonicalizer = 1;
}

def ReturnOp : Test_Op<"return", [NoSideEffect, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        toy.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  //let builders = [
  //  OpBuilder<"", [{ build($_builder, $_state, llvm::None); }]>
  //];

  // Provide extra utility definitions on the c++ operation class definition.
  //let extraClassDeclaration = [{
  //  bool hasOperand() { return getNumOperands() != 0; }
  //}];

  // Invoke a static verify method to verify this return operation.
  //let verifier = [{ return ::verify(*this); }];
}

def TransposeOp : Test_Op<"transpose", [NoSideEffect]> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  // Allow building a TransposeOp with from the input operand.
  //let builders = [
  //  OpBuilder<"OpBuilder &b, OperationState &state, Value input">
  //];

  // Invoke a static verify method to verify this transpose operation.
  //let verifier = [{ return ::verify(*this); }];
}

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;

#endif // TEST_OPS
