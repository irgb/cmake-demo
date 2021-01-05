func @transpose_transpose(%arg0: tensor<2x3xf64>) -> tensor<2x3xf64> {
  %0 = test.transpose(%arg0 : tensor<2x3xf64>) to tensor<2x3xf64>
  %1 = test.transpose(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
  return %1 : tensor<2x3xf64>
}

func @main() -> () {
    %0 = test.constant_tensor {value=dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf64>} : tensor<6xf64>
    // test reshape canonicalizer
    %1 = test.reshape(%0 : tensor<6xf64>) to tensor<3x2xf64>
    %2 = test.reshape(%1 : tensor<3x2xf64>) to tensor<2x3xf64>

    // test transpose canonicalizer
    %4 = call @transpose_transpose(%2) : (tensor<2x3xf64>) -> tensor<2x3xf64>
    return
}
