func @main() {
    %1:1 = "test.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    //%2 = test.constant {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> } : () -> tensor<2x3xf64>
    test.return
}
