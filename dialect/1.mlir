// define a type alias
!i32_alias = type i32

// ModuleOp contains a single region containing a single block.
module @module_symbol attributes {module.attr="this is attr"} {
  // A function is a FuncOp containing a single region, i.e. function body.
  func @add(%arg0 : i32, %arg1 : i32) -> i32 {
      // "%res:N" means %res contains N results.
      %res:1 = "test.add"(%arg0, %arg1) {} : (i32, i32) -> i32
      // "%res#N" means get the N-th result in %res.
      std.return %res#0 : i32
  }

  func @sub(%arg0 : i32, %arg1 : i32) -> i32 {
      %res = "test.sub"(%arg0, %arg1) {} : (i32, i32) -> i32
      // op "return" use default "std" namespace.
      return %res : i32
  }

  // use std.constant to define value.
  // "std.constant" is a custom operation with a custom assembly form.
  %1 = constant 2 : i32
  %2 = constant 3 : !i32_alias

  // "get_string" is generic operation has no custom assembly form.
  //%op = "test.compute_constant"() {value="+"}: () ->!test.compute_type<1>
  %op = test.compute_constant "+" : !test.compute_type<1>

  //******************
  //An SSACFG (SSA-style Control Flow Graph) region
  //******************
  %res = "do_async"(%1, %2, %op) ({
  //^bb0
    %is_add = "is_add"(%op) : (!test.compute_type<1>) -> i1
    // cond_br is a terminator operation, and it has 2 successors, i.e., ^bb1, ^bb2
    //"cond_br"(%is_add)[^bb1, ^bb2] : (i1) -> ()
    cond_br %is_add, ^bb1(%1, %2: i32, i32), ^bb2(%1, %2: i32, i32)

  ^bb1(%arg00 : i32, %arg01 : i32): // %arg00, %arg01 are BlockArgument, not BlockOperand
    %br1_res = call @add(%arg00, %arg01) : (i32, i32) -> i32
    "dialect.innerop7"(%1, %2) : (i32, i32) -> ()
    // br is a terminator operation, and it has 1 successor, i.e., ^bb3
    br ^bb3(%1, %2: i32, i32) // %1, %2 are BlockArgument, not BlockOperand

  ^bb2(%arg10 : i32, %arg11 : i32):
    %br2_res = call @sub(%arg10, %arg11) : (i32, i32) -> i32
    "dialect.innerop7"(%br2_res) : (i32) -> ()

  // ^bb2, ^bb3 contains no terminator, they have 0 successor.
  ^bb3(%arg20 : i32, %arg21 : i32):
    "dialect.innerop7"() : () -> ()

  }, {}) : (i32, i32, !test.compute_type<1>) -> (i32)

  // module_terminator will be add implicitly to the end of a module.
  "module_terminator"() : () -> ()
}
