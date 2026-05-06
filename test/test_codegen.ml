open Tesserae

let i n    = Modes.Int n
let tup ts = Modes.Tuple ts
let lay s d = Layout.make s d

(* --- emit_shape --- *)
let test_shape_scalar () =
  Alcotest.(check string) "scalar" "_4"
    (Codegen.emit_shape (i 4))

let test_shape_flat () =
  Alcotest.(check string) "flat" "Shape<_2,_3>"
    (Codegen.emit_shape (tup [i 2; i 3]))

let test_shape_nested () =
  Alcotest.(check string) "nested" "Shape<_2,Shape<_3,_4>>"
    (Codegen.emit_shape (tup [i 2; tup [i 3; i 4]]))

(* --- emit_stride --- *)
let test_stride_scalar () =
  Alcotest.(check string) "scalar" "_1"
    (Codegen.emit_stride (i 1))

let test_stride_flat () =
  Alcotest.(check string) "flat" "Stride<_1,_4>"
    (Codegen.emit_stride (tup [i 1; i 4]))

let test_stride_nested () =
  Alcotest.(check string) "nested" "Stride<_1,Stride<_2,_8>>"
    (Codegen.emit_stride (tup [i 1; tup [i 2; i 8]]))

(* --- emit_layout --- *)
let test_layout_simple () =
  Alcotest.(check string) "simple"
    "Layout<Shape<_2,_3>,Stride<_1,_2>>"
    (Codegen.emit_layout (lay (tup [i 2; i 3]) (tup [i 1; i 2])))

let test_layout_scalar () =
  Alcotest.(check string) "scalar"
    "Layout<_8,_1>"
    (Codegen.emit_layout (lay (i 8) (i 1)))

let test_layout_nested () =
  Alcotest.(check string) "nested"
    "Layout<Shape<_2,Shape<_3,_4>>,Stride<_1,Stride<_2,_6>>>"
    (Codegen.emit_layout
       (lay (tup [i 2; tup [i 3; i 4]])
            (tup [i 1; tup [i 2; i 6]])))

(* --- emit_tensor_type --- *)
let test_tensor_type_global () =
  Alcotest.(check string) "global float"
    "Tensor<float*, Layout<Shape<_2,_3>,Stride<_1,_2>>>"
    (Codegen.emit_tensor_type Elemtype.Float32 Memspace.Global
       (lay (tup [i 2; i 3]) (tup [i 1; i 2])))

let test_tensor_type_shared () =
  Alcotest.(check string) "shared half"
    "Tensor<__half*, Layout<_8,_1>>"
    (Codegen.emit_tensor_type Elemtype.Float16 Memspace.Shared
       (lay (i 8) (i 1)))

(* --- emit_make_tensor --- *)
let test_make_tensor () =
  Alcotest.(check string) "make_tensor"
    "make_tensor<float>(ptr_A, Layout<Shape<_2,_3>,Stride<_1,_2>>{})"
    (Codegen.emit_make_tensor "ptr_A" Elemtype.Float32 Memspace.Global
       (lay (tup [i 2; i 3]) (tup [i 1; i 2])))

(* --- emit_smem_decl --- *)
let test_smem_decl () =
  Alcotest.(check string) "smem"
    "__shared__ float smem_A[6];"
    (Codegen.emit_smem_decl "smem_A" Elemtype.Float32
       (lay (tup [i 2; i 3]) (tup [i 1; i 2])))

(* --- emit_include_guard --- *)
let test_include_guard () =
  Alcotest.(check string) "pragma"
    "#pragma once"
    (Codegen.emit_include_guard ())

(* --- emit_cute_includes --- *)
let test_cute_includes () =
  let s = Codegen.emit_cute_includes () in
  let found = ref false in
  let needle = "cute/cute.hpp" in
  let nlen = String.length needle in
  let slen = String.length s in
  for i = 0 to slen - nlen do
    if String.sub s i nlen = needle then found := true
  done;
  Alcotest.(check bool) "has cute.hpp" true !found

(* --- runner --- *)
let () =
  Alcotest.run "Codegen" [
    "shape",   [ Alcotest.test_case "scalar" `Quick test_shape_scalar
               ; Alcotest.test_case "flat"   `Quick test_shape_flat
               ; Alcotest.test_case "nested" `Quick test_shape_nested ];
    "stride",  [ Alcotest.test_case "scalar" `Quick test_stride_scalar
               ; Alcotest.test_case "flat"   `Quick test_stride_flat
               ; Alcotest.test_case "nested" `Quick test_stride_nested ];
    "layout",  [ Alcotest.test_case "simple" `Quick test_layout_simple
               ; Alcotest.test_case "scalar" `Quick test_layout_scalar
               ; Alcotest.test_case "nested" `Quick test_layout_nested ];
    "tensor",  [ Alcotest.test_case "global" `Quick test_tensor_type_global
               ; Alcotest.test_case "shared" `Quick test_tensor_type_shared ];
    "make",    [ Alcotest.test_case "basic"  `Quick test_make_tensor ];
    "smem",    [ Alcotest.test_case "basic"  `Quick test_smem_decl ];
    "guard",   [ Alcotest.test_case "pragma" `Quick test_include_guard ];
    "includes",[ Alcotest.test_case "cute"   `Quick test_cute_includes ];
  ]
