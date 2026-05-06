open Tesserae

(* --- cpp_name --- *)
let test_cpp_name_float32 () =
  Alcotest.(check string) "float32" "float"
    (Elemtype.cpp_name Elemtype.Float32)

let test_cpp_name_float16 () =
  Alcotest.(check string) "float16" "__half"
    (Elemtype.cpp_name Elemtype.Float16)

let test_cpp_name_bfloat16 () =
  Alcotest.(check string) "bfloat16" "__nv_bfloat16"
    (Elemtype.cpp_name Elemtype.Bfloat16)

let test_cpp_name_int8 () =
  Alcotest.(check string) "int8" "int8_t"
    (Elemtype.cpp_name Elemtype.Int8)

let test_cpp_name_int32 () =
  Alcotest.(check string) "int32" "int32_t"
    (Elemtype.cpp_name Elemtype.Int32)

(* --- byte_width --- *)
let test_byte_width_float32 () =
  Alcotest.(check int) "float32" 4 (Elemtype.byte_width Elemtype.Float32)

let test_byte_width_float16 () =
  Alcotest.(check int) "float16" 2 (Elemtype.byte_width Elemtype.Float16)

let test_byte_width_bfloat16 () =
  Alcotest.(check int) "bfloat16" 2 (Elemtype.byte_width Elemtype.Bfloat16)

let test_byte_width_int8 () =
  Alcotest.(check int) "int8" 1 (Elemtype.byte_width Elemtype.Int8)

let test_byte_width_int32 () =
  Alcotest.(check int) "int32" 4 (Elemtype.byte_width Elemtype.Int32)

(* --- bits --- *)
let test_bits_float32 () =
  Alcotest.(check int) "float32" 32 (Elemtype.bits Elemtype.Float32)

let test_bits_float16 () =
  Alcotest.(check int) "float16" 16 (Elemtype.bits Elemtype.Float16)

let test_bits_int8 () =
  Alcotest.(check int) "int8" 8 (Elemtype.bits Elemtype.Int8)

(* --- is_floating / is_integer --- *)
let test_is_floating_float32 () =
  Alcotest.(check bool) "float32 floating" true
    (Elemtype.is_floating Elemtype.Float32)

let test_is_floating_int32 () =
  Alcotest.(check bool) "int32 not floating" false
    (Elemtype.is_floating Elemtype.Int32)

let test_is_integer_int8 () =
  Alcotest.(check bool) "int8 integer" true
    (Elemtype.is_integer Elemtype.Int8)

let test_is_integer_float16 () =
  Alcotest.(check bool) "float16 not integer" false
    (Elemtype.is_integer Elemtype.Float16)

(* --- vec_width --- *)
let test_vec_width_float32 () =
  (* 128 / 32 = 4 *)
  Alcotest.(check int) "float32 vec" 4
    (Elemtype.vec_width Elemtype.Float32)

let test_vec_width_float16 () =
  (* 128 / 16 = 8 *)
  Alcotest.(check int) "float16 vec" 8
    (Elemtype.vec_width Elemtype.Float16)

let test_vec_width_int8 () =
  (* 128 / 8 = 16 *)
  Alcotest.(check int) "int8 vec" 16
    (Elemtype.vec_width Elemtype.Int8)

(* --- pp --- *)
let test_pp_float32 () =
  let s = Stdlib.Format.asprintf "%a" Elemtype.pp Elemtype.Float32 in
  Alcotest.(check string) "pp float32" "float" s

let test_pp_bfloat16 () =
  let s = Stdlib.Format.asprintf "%a" Elemtype.pp Elemtype.Bfloat16 in
  Alcotest.(check string) "pp bfloat16" "__nv_bfloat16" s

(* --- runner --- *)
let () =
  Alcotest.run "Elemtype" [
    "cpp_name",   [ Alcotest.test_case "float32"  `Quick test_cpp_name_float32
                  ; Alcotest.test_case "float16"  `Quick test_cpp_name_float16
                  ; Alcotest.test_case "bfloat16" `Quick test_cpp_name_bfloat16
                  ; Alcotest.test_case "int8"     `Quick test_cpp_name_int8
                  ; Alcotest.test_case "int32"    `Quick test_cpp_name_int32 ];
    "byte_width", [ Alcotest.test_case "float32"  `Quick test_byte_width_float32
                  ; Alcotest.test_case "float16"  `Quick test_byte_width_float16
                  ; Alcotest.test_case "bfloat16" `Quick test_byte_width_bfloat16
                  ; Alcotest.test_case "int8"     `Quick test_byte_width_int8
                  ; Alcotest.test_case "int32"    `Quick test_byte_width_int32 ];
    "bits",       [ Alcotest.test_case "float32"  `Quick test_bits_float32
                  ; Alcotest.test_case "float16"  `Quick test_bits_float16
                  ; Alcotest.test_case "int8"     `Quick test_bits_int8 ];
    "floating",   [ Alcotest.test_case "float32"  `Quick test_is_floating_float32
                  ; Alcotest.test_case "int32"    `Quick test_is_floating_int32 ];
    "integer",    [ Alcotest.test_case "int8"     `Quick test_is_integer_int8
                  ; Alcotest.test_case "float16"  `Quick test_is_integer_float16 ];
    "vec_width",  [ Alcotest.test_case "float32"  `Quick test_vec_width_float32
                  ; Alcotest.test_case "float16"  `Quick test_vec_width_float16
                  ; Alcotest.test_case "int8"     `Quick test_vec_width_int8 ];
    "pp",         [ Alcotest.test_case "float32"  `Quick test_pp_float32
                  ; Alcotest.test_case "bfloat16" `Quick test_pp_bfloat16 ];
  ]
