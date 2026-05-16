open Tesserae

let t_int n = Modes.Int n
let t_tup ts = Modes.Tuple ts

(* --- size --- *)
let test_size_leaf () =
  Alcotest.(check int) "leaf size" 8 (Modes.size (t_int 8))

let test_size_flat_tuple () =
  Alcotest.(check int) "flat tuple size" 24
    (Modes.size (t_tup [t_int 2; t_int 3; t_int 4]))

let test_size_nested () =
  (* shape (2, (3, 4)) equals 2 * 3 * 4 = 24 *)
  Alcotest.(check int) "nested size" 24
    (Modes.size (t_tup [t_int 2; t_tup [t_int 3; t_int 4]]))

(* --- depth --- *)
let test_depth_leaf () =
  Alcotest.(check int) "leaf depth" 0 (Modes.depth (t_int 1))

let test_depth_flat () =
  Alcotest.(check int) "flat depth" 1
    (Modes.depth (t_tup [t_int 2; t_int 3]))

let test_depth_nested () =
  Alcotest.(check int) "nested depth" 2
    (Modes.depth (t_tup [t_int 2; t_tup [t_int 3; t_int 4]]))

(* --- rank --- *)
let test_rank_leaf () =
  Alcotest.(check int) "leaf rank" 1 (Modes.rank (t_int 5))

let test_rank_tuple () =
  Alcotest.(check int) "tuple rank" 3
    (Modes.rank (t_tup [t_int 2; t_int 3; t_int 4]))

(* --- flatten --- *)
let test_flatten_leaf () =
  Alcotest.(check (list int)) "flatten leaf" [7] (Modes.flatten (t_int 7))

let test_flatten_nested () =
  Alcotest.(check (list int)) "flatten nested" [2; 3; 4]
    (Modes.flatten (t_tup [t_int 2; t_tup [t_int 3; t_int 4]]))

(* --- compatible --- *)
let test_compatible_int_int () =
  Alcotest.(check bool) "int-int compat" true
    (Modes.compatible (t_int 4) (t_int 1))

let test_compatible_tuple_tuple () =
  Alcotest.(check bool) "tuple compat" true
    (Modes.compatible
       (t_tup [t_int 2; t_int 3])
       (t_tup [t_int 1; t_int 6]))

let test_compatible_nested () =
  Alcotest.(check bool) "nested compat" true
    (Modes.compatible
       (t_tup [t_int 2; t_tup [t_int 3; t_int 4]])
       (t_tup [t_int 1; t_tup [t_int 2; t_int 8]]))

let test_incompatible_int_tuple () =
  Alcotest.(check bool) "int vs tuple" false
    (Modes.compatible (t_int 4) (t_tup [t_int 2; t_int 2]))

let test_incompatible_depth () =
  Alcotest.(check bool) "depth mismatch" false
    (Modes.compatible
       (t_tup [t_int 2; t_int 3])
       (t_tup [t_int 2; t_tup [t_int 1; t_int 3]]))

(* --- pp --- *)
let test_pp_int () =
  let s = Format.asprintf "%a" Modes.pp (t_int 4) in
  Alcotest.(check string) "pp int" "4" s

let test_pp_flat_tuple () =
  let s = Format.asprintf "%a" Modes.pp (t_tup [t_int 2; t_int 3]) in
  Alcotest.(check string) "pp flat" "(2, 3)" s

let test_pp_nested () =
  let s = Format.asprintf "%a" Modes.pp
    (t_tup [t_int 2; t_tup [t_int 3; t_int 4]]) in
  Alcotest.(check string) "pp nested" "(2, (3, 4))" s

(* --- runner --- *)
let () =
  Alcotest.run "Mode" [
    "size",  [ Alcotest.test_case "leaf"   `Quick test_size_leaf
             ; Alcotest.test_case "flat"   `Quick test_size_flat_tuple
             ; Alcotest.test_case "nested" `Quick test_size_nested ];
    "depth", [ Alcotest.test_case "leaf"   `Quick test_depth_leaf
             ; Alcotest.test_case "flat"   `Quick test_depth_flat
             ; Alcotest.test_case "nested" `Quick test_depth_nested ];
    "rank",  [ Alcotest.test_case "leaf"   `Quick test_rank_leaf
             ; Alcotest.test_case "tuple"  `Quick test_rank_tuple ];
    "flatten",[ Alcotest.test_case "leaf"   `Quick test_flatten_leaf
              ; Alcotest.test_case "nested" `Quick test_flatten_nested ];
    "compatible", [ Alcotest.test_case "int-int"    `Quick test_compatible_int_int
                  ; Alcotest.test_case "tuple"      `Quick test_compatible_tuple_tuple
                  ; Alcotest.test_case "nested"     `Quick test_compatible_nested
                  ; Alcotest.test_case "int-tuple"  `Quick test_incompatible_int_tuple
                  ; Alcotest.test_case "depth-mis"  `Quick test_incompatible_depth ];
    "pp",    [ Alcotest.test_case "int"    `Quick test_pp_int
             ; Alcotest.test_case "flat"   `Quick test_pp_flat_tuple
             ; Alcotest.test_case "nested" `Quick test_pp_nested ];
  ]
