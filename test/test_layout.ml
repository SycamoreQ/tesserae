open Tesserae

(* helpers *)
let int n    = Modes.Int n
let tup ts   = Modes.Tuple ts
let layout s d = Layout.make s d

(* --- make / invalid --- *)
let test_make_valid () =
  let l = layout (int 4) (int 1) in
  Alcotest.(check int) "size" 4 (Layout.size l)

let test_make_invalid () =
  let exn = Invalid_argument "incompatible shape and stride" in
  Alcotest.check_raises "int vs tuple" exn (fun () ->
    ignore (layout (int 4) (tup [int 1; int 4])))

(* --- size --- *)
let test_size_flat () =
  let l = layout (tup [int 2; int 3]) (tup [int 1; int 2]) in
  Alcotest.(check int) "size 6" 6 (Layout.size l)

let test_size_nested () =
  let l = layout
    (tup [int 2; tup [int 3; int 4]])
    (tup [int 1; tup [int 2; int 6]]) in
  Alcotest.(check int) "size 24" 24 (Layout.size l)

(* --- rank --- *)
let test_rank_int () =
  Alcotest.(check int) "rank 1" 1 (Layout.rank (layout (int 8) (int 1)))

let test_rank_tuple () =
  Alcotest.(check int) "rank 2" 2
    (Layout.rank (layout (tup [int 2; int 4]) (tup [int 1; int 2])))

(* --- cosize --- *)
let test_cosize_simple () =
  (* shape=4 stride=1 → (4-1)*1 + 1 = 4 *)
  Alcotest.(check int) "cosize 4" 4
    (Layout.cosize (layout (int 4) (int 1)))

let test_cosize_col_major () =
  (* shape=(2,3) stride=(1,2) -> (2-1)*1 + (3-1)*2 + 1 = 1 + 4 + 1 = 6 *)
  Alcotest.(check int) "cosize 6" 6
    (Layout.cosize (layout (tup [int 2; int 3]) (tup [int 1; int 2])))

let test_cosize_noncontiguous () =
  (* shape=(2,3) stride=(1,4) -> (2-1)*1 + (3-1)*4 + 1 = 1 + 8 + 1 = 10 *)
  Alcotest.(check int) "cosize 10" 10
    (Layout.cosize (layout (tup [int 2; int 3]) (tup [int 1; int 4])))

(* --- idx --- *)
let test_idx_scalar () =
  let l = layout (int 4) (int 1) in
  Alcotest.(check int) "idx 3" 3 (Layout.idx l (int 3))

let test_idx_flat () =
  (* col-major (2,3) stride (1,2): coord (1,2) → 1*1 + 2*2 = 5 *)
  let l = layout (tup [int 2; int 3]) (tup [int 1; int 2]) in
  Alcotest.(check int) "idx 5" 5 (Layout.idx l (tup [int 1; int 2]))

let test_idx_nested () =
  (* shape (2,(3,4)) stride (1,(2,6)) coord (1,(2,3)) → 1 + 2*2 + 3*6 = 23 *)
  let l = layout
    (tup [int 2; tup [int 3; int 4]])
    (tup [int 1; tup [int 2; int 6]]) in
  Alcotest.(check int) "idx 23" 23
    (Layout.idx l (tup [int 1; tup [int 2; int 3]]))

(* --- pp --- *)
let test_pp () =
  let l = layout (tup [int 2; int 3]) (tup [int 1; int 2]) in
  let s = Stdlib.Format.asprintf "%a" Layout.pp l in
  Alcotest.(check string) "pp" "(2, 3):(1, 2)" s

(* --- runner --- *)
let () =
  Alcotest.run "Layout" [
    "make",   [ Alcotest.test_case "valid"   `Quick test_make_valid
              ; Alcotest.test_case "invalid" `Quick test_make_invalid ];
    "size",   [ Alcotest.test_case "flat"    `Quick test_size_flat
              ; Alcotest.test_case "nested"  `Quick test_size_nested ];
    "rank",   [ Alcotest.test_case "int"     `Quick test_rank_int
              ; Alcotest.test_case "tuple"   `Quick test_rank_tuple ];
    "cosize", [ Alcotest.test_case "simple"  `Quick test_cosize_simple
              ; Alcotest.test_case "colmaj"  `Quick test_cosize_col_major
              ; Alcotest.test_case "noncont" `Quick test_cosize_noncontiguous ];
    "idx",    [ Alcotest.test_case "scalar"  `Quick test_idx_scalar
              ; Alcotest.test_case "flat"    `Quick test_idx_flat
              ; Alcotest.test_case "nested"  `Quick test_idx_nested ];
    "pp",     [ Alcotest.test_case "basic"   `Quick test_pp ];
  ]
