open Tesserae

let i n    = Modes.Int n
let tup ts = Modes.Tuple ts
let lay s d = Layout.make s d

let test_sort_already_sorted () =
  let l = lay (tup [i 4; i 2]) (tup [i 1; i 4]) in
  let s = Algebra.sort l in
  Alcotest.(check int) "size" 8 (Layout.size s);
  let strides = Modes.flatten s.Layout.stride in
  Alcotest.(check (list int)) "strides" [1; 4] strides

let test_sort_reverses () =
  let l = lay (tup [i 2; i 4]) (tup [i 4; i 1]) in
  let s = Algebra.sort l in
  let strides = Modes.flatten s.Layout.stride in
  Alcotest.(check (list int)) "strides sorted" [1; 4] strides;
  let shapes = Modes.flatten s.Layout.shape in
  Alcotest.(check (list int)) "shapes reordered" [4; 2] shapes

let test_sort_three_modes () =
  let l = lay (tup [i 2; i 3; i 4]) (tup [i 12; i 1; i 3]) in
  let s = Algebra.sort l in
  let strides = Modes.flatten s.Layout.stride in
  Alcotest.(check (list int)) "strides" [1; 3; 12] strides;
  let shapes = Modes.flatten s.Layout.shape in
  Alcotest.(check (list int)) "shapes" [3; 4; 2] shapes

let test_coalesce_contiguous () =
  let l = lay (tup [i 2; i 4]) (tup [i 1; i 2]) in
  let c = Algebra.coalesce l in
  Alcotest.(check int) "size" 8  (Layout.size c);
  Alcotest.(check int) "rank" 1  (Layout.rank c);
  let shapes  = Modes.flatten c.Layout.shape  in
  let strides = Modes.flatten c.Layout.stride in
  Alcotest.(check (list int)) "shape"  [8] shapes;
  Alcotest.(check (list int)) "stride" [1] strides

let test_coalesce_noncontiguous () =
  let l = lay (tup [i 2; i 4]) (tup [i 1; i 4]) in
  let c = Algebra.coalesce l in
  Alcotest.(check int) "rank" 2 (Layout.rank c);
  let strides = Modes.flatten c.Layout.stride in
  Alcotest.(check (list int)) "strides" [1; 4] strides

let test_coalesce_drop_size_one () =
  let l = lay
    (tup [i 1; i 4; i 1; i 2])
    (tup [i 0; i 1; i 99; i 4]) in
  let c = Algebra.coalesce l in
  Alcotest.(check int) "rank" 2 (Layout.rank c);
  let shapes  = Modes.flatten c.Layout.shape  in
  let strides = Modes.flatten c.Layout.stride in
  Alcotest.(check (list int)) "shapes"  [4; 2] shapes;
  Alcotest.(check (list int)) "strides" [1; 4] strides

let test_coalesce_three_contiguous () =
  let l = lay (tup [i 2; i 3; i 4]) (tup [i 1; i 2; i 6]) in
  let c = Algebra.coalesce l in
  Alcotest.(check int) "rank" 1 (Layout.rank c);
  let shapes = Modes.flatten c.Layout.shape in
  Alcotest.(check (list int)) "shape" [24] shapes

let test_admissible_simple () =
  Alcotest.(check bool) "simple" true
    (Algebra.is_admissible (lay (i 4) (i 1)) 8)

let test_admissible_two_modes () =
  Alcotest.(check bool) "two modes" true
    (Algebra.is_admissible (lay (tup [i 2; i 4]) (tup [i 1; i 2])) 8)

let test_admissible_false_gap () =
  Alcotest.(check bool) "gap" false
    (Algebra.is_admissible (lay (tup [i 2; i 4]) (tup [i 1; i 4])) 8)

let test_admissible_false_m () =
  Alcotest.(check bool) "bad m" false
    (Algebra.is_admissible (lay (i 4) (i 1)) 6)

let test_complement_simple () =
  let a = lay (i 4) (i 1) in
  let b = Algebra.complement a 8 in
  Alcotest.(check int) "size" 2 (Layout.size b);
  let shapes  = Modes.flatten b.Layout.shape  in
  let strides = Modes.flatten b.Layout.stride in
  Alcotest.(check (list int)) "shapes"  [1; 2] shapes;
  Alcotest.(check (list int)) "strides" [1; 4] strides

let test_complement_col_major () =
  let a = lay (tup [i 2; i 4]) (tup [i 1; i 2]) in
  let b = Algebra.complement a 16 in
  Alcotest.(check int) "size" 2 (Layout.size b);
  let shapes  = Modes.flatten b.Layout.shape  in
  let strides = Modes.flatten b.Layout.stride in
  Alcotest.(check (list int)) "shapes"  [1; 1; 2] shapes;
  Alcotest.(check (list int)) "strides" [1; 2; 8] strides

let test_complement_invalid () =
  Alcotest.check_raises "not admissible"
    (Invalid_argument "layout is not admissible for complementation")
    (fun () ->
       ignore (Algebra.complement
         (lay (tup [i 2; i 4]) (tup [i 1; i 4])) 8))

let test_complement_bijection () =
  let hits = Array.make 8 0 in
  let ab = lay (tup [i 4; i 2]) (tup [i 1; i 4]) in
  for k = 0 to 7 do
    let v = Layout.idx ab (tup [i (k mod 4); i (k / 4)]) in
    hits.(v) <- hits.(v) + 1
  done;
  let all_one = Array.for_all (fun x -> x = 1) hits in
  Alcotest.(check bool) "bijection" true all_one

let () =
  Alcotest.run "Algebra" [
    "sort",      [ Alcotest.test_case "sorted"    `Quick test_sort_already_sorted
                 ; Alcotest.test_case "reverses"  `Quick test_sort_reverses
                 ; Alcotest.test_case "three"     `Quick test_sort_three_modes ];
    "coalesce",  [ Alcotest.test_case "contig"    `Quick test_coalesce_contiguous
                 ; Alcotest.test_case "noncontig" `Quick test_coalesce_noncontiguous
                 ; Alcotest.test_case "size-one"  `Quick test_coalesce_drop_size_one
                 ; Alcotest.test_case "three"     `Quick test_coalesce_three_contiguous ];
    "admissible",[ Alcotest.test_case "simple"    `Quick test_admissible_simple
                 ; Alcotest.test_case "two-modes" `Quick test_admissible_two_modes
                 ; Alcotest.test_case "gap"       `Quick test_admissible_false_gap
                 ; Alcotest.test_case "bad-m"     `Quick test_admissible_false_m ];
    "complement",[ Alcotest.test_case "simple"    `Quick test_complement_simple
                 ; Alcotest.test_case "col-major" `Quick test_complement_col_major
                 ; Alcotest.test_case "invalid"   `Quick test_complement_invalid
                 ; Alcotest.test_case "bijection" `Quick test_complement_bijection ];
  ]
