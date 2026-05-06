open Tesserae

let i n    = Modes.Int n
let tup ts = Modes.Tuple ts
let lay s d = Layout.make s d

(* ------------------------------------------------------------------ *)
(* flat_divide                                                         *)
(* ------------------------------------------------------------------ *)

let test_flat_divide_simple () =
  (* layout: 8:1, divisor: 4
     result: (4,2):(1,4) — 4 within, 2 across *)
  let l = lay (i 8) (i 1) in
  let r = Algebra.flat_divide l 4 in
  let shapes  = Modes.flatten r.Layout.shape  in
  let strides = Modes.flatten r.Layout.stride in
  Alcotest.(check (list int)) "shapes"  [4; 2] shapes;
  Alcotest.(check (list int)) "strides" [1; 4] strides

let test_flat_divide_strided () =
  (* layout: 8:2, divisor: 4
     result: (4,2):(2,8) *)
  let l = lay (i 8) (i 2) in
  let r = Algebra.flat_divide l 4 in
  let shapes  = Modes.flatten r.Layout.shape  in
  let strides = Modes.flatten r.Layout.stride in
  Alcotest.(check (list int)) "shapes"  [4; 2] shapes;
  Alcotest.(check (list int)) "strides" [2; 8] strides

let test_flat_divide_invalid () =
  let l = lay (i 8) (i 1) in
  Alcotest.check_raises "indivisible"
    (Invalid_argument "divisor does not divide layout size")
    (fun () -> ignore (Algebra.flat_divide l 3))

(* ------------------------------------------------------------------ *)
(* logical_divide                                                      *)
(* ------------------------------------------------------------------ *)

let test_logical_divide_1d () =
  (* layout: 8:1, tile: 4:1
     result shape: (4,2), strides: (1,4) *)
  let l    = lay (i 8) (i 1) in
  let tile = lay (i 4) (i 1) in
  let r    = Algebra.logical_divide l tile in
  Alcotest.(check int) "size" 8 (Layout.size r);
  let shapes  = Modes.flatten r.Layout.shape  in
  let strides = Modes.flatten r.Layout.stride in
  Alcotest.(check (list int)) "shapes"  [4; 2] shapes;
  Alcotest.(check (list int)) "strides" [1; 4] strides

let test_logical_divide_2d () =
  (* layout: (4,4):(1,4)  col-major 4x4
     tile:   (2,2):(1,2)  col-major 2x2
     result: shape=((2,2),(2,2)) stride=((1,2),(4,8)) *)
  let l    = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let tile = lay (tup [i 2; i 2]) (tup [i 1; i 2]) in
  let r    = Algebra.logical_divide l tile in
  Alcotest.(check int) "size" 16 (Layout.size r);
  Alcotest.(check int) "rank"  2 (Layout.rank r);
  (* within-tile index (1,1) in first mode *)
  Alcotest.(check int) "within (1,1)" 5
    (Layout.idx r (tup [tup [i 1; i 1]; tup [i 0; i 0]]));
  Alcotest.(check int) "across (1,1)" 10
    (Layout.idx r (tup [tup [i 0; i 0]; tup [i 1; i 1]]))

let test_logical_divide_invalid () =
  let l    = lay (i 6) (i 1) in
  let tile = lay (i 4) (i 1) in
  Alcotest.check_raises "indivisible"
    (Invalid_argument "tile does not divide layout")
    (fun () -> ignore (Algebra.logical_divide l tile))

(* ------------------------------------------------------------------ *)
(* zipped_divide                                                       *)
(* ------------------------------------------------------------------ *)

let test_zipped_divide_2d () =
  (* layout: (4,4):(1,4)
     tile:   (2,2):(1,2)
     zipped result shape: ((2,2),(2,2))
     but with interleaved within/across per original mode *)
  let l    = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let tile = lay (tup [i 2; i 2]) (tup [i 1; i 2]) in
  let r    = Algebra.zipped_divide l tile in
  Alcotest.(check int) "size" 16 (Layout.size r);
  Alcotest.(check int) "rank"  2 (Layout.rank r)

let test_zipped_divide_idx () =
  (* zipped_divide should produce same indices as logical_divide
     just with different coordinate structure *)
  let l    = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let tile = lay (tup [i 2; i 2]) (tup [i 1; i 2]) in
  let ld   = Algebra.logical_divide l tile in
  let zd   = Algebra.zipped_divide  l tile in
  (* collect all indices from both and check they're the same set *)
  let collect_indices layout =
    let n = Layout.size layout in
    let idxs = ref [] in
    for k = 0 to n - 1 do
      let shapes = Modes.flatten layout.Layout.shape in
      let coords =
        let (_, digits) =
          List.fold_left
            (fun (rem, acc) sh -> (rem / sh, acc @ [rem mod sh]))
            (k, []) shapes
        in
        Modes.Tuple (List.map (fun n -> Modes.Int n) digits)
      in
      idxs := Layout.idx layout coords :: !idxs
    done;
    List.sort compare !idxs
  in
  let ld_idxs = collect_indices ld in
  let zd_idxs = collect_indices zd in
  Alcotest.(check (list int)) "same index set" ld_idxs zd_idxs

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Divide" [
    "flat_divide",    [ Alcotest.test_case "simple"   `Quick test_flat_divide_simple
                      ; Alcotest.test_case "strided"  `Quick test_flat_divide_strided
                      ; Alcotest.test_case "invalid"  `Quick test_flat_divide_invalid ];
    "logical_divide", [ Alcotest.test_case "1d"       `Quick test_logical_divide_1d
                      ; Alcotest.test_case "2d"       `Quick test_logical_divide_2d
                      ; Alcotest.test_case "invalid"  `Quick test_logical_divide_invalid ];
    "zipped_divide",  [ Alcotest.test_case "2d"       `Quick test_zipped_divide_2d
                      ; Alcotest.test_case "idx"      `Quick test_zipped_divide_idx ];
  ]
