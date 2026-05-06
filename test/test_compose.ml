open Tesserae

let i n    = Modes.Int n
let tup ts = Modes.Tuple ts
let lay s d = Layout.make s d

(* --- compose --- *)

let test_compose_identity () =
  (* composing with identity layout (shape=n, stride=1) is a no-op *)
  let outer = lay (i 8) (i 1) in
  let inner = lay (i 8) (i 1) in
  let c = Compose.compose outer inner in
  Alcotest.(check int) "size" 8 (Layout.size c);
  Alcotest.(check int) "idx 0" 0 (Layout.idx c (i 0));
  Alcotest.(check int) "idx 3" 3 (Layout.idx c (i 3))

let test_compose_stride () =
  (* outer: shape=8, stride=2  (picks every other element)
     inner: shape=4, stride=1
     result: shape=4, stride=2 *)
  let outer = lay (i 8) (i 2) in
  let inner = lay (i 4) (i 1) in
  let c = Compose.compose outer inner in
  Alcotest.(check int) "size"  4 (Layout.size c);
  Alcotest.(check int) "idx 0" 0 (Layout.idx c (i 0));
  Alcotest.(check int) "idx 3" 6 (Layout.idx c (i 3))

let test_compose_flat () =
  (* outer: shape=(4,2), stride=(1,4)  — col-major 4x2
     inner: shape=(2,2), stride=(1,2)
     composing selects a sub-tile *)
  let outer = lay (tup [i 4; i 2]) (tup [i 1; i 4]) in
  let inner = lay (tup [i 2; i 2]) (tup [i 1; i 2]) in
  Format.printf "outer shape flat: %s\n"
    (String.concat ", " (List.map string_of_int (Modes.flatten outer.shape)));
  Format.printf "outer stride flat: %s\n"
    (String.concat ", " (List.map string_of_int (Modes.flatten outer.stride)));
  Format.printf "outer size: %d\n" (Layout.size outer);
  Format.printf "outer cosize: %d\n" (Layout.cosize outer);
  Format.printf "inner shape flat: %s\n"
    (String.concat ", " (List.map string_of_int (Modes.flatten inner.shape)));
  Format.printf "inner stride flat: %s\n"
    (String.concat ", " (List.map string_of_int (Modes.flatten inner.stride)));
  Format.printf "outer idx (0,0) = %d\n" (Layout.idx outer (tup [i 0; i 0]));
  Format.printf "outer idx (1,0) = %d\n" (Layout.idx outer (tup [i 1; i 0]));
  Format.printf "outer idx (0,1) = %d\n" (Layout.idx outer (tup [i 0; i 1]));
  Format.printf "outer idx (1,1) = %d\n" (Layout.idx outer (tup [i 1; i 1]));
  Format.printf "outer idx (2,0) = %d\n" (Layout.idx outer (tup [i 2; i 0]));

  (* what should composed stride leaves map to? *)
  Format.printf "apply_outer(1) should be = %d\n" (Layout.idx outer (tup [i 1; i 0]));
  Format.printf "apply_outer(2) should be = %d\n" (Layout.idx outer (tup [i 0; i 1]));
  let c = Compose.compose outer inner in
  Format.printf "Composed Stride: %a\n" Modes.pp c.stride;
  Stdlib.Format.printf "Inner Cosize: %d, Outer Size: %d\n" (Layout.cosize inner) (Layout.size outer);
  Alcotest.(check int) "size" 4 (Layout.size c);
  Alcotest.(check int) "idx (0,0)" 0 (Layout.idx c (tup [i 0; i 0]));
  Alcotest.(check int) "idx (1,1)" 5 (Layout.idx c (tup [i 1; i 1]))

let test_compose_invalid () =
  let outer = lay (i 4) (i 1) in
  let inner = lay (i 8) (i 1) in
  Alcotest.check_raises "cosize too large" (Invalid_argument "inner cosize exceeds outer size") (fun () ->
    ignore (Compose.compose outer inner))

(* --- tile --- *)

let test_tile_simple () =
  (* layout: shape=8, stride=1
     tile:   shape=4
     result: shape=(4,2), stride=(1,4) *)
  let l = lay (i 8) (i 1) in
  let c = Compose.tile l (i 4) in
  Alcotest.(check int) "size" 8 (Layout.size c);
  Alcotest.(check int) "cosize" 8 (Layout.cosize c);
  Alcotest.(check int) "idx (0,0)" 0 (Layout.idx c (tup [i 0; i 0]));
  Alcotest.(check int) "idx (3,1)" 7 (Layout.idx c (tup [i 3; i 1]))

let test_tile_2d () =
  (* layout: shape=(4,4), stride=(1,4)  — col-major 4x4
     tile:   shape=(2,2)
     result: shape=((2,2),(2,2)), stride=((1,4),(2,8)) *)
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let c = Compose.tile l (tup [i 2; i 2]) in
  Alcotest.(check int) "size" 16 (Layout.size c);
  Alcotest.(check int) "idx (0,0),(0,0)" 0
    (Layout.idx c (tup [tup [i 0; i 0]; tup [i 0; i 0]]));
  Alcotest.(check int) "idx (1,1),(1,1)" 15
    (Layout.idx c (tup [tup [i 1; i 1]; tup [i 1; i 1]]))

let test_tile_invalid () =
  let l = lay (i 6) (i 1) in
  Alcotest.check_raises "indivisible" (Invalid_argument "tile shape does not divide layout shape") (fun () ->
    ignore (Compose.tile l (i 4)))

(* --- runner --- *)
let () =
  Alcotest.run "Compose" [
    "compose", [ Alcotest.test_case "identity" `Quick test_compose_identity
               ; Alcotest.test_case "stride"   `Quick test_compose_stride
               ; Alcotest.test_case "flat"     `Quick test_compose_flat
               ; Alcotest.test_case "invalid"  `Quick test_compose_invalid ];
    "tile",    [ Alcotest.test_case "simple"   `Quick test_tile_simple
               ; Alcotest.test_case "2d"       `Quick test_tile_2d
               ; Alcotest.test_case "invalid"  `Quick test_tile_invalid ];
  ]
