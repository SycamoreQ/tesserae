open Tesserae

let i n    = Modes.Int n
let tup ts = Modes.Tuple ts
let lay s d = Layout.make s d

(* --- make / accessors --- *)
let test_make_global () =
  let l = lay (i 8) (i 1) in
  let t : (float, _) Tensor.t = Tensor.make l Memspace.Global in
  Alcotest.(check int) "size" 8 (Tensor.size t);
  Alcotest.(check int) "rank" 1 (Tensor.rank t)

let test_make_shared () =
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let t : (float, _) Tensor.t = Tensor.make l Memspace.Shared in
  Alcotest.(check int) "size" 16 (Tensor.size t);
  Alcotest.(check int) "rank" 2 (Tensor.rank t)

(* --- local_tile --- *)
let test_local_tile_simple () =
  let l = lay (i 8) (i 1) in
  let t : (float, _) Tensor.t = Tensor.make l Memspace.Global in
  let tiled = Tensor.local_tile t (i 4) in
  Alcotest.(check int) "tiled size" 8 (Tensor.size tiled);
  Alcotest.(check int) "tiled rank" 2 (Tensor.rank tiled)

let test_local_tile_2d () =
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let t : (float, _) Tensor.t = Tensor.make l Memspace.Shared in
  let tiled = Tensor.local_tile t (tup [i 2; i 2]) in
  Alcotest.(check int) "tiled size" 16 (Tensor.size tiled);
  Alcotest.(check int) "tiled rank" 2 (Tensor.rank tiled)

(* --- transfer --- *)
let test_transfer_valid () =
  let l = lay (i 8) (i 1) in
  let src : (float, _) Tensor.t = Tensor.make l Memspace.Global in
  let dst = Tensor.transfer ~src ~dst_space:Memspace.Shared in
  Alcotest.(check int) "dst size" 8 (Tensor.size dst);
  Alcotest.(check string) "dst space" "shared"
    (Memspace.name (Tensor.space dst))

let test_transfer_invalid () =
  let l = lay (i 8) (i 1) in
  let src : (float, _) Tensor.t = Tensor.make l Memspace.Shared in
  Alcotest.check_raises "shared->global invalid"
    (Invalid_argument "invalid transfer between memory spaces")
    (fun () -> ignore (Tensor.transfer ~src ~dst_space:Memspace.Global))

(* --- pp --- *)
let test_pp () =
  let l = lay (tup [i 2; i 3]) (tup [i 1; i 2]) in
  let t : (float, _) Tensor.t = Tensor.make l Memspace.Register in
  let s = Stdlib.Format.asprintf "%a" Tensor.pp t in
  Alcotest.(check string) "pp" "register:(2, 3):(1, 2)" s

(* --- runner --- *)
let () =
  Alcotest.run "Tensor" [
    "make",       [ Alcotest.test_case "global" `Quick test_make_global
                  ; Alcotest.test_case "shared" `Quick test_make_shared ];
    "local_tile", [ Alcotest.test_case "simple" `Quick test_local_tile_simple
                  ; Alcotest.test_case "2d"     `Quick test_local_tile_2d ];
    "transfer",   [ Alcotest.test_case "valid"   `Quick test_transfer_valid
                  ; Alcotest.test_case "invalid" `Quick test_transfer_invalid ];
    "pp",         [ Alcotest.test_case "basic"   `Quick test_pp ];
  ]
