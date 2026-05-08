open Tesserae

let cluster_2sm () =
  Cluster.make
    { Cluster.x = 2; y = 1; z = 1 } 6
    [ (0, Cluster.Producer); (1, Cluster.Consumer)
    ; (2, Cluster.Epilogue); (3, Cluster.Epilogue)
    ; (4, Cluster.Epilogue); (5, Cluster.Scheduler) ]

let cluster_1sm () =
  Cluster.make
    { Cluster.x = 1; y = 1; z = 1 } 4
    [ (0, Cluster.Producer); (1, Cluster.Consumer)
    ; (2, Cluster.Epilogue); (3, Cluster.Epilogue) ]

let sw128 () = Swizzle.make 3 4 3

(* BM=128, BN=256, BK=64, bf16 *)
let pipe () = Pipeline.make 4 ((128 + 128) * 64 * 2)

let tma_2sm () =
  Tile_io.make Tile_io.TmaMulticast Elemtype.Bfloat16
    (cluster_2sm ()) (pipe ()) (sw128 ()) 128 256 64

let tma_1sm () =
  Tile_io.make Tile_io.TmaLoad Elemtype.Bfloat16
    (cluster_1sm ()) (pipe ()) (sw128 ()) 128 256 64

let cp_async () =
  Tile_io.make Tile_io.CpAsync Elemtype.Float16
    (cluster_1sm ()) (pipe ()) (sw128 ()) 128 256 64

(* ------------------------------------------------------------------ *)
(* make / is_tma / requires_mbar                                       *)
(* ------------------------------------------------------------------ *)

let test_is_tma_true () =
  Alcotest.(check bool) "tma" true (Tile_io.is_tma (tma_2sm ()))

let test_is_tma_false () =
  Alcotest.(check bool) "cp.async" false (Tile_io.is_tma (cp_async ()))

let test_requires_mbar_tma () =
  Alcotest.(check bool) "tma mbar" true (Tile_io.requires_mbar (tma_2sm ()))

let test_requires_mbar_cp () =
  Alcotest.(check bool) "cp mbar" false (Tile_io.requires_mbar (cp_async ()))

(* ------------------------------------------------------------------ *)
(* bytes_per_load                                                      *)
(* ------------------------------------------------------------------ *)

let test_bytes_a_tma () =
  (* BM=128, BK=64, bf16: 128*64*2 = 16384 bytes *)
  Alcotest.(check int) "a bytes" 16384
    (Tile_io.bytes_per_load_a (tma_2sm ()) 128 64)

let test_bytes_b_tma_2sm () =
  (* BN=256, BK=64, bf16, 2SM → each CTA loads BN/2=128: 128*64*2 = 16384 *)
  Alcotest.(check int) "b bytes 2sm" 16384
    (Tile_io.bytes_per_load_b (tma_2sm ()) 256 64 Tmem.CTA2)

let test_bytes_b_tma_1sm () =
  (* BN=256, BK=64, bf16, 1SM → full B: 256*64*2 = 32768 *)
  Alcotest.(check int) "b bytes 1sm" 32768
    (Tile_io.bytes_per_load_b (tma_1sm ()) 256 64 Tmem.CTA1)

(* ------------------------------------------------------------------ *)
(* emit_mbar_expect                                                    *)
(* ------------------------------------------------------------------ *)

let test_mbar_expect_2sm () =
  (* 2SM: A + B/2 = (128+128)*64*2 = 32768 bytes *)
  let t   = tma_2sm () in
  let s   = Tile_io.emit_mbar_expect t "mbar" 128 256 64 in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has 32768" true (contains "32768" s);
  Alcotest.(check bool) "has mbar"  true (contains "mbar" s)

let test_mbar_expect_1sm () =
  (* 1SM: A + B = (128+256)*64*2 = 49152 bytes *)
  let t   = tma_1sm () in
  let s   = Tile_io.emit_mbar_expect t "mbar" 128 256 64 in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has 49152" true (contains "49152" s)

(* ------------------------------------------------------------------ *)
(* emit_tma_load                                                       *)
(* ------------------------------------------------------------------ *)

let test_emit_tma_load_a () =
  let t = tma_2sm () in
  let s = Tile_io.emit_tma_load_a t "A_smem" "A_tmap" "mbar" "row" "col" "k" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has tma" true (contains "tma" s);
  Alcotest.(check bool) "has A_smem" true (contains "A_smem" s)

let test_emit_cp_async () =
  let t = cp_async () in
  let s = Tile_io.emit_cp_async_load t "smem" "gmem_ptr" "offset" "mbar" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has cp.async" true (contains "cp.async" s)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Tile_io" [
    "flags",   [ Alcotest.test_case "tma-true"  `Quick test_is_tma_true
               ; Alcotest.test_case "tma-false" `Quick test_is_tma_false
               ; Alcotest.test_case "mbar-tma"  `Quick test_requires_mbar_tma
               ; Alcotest.test_case "mbar-cp"   `Quick test_requires_mbar_cp ];
    "bytes",   [ Alcotest.test_case "a-tma"     `Quick test_bytes_a_tma
               ; Alcotest.test_case "b-2sm"     `Quick test_bytes_b_tma_2sm
               ; Alcotest.test_case "b-1sm"     `Quick test_bytes_b_tma_1sm ];
    "mbar",    [ Alcotest.test_case "2sm"       `Quick test_mbar_expect_2sm
               ; Alcotest.test_case "1sm"       `Quick test_mbar_expect_1sm ];
    "emit",    [ Alcotest.test_case "tma-a"     `Quick test_emit_tma_load_a
               ; Alcotest.test_case "cp-async"  `Quick test_emit_cp_async ];
  ]
