open Tesserae

let i n    = Modes.Int n
let tup ts = Modes.Tuple ts
let lay s d = Layout.make s d

(* A standard Ampere single-warp TiledMMA:
   atom  = SM80_16x8x16_F32F16F16F32_TN
   threads arranged as 32 threads in a flat warp
   warps = 1 warp in M, 1 in N
   tile  = (16, 8) — matches atom shape exactly *)
let single_warp_mma () =
  Tiled_mma.make
    (Mma_atom.sm80_16x8x16_f32f16f16f32
       Mma_atom.ColMajor Mma_atom.RowMajor)
    (lay (i 32) (i 1))
    (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
    (tup [i 16; i 8])

(* A 4-warp Ampere TiledMMA:
   atom  = SM80_16x8x16_F32F16F16F32_TN
   threads = 32 per warp (flat)
   warps = (4,1) — 4 warps in M direction
   tile  = (64, 8) — 4 * atom_m, 1 * atom_n *)
let four_warp_mma () =
  Tiled_mma.make
    (Mma_atom.sm80_16x8x16_f32f16f16f32
       Mma_atom.ColMajor Mma_atom.RowMajor)
    (lay (i 32) (i 1))
    (lay (tup [i 4; i 1]) (tup [i 1; i 0]))
    (tup [i 64; i 8])

(* Hopper warpgroup TiledMMA:
   atom  = SM90_64x64x16_F32F16F16F32_TN
   threads = 128 (one warpgroup)
   warps = (1,1)
   tile  = (64, 64) *)
let hopper_wgmma () =
  Tiled_mma.make
    (Mma_atom.sm90_64x64x16_f32f16f16f32
       Mma_atom.ColMajor Mma_atom.RowMajor)
    (lay (i 128) (i 1))
    (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
    (tup [i 64; i 64])

(* ------------------------------------------------------------------ *)
(* make / validation                                                   *)
(* ------------------------------------------------------------------ *)

let test_make_valid () =
  let t = single_warp_mma () in
  Alcotest.(check int) "threads" 32 (Tiled_mma.thread_count t)

let test_make_invalid_threads () =
  Alcotest.check_raises "wrong thread count"
    (Invalid_argument "thread_layout size does not match atom thread count")
    (fun () ->
       ignore (Tiled_mma.make
         (Mma_atom.sm80_16x8x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor)
         (lay (i 64) (i 1))   (* wrong: 64 instead of 32 *)
         (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
         (tup [i 16; i 8])))

(* ------------------------------------------------------------------ *)
(* thread_count / warp_count                                           *)
(* ------------------------------------------------------------------ *)

let test_thread_count_single () =
  Alcotest.(check int) "32" 32
    (Tiled_mma.thread_count (single_warp_mma ()))

let test_thread_count_four () =
  Alcotest.(check int) "128" 128
    (Tiled_mma.thread_count (four_warp_mma ()))

let test_warp_count_single () =
  Alcotest.(check int) "1" 1
    (Tiled_mma.warp_count (single_warp_mma ()))

let test_warp_count_four () =
  Alcotest.(check int) "4" 4
    (Tiled_mma.warp_count (four_warp_mma ()))

(* ------------------------------------------------------------------ *)
(* tile_shape_mnk                                                      *)
(* ------------------------------------------------------------------ *)

let test_tile_mnk_single () =
  Alcotest.(check (triple int int int)) "16x8x16" (16, 8, 16)
    (Tiled_mma.tile_shape_mnk (single_warp_mma ()))

let test_tile_mnk_four () =
  Alcotest.(check (triple int int int)) "64x8x16" (64, 8, 16)
    (Tiled_mma.tile_shape_mnk (four_warp_mma ()))

let test_tile_mnk_hopper () =
  Alcotest.(check (triple int int int)) "64x64x16" (64, 64, 16)
    (Tiled_mma.tile_shape_mnk (hopper_wgmma ()))

(* ------------------------------------------------------------------ *)
(* partition_c                                                         *)
(* ------------------------------------------------------------------ *)

let test_partition_c_single () =
  (* atom 16x8, 32 threads, 1 warp
     each thread owns 16*8/32 = 4 elements of C
     laid out as (2,2):(1,2) in the fragment *)
  let t = single_warp_mma () in
  let c = Tiled_mma.partition_c t in
  Alcotest.(check int) "c size" 4 (Layout.size c)

let test_partition_c_four () =
  (* atom 16x8 * 4 warps in M = 64x8 tile, 128 threads
     each thread owns 64*8/128 = 4 elements *)
  let t = four_warp_mma () in
  let c = Tiled_mma.partition_c t in
  Alcotest.(check int) "c size" 4 (Layout.size c)

let test_partition_c_hopper () =
  (* atom 64x64, 128 threads
     each thread owns 64*64/128 = 32 elements *)
  let t = hopper_wgmma () in
  let c = Tiled_mma.partition_c t in
  Alcotest.(check int) "c size" 32 (Layout.size c)

(* ------------------------------------------------------------------ *)
(* emit_cpp                                                            *)
(* ------------------------------------------------------------------ *)

let test_emit_single_warp () =
  let t = single_warp_mma () in
  let s = Tiled_mma.emit_cpp t in
  (* must contain the atom string *)
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has atom"  true
    (contains "SM80_16x8x16_F32F16F16F32_TN" s);
  Alcotest.(check bool) "has TiledMMA" true
    (contains "TiledMMA" s)

let test_emit_hopper () =
  let t = hopper_wgmma () in
  let s = Tiled_mma.emit_cpp t in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has sm90 atom" true
    (contains "SM90_64x64x16_F32F16F16F32_TN" s)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Tiled_mma" [
    "make",       [ Alcotest.test_case "valid"   `Quick test_make_valid
                  ; Alcotest.test_case "invalid" `Quick test_make_invalid_threads ];
    "threads",    [ Alcotest.test_case "single"  `Quick test_thread_count_single
                  ; Alcotest.test_case "four"    `Quick test_thread_count_four ];
    "warps",      [ Alcotest.test_case "single"  `Quick test_warp_count_single
                  ; Alcotest.test_case "four"    `Quick test_warp_count_four ];
    "tile_mnk",   [ Alcotest.test_case "single"  `Quick test_tile_mnk_single
                  ; Alcotest.test_case "four"    `Quick test_tile_mnk_four
                  ; Alcotest.test_case "hopper"  `Quick test_tile_mnk_hopper ];
    "partition_c",[ Alcotest.test_case "single"  `Quick test_partition_c_single
                  ; Alcotest.test_case "four"    `Quick test_partition_c_four
                  ; Alcotest.test_case "hopper"  `Quick test_partition_c_hopper ];
    "emit_cpp",   [ Alcotest.test_case "single"  `Quick test_emit_single_warp
                  ; Alcotest.test_case "hopper"  `Quick test_emit_hopper ];
  ]
