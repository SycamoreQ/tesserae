open Tesserae

let i n    = Modes.Int n
let lay s d = Layout.make s d

(* Standard Ampere gmem to smem async copy:
   atom  = SM80_CP_ASYNC_CACHEGLOBAL float32 (16 bytes, 4 elems)
   threads  = 32 flat
   vals = 4 elements per thread (one 128-bit vector load) *)
let ampere_gmem_smem () =
  Tiled_copy.make
    (Copy_atom.sm80_cp_async_global Elemtype.Float32)
    (lay (i 32) (i 1))
    (lay (i 4)  (i 1))

(* Ampere float16 async copy:
   atom = SM80_CP_ASYNC_CACHEGLOBAL float16 (16 bytes, 8 elems)
   threads = 32 flat
   vals = 8 elements per thread *)
let ampere_f16_copy () =
  Tiled_copy.make
    (Copy_atom.sm80_cp_async_global Elemtype.Float16)
    (lay (i 32) (i 1))
    (lay (i 8)  (i 1))

(* Hopper TMA load:
   atom = SM90_TMA_LOAD float16 (128 bytes, 64 elems)
   threads = 128 (warpgroup, but TMA only uses 1 thread)
   vals = 64 elements per "thread" (TMA bulk) *)
let hopper_tma () =
  Tiled_copy.make
    (Copy_atom.sm90_tma_load Elemtype.Float16)
    (lay (i 128) (i 1))
    (lay (i 64)  (i 1))

(* Universal copy for testing partition *)
let universal_copy () =
  Tiled_copy.make
    (Copy_atom.universal Memspace.Global Memspace.Shared Elemtype.Float32)
    (lay (i 32) (i 1))
    (lay (i 1)  (i 1))

let test_make_valid () =
  let t = ampere_gmem_smem () in
  Alcotest.(check int) "threads" 32 (Tiled_copy.thread_count t)

let test_make_invalid_val () =
  Alcotest.check_raises "wrong val width"
    (Invalid_argument "val_layout size does not match atom vec_width")
    (fun () ->
       ignore (Tiled_copy.make
         (Copy_atom.sm80_cp_async_global Elemtype.Float32)
         (lay (i 32) (i 1))
         (lay (i 2) (i 1))))

let test_thread_count () =
  Alcotest.(check int) "32" 32
    (Tiled_copy.thread_count (ampere_gmem_smem ()))

let test_elements_per_thread_f32 () =
  Alcotest.(check int) "4" 4
    (Tiled_copy.elements_per_thread (ampere_gmem_smem ()))

let test_elements_per_thread_f16 () =
  Alcotest.(check int) "8" 8
    (Tiled_copy.elements_per_thread (ampere_f16_copy ()))

let test_tile_size_f32 () =
  (* 32 threads * 4 elems = 128 elements per instruction *)
  Alcotest.(check int) "128" 128
    (Tiled_copy.tile_size (ampere_gmem_smem ()))

let test_tile_size_f16 () =
  (* 32 threads * 8 elems = 256 elements per instruction *)
  Alcotest.(check int) "256" 256
    (Tiled_copy.tile_size (ampere_f16_copy ()))

let test_is_tma_false () =
  Alcotest.(check bool) "cp.async not tma" false
    (Tiled_copy.is_tma (ampere_gmem_smem ()))

let test_is_tma_true () =
  Alcotest.(check bool) "tma is tma" true
    (Tiled_copy.is_tma (hopper_tma ()))

let test_requires_mbar_false () =
  Alcotest.(check bool) "cp.async no mbar" false
    (Tiled_copy.requires_mbar (ampere_gmem_smem ()))

let test_requires_mbar_true () =
  Alcotest.(check bool) "tma needs mbar" true
    (Tiled_copy.requires_mbar (hopper_tma ()))

let test_partition_src_size () =
  (* universal copy, 32 threads over a 128-element layout
     each thread gets 128/32 = 4 elements *)
  let t  = universal_copy () in
  let l  = lay (i 128) (i 1) in
  let p  = Tiled_copy.partition_src t l in
  Alcotest.(check int) "per-thread 4" 4 (Layout.size p)

let test_partition_dst_size () =
  let t  = universal_copy () in
  let l  = lay (i 128) (i 1) in
  let p  = Tiled_copy.partition_dst t l in
  Alcotest.(check int) "per-thread 4" 4 (Layout.size p)

let test_emit_cp_async () =
  let t = ampere_gmem_smem () in
  let s = Tiled_copy.emit_cpp t in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has TiledCopy" true (contains "TiledCopy" s);
  Alcotest.(check bool) "has Copy_Atom" true (contains "Copy_Atom" s);
  Alcotest.(check bool) "has CP_ASYNC"  true (contains "CP_ASYNC" s)

let test_emit_tma () =
  let t = hopper_tma () in
  let s = Tiled_copy.emit_cpp t in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has TMA_LOAD" true (contains "TMA_LOAD" s)

let () =
  Alcotest.run "Tiled_copy" [
    "make",      [ Alcotest.test_case "valid"   `Quick test_make_valid
                 ; Alcotest.test_case "invalid" `Quick test_make_invalid_val ];
    "threads",   [ Alcotest.test_case "count"   `Quick test_thread_count ];
    "per_thread",[ Alcotest.test_case "f32"     `Quick test_elements_per_thread_f32
                 ; Alcotest.test_case "f16"     `Quick test_elements_per_thread_f16 ];
    "tile_size", [ Alcotest.test_case "f32"     `Quick test_tile_size_f32
                 ; Alcotest.test_case "f16"     `Quick test_tile_size_f16 ];
    "is_tma",    [ Alcotest.test_case "false"   `Quick test_is_tma_false
                 ; Alcotest.test_case "true"    `Quick test_is_tma_true ];
    "mbar",      [ Alcotest.test_case "false"   `Quick test_requires_mbar_false
                 ; Alcotest.test_case "true"    `Quick test_requires_mbar_true ];
    "partition", [ Alcotest.test_case "src"     `Quick test_partition_src_size
                 ; Alcotest.test_case "dst"     `Quick test_partition_dst_size ];
    "emit_cpp",  [ Alcotest.test_case "cp_async"`Quick test_emit_cp_async
                 ; Alcotest.test_case "tma"     `Quick test_emit_tma ];
  ]
