open Tesserae

let i n = Modes.Int n
let tup ts = Modes.Tuple ts
let lay s d = Layout.make s d

(* SM80 single-warp, register accumulators *)
let sm80_op () =
  let atom = Mma_atom.sm80_16x8x16_f32f16f16f32
               Mma_atom.ColMajor Mma_atom.RowMajor in
  let tmma = Tiled_mma.make atom
               (lay (i 32) (i 1))
               (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
               (tup [i 16; i 8]) in
  Tile_op.make tmma Tile_op.Registers ()

(* SM100 warpgroup, TMEM accumulators *)
let sm100_op () =
  let atom = Mma_atom.sm100_64x64x16_f32f16f16f32
               Mma_atom.ColMajor Mma_atom.RowMajor in
  let tmma = Tiled_mma.make atom
               (lay (i 128) (i 1))
               (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
               (tup [i 64; i 64]) in
  let tmem = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:64 ~num_rows:128 in
  Tile_op.make tmma Tile_op.TensorMem ~tmem ()

(* SM100 double-buffered TMEM *)
let sm100_double () =
  let atom = Mma_atom.sm100_128x128x16_f32f16f16f32
               Mma_atom.ColMajor Mma_atom.RowMajor in
  let tmma = Tiled_mma.make atom
               (lay (i 128) (i 1))
               (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
               (tup [i 128; i 128]) in
  let tmem = Tmem.double_buf_make ~cta_group:Tmem.CTA2 ~num_rows:128 in
  Tile_op.make tmma Tile_op.TensorMem ~tmem ~double_buf:true ()

let test_is_tmem_false () =
  Alcotest.(check bool) "sm80 not tmem" false (Tile_op.is_tmem (sm80_op ()))

let test_is_tmem_true () =
  Alcotest.(check bool) "sm100 tmem" true (Tile_op.is_tmem (sm100_op ()))

let test_is_wgmma_false () =
  Alcotest.(check bool) "sm80 not wgmma" false (Tile_op.is_wgmma (sm80_op ()))

let test_is_wgmma_true () =
  Alcotest.(check bool) "sm100 wgmma" true (Tile_op.is_wgmma (sm100_op ()))

let test_double_buf_false () =
  Alcotest.(check bool) "not double" false (sm100_op ()).Tile_op.double_buf

let test_double_buf_true () =
  Alcotest.(check bool) "double" true (sm100_double ()).Tile_op.double_buf

let test_accum_sm80 () =
  (* 16*8 / 32 threads = 4 *)
  Alcotest.(check int) "sm80 4" 4
    (Tile_op.accum_elems_per_thread (sm80_op ()))

let test_accum_sm100 () =
  (* 64*64 / 128 threads = 32 *)
  Alcotest.(check int) "sm100 32" 32
    (Tile_op.accum_elems_per_thread (sm100_op ()))

let test_accum_decl_registers () =
  let s = Tile_op.emit_accum_decl (sm80_op ()) "acc" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has float" true (contains "float" s);
  Alcotest.(check bool) "has acc"   true (contains "acc" s)

let test_accum_decl_tmem () =
  (* TMEM: no register declaration needed *)
  let s = Tile_op.emit_accum_decl (sm100_op ()) "acc" in
  Alcotest.(check string) "empty" "" s

let test_emit_mma_sm80 () =
  let s = Tile_op.emit_mma (sm80_op ()) "a_desc" "b_desc" false in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has mma" true (contains "mma" s)

let test_emit_mma_sm100 () =
  let s = Tile_op.emit_mma (sm100_op ()) "a_desc" "b_desc" false in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has tcgen05" true (contains "tcgen05" s)

let test_commit_sm80_empty () =
  let s = Tile_op.emit_commit (sm80_op ()) "mbar" 0 in
  Alcotest.(check string) "empty" "" s

let test_commit_sm100 () =
  let s = Tile_op.emit_commit (sm100_op ()) "mbar" 3 in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has commit" true (contains "commit" s);
  Alcotest.(check bool) "has mbar"   true (contains "mbar" s)

let test_alloc_registers_empty () =
  let s = Tile_op.emit_tmem_alloc (sm80_op ()) "tmem_addr" in
  Alcotest.(check string) "empty" "" s

let test_alloc_tmem () =
  let s = Tile_op.emit_tmem_alloc (sm100_op ()) "tmem_addr" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has alloc" true (contains "alloc" s)

let test_dealloc_tmem () =
  let s = Tile_op.emit_tmem_dealloc (sm100_op ()) "taddr" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has dealloc" true (contains "dealloc" s)

let () =
  Alcotest.run "Tile_op" [
    "flags",   [ Alcotest.test_case "tmem-f"   `Quick test_is_tmem_false
               ; Alcotest.test_case "tmem-t"   `Quick test_is_tmem_true
               ; Alcotest.test_case "wgmma-f"  `Quick test_is_wgmma_false
               ; Alcotest.test_case "wgmma-t"  `Quick test_is_wgmma_true
               ; Alcotest.test_case "dbl-f"    `Quick test_double_buf_false
               ; Alcotest.test_case "dbl-t"    `Quick test_double_buf_true ];
    "accum",   [ Alcotest.test_case "sm80"     `Quick test_accum_sm80
               ; Alcotest.test_case "sm100"    `Quick test_accum_sm100 ];
    "decl",    [ Alcotest.test_case "reg"      `Quick test_accum_decl_registers
               ; Alcotest.test_case "tmem"     `Quick test_accum_decl_tmem ];
    "mma",     [ Alcotest.test_case "sm80"     `Quick test_emit_mma_sm80
               ; Alcotest.test_case "sm100"    `Quick test_emit_mma_sm100 ];
    "commit",  [ Alcotest.test_case "sm80"     `Quick test_commit_sm80_empty
               ; Alcotest.test_case "sm100"    `Quick test_commit_sm100 ];
    "alloc",   [ Alcotest.test_case "reg-empty"`Quick test_alloc_registers_empty
               ; Alcotest.test_case "tmem"     `Quick test_alloc_tmem
               ; Alcotest.test_case "dealloc"  `Quick test_dealloc_tmem ];
  ]
