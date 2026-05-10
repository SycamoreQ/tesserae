open Tesserae

let test_bulk_bytes_f32 () =
  (* cp.async always transfers 16 bytes = 128 bits *)
  let a = Copy_atom.sm80_cp_async_global Elemtype.Float32 in
  Alcotest.(check int) "16 bytes" 16 (Copy_atom.bulk_bytes_of a)

let test_bulk_bytes_f16 () =
  let a = Copy_atom.sm80_cp_async_global Elemtype.Float16 in
  Alcotest.(check int) "16 bytes" 16 (Copy_atom.bulk_bytes_of a)

let test_bulk_bytes_tma () =
  (* TMA always transfers 128 bytes *)
  let a = Copy_atom.sm90_tma_load Elemtype.Float16 in
  Alcotest.(check int) "128 bytes" 128 (Copy_atom.bulk_bytes_of a)

let test_vec_width_f32_async () =
  (* 16 bytes / 4 bytes per float = 4 elements *)
  let a = Copy_atom.sm80_cp_async_global Elemtype.Float32 in
  Alcotest.(check int) "4 elems" 4 a.Copy_atom.vec_width

let test_vec_width_f16_async () =
  (* 16 bytes / 2 bytes per half = 8 elements *)
  let a = Copy_atom.sm80_cp_async_global Elemtype.Float16 in
  Alcotest.(check int) "8 elems" 8 a.Copy_atom.vec_width

let test_vec_width_ldmatrix () =
  (* ldmatrix always loads 8 elements *)
  let a = Copy_atom.sm80_ldmatrix Elemtype.Float16 in
  Alcotest.(check int) "8 elems" 8 a.Copy_atom.vec_width

let test_is_async_cp () =
  let a = Copy_atom.sm80_cp_async_global Elemtype.Float32 in
  Alcotest.(check bool) "cp.async is async" true
    (Copy_atom.is_async a)

let test_is_async_tma () =
  let a = Copy_atom.sm90_tma_load Elemtype.Float32 in
  Alcotest.(check bool) "tma is async" true
    (Copy_atom.is_async a)

let test_is_async_ldmatrix () =
  let a = Copy_atom.sm80_ldmatrix Elemtype.Float16 in
  Alcotest.(check bool) "ldmatrix not async" false
    (Copy_atom.is_async a)

let test_is_async_universal () =
  let a = Copy_atom.universal
    Memspace.Global Memspace.Shared Elemtype.Float32 in
  Alcotest.(check bool) "universal not async" false
    (Copy_atom.is_async a)


let test_is_tma_load () =
  let a = Copy_atom.sm90_tma_load Elemtype.Float16 in
  Alcotest.(check bool) "tma load" true (Copy_atom.is_tma a)

let test_is_tma_store () =
  let a = Copy_atom.sm90_tma_store Elemtype.Float16 in
  Alcotest.(check bool) "tma store" true (Copy_atom.is_tma a)

let test_is_tma_multicast () =
  let a = Copy_atom.sm100_tma_load_multicast Elemtype.Float16 in
  Alcotest.(check bool) "tma multicast" true (Copy_atom.is_tma a)

let test_is_tma_false () =
  let a = Copy_atom.sm80_cp_async_global Elemtype.Float32 in
  Alcotest.(check bool) "cp.async not tma" false (Copy_atom.is_tma a)

let test_mbar_tma () =
  let a = Copy_atom.sm90_tma_load Elemtype.Float32 in
  Alcotest.(check bool) "tma needs mbar" true
    (Copy_atom.requires_mbar a)

let test_mbar_cp_async () =
  let a = Copy_atom.sm80_cp_async_global Elemtype.Float32 in
  Alcotest.(check bool) "cp.async no mbar" false
    (Copy_atom.requires_mbar a)

let test_mbar_universal () =
  let a = Copy_atom.universal
    Memspace.Global Memspace.Shared Elemtype.Float32 in
  Alcotest.(check bool) "universal no mbar" false
    (Copy_atom.requires_mbar a)

let test_emit_cp_async_global_f32 () =
  let a = Copy_atom.sm80_cp_async_global Elemtype.Float32 in
  Alcotest.(check string) "cp async global f32"
    "SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>"
    (Copy_atom.emit_cpp a)

let test_emit_cp_async_cached_f16 () =
  let a = Copy_atom.sm80_cp_async_cached Elemtype.Float16 in
  Alcotest.(check string) "cp async cached f16"
    "SM80_CP_ASYNC_CACHEALL<cute::uint128_t>"
    (Copy_atom.emit_cpp a)

let test_emit_tma_load () =
  let a = Copy_atom.sm90_tma_load Elemtype.Float16 in
  Alcotest.(check string) "tma load"
    "SM90_TMA_LOAD"
    (Copy_atom.emit_cpp a)

let test_emit_tma_store () =
  let a = Copy_atom.sm90_tma_store Elemtype.Float32 in
  Alcotest.(check string) "tma store"
    "SM90_TMA_STORE"
    (Copy_atom.emit_cpp a)

let test_emit_tma_multicast () =
  let a = Copy_atom.sm100_tma_load_multicast Elemtype.Float16 in
  Alcotest.(check string) "tma multicast"
    "SM100_TMA_LOAD_MULTICAST"
    (Copy_atom.emit_cpp a)

let test_emit_ldmatrix () =
  let a = Copy_atom.sm80_ldmatrix Elemtype.Float16 in
  Alcotest.(check string) "ldmatrix"
    "SM80_U32x4_LDSM_N"
    (Copy_atom.emit_cpp a)

let test_emit_ldmatrix_trans () =
  let a = Copy_atom.sm80_ldmatrix_trans Elemtype.Float16 in
  Alcotest.(check string) "ldmatrix trans"
    "SM80_U16x8_LDSM_T"
    (Copy_atom.emit_cpp a)

let test_emit_universal () =
  let a = Copy_atom.universal
    Memspace.Global Memspace.Shared Elemtype.Float32 in
  Alcotest.(check string) "universal"
    "UniversalCopy<float>"
    (Copy_atom.emit_cpp a)

let test_pp () =
  let a = Copy_atom.sm90_tma_load Elemtype.Float16 in
  let s = Stdlib.Format.asprintf "%a" Copy_atom.pp a in
  Alcotest.(check string) "pp tma" "SM90_TMA_LOAD" s

let () =
  Alcotest.run "Copy_atom" [
    "bulk_bytes", [ Alcotest.test_case "f32"  `Quick test_bulk_bytes_f32
                  ; Alcotest.test_case "f16"  `Quick test_bulk_bytes_f16
                  ; Alcotest.test_case "tma"  `Quick test_bulk_bytes_tma ];
    "vec_width",  [ Alcotest.test_case "f32-async"  `Quick test_vec_width_f32_async
                  ; Alcotest.test_case "f16-async"  `Quick test_vec_width_f16_async
                  ; Alcotest.test_case "ldmatrix"   `Quick test_vec_width_ldmatrix ];
    "is_async",   [ Alcotest.test_case "cp"        `Quick test_is_async_cp
                  ; Alcotest.test_case "tma"       `Quick test_is_async_tma
                  ; Alcotest.test_case "ldmatrix"  `Quick test_is_async_ldmatrix
                  ; Alcotest.test_case "universal" `Quick test_is_async_universal ];
    "is_tma",     [ Alcotest.test_case "load"      `Quick test_is_tma_load
                  ; Alcotest.test_case "store"     `Quick test_is_tma_store
                  ; Alcotest.test_case "multicast" `Quick test_is_tma_multicast
                  ; Alcotest.test_case "false"     `Quick test_is_tma_false ];
    "mbar",       [ Alcotest.test_case "tma"       `Quick test_mbar_tma
                  ; Alcotest.test_case "cp_async"  `Quick test_mbar_cp_async
                  ; Alcotest.test_case "universal" `Quick test_mbar_universal ];
    "emit_cpp",   [ Alcotest.test_case "cp-global" `Quick test_emit_cp_async_global_f32
                  ; Alcotest.test_case "cp-cached" `Quick test_emit_cp_async_cached_f16
                  ; Alcotest.test_case "tma-load"  `Quick test_emit_tma_load
                  ; Alcotest.test_case "tma-store" `Quick test_emit_tma_store
                  ; Alcotest.test_case "multicast" `Quick test_emit_tma_multicast
                  ; Alcotest.test_case "ldmatrix"  `Quick test_emit_ldmatrix
                  ; Alcotest.test_case "ldm-trans" `Quick test_emit_ldmatrix_trans
                  ; Alcotest.test_case "universal" `Quick test_emit_universal ];
    "pp",         [ Alcotest.test_case "basic" `Quick test_pp ];
  ]
