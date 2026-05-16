open Tesserae

let test_shape_sm80 () =
  let a = Mma_atom.sm80_16x8x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check (triple int int int)) "16x8x16" (16, 8, 16)
    (Mma_atom.shape a)

let test_shape_sm90 () =
  let a = Mma_atom.sm90_64x128x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check (triple int int int)) "64x128x16" (64, 128, 16)
    (Mma_atom.shape a)

let test_shape_sm100 () =
  let a = Mma_atom.sm100_128x128x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check (triple int int int)) "128x128x16" (128, 128, 16)
    (Mma_atom.shape a)

let test_thread_count_sm80 () =
  let a = Mma_atom.sm80_16x8x8_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check int) "sm80 32 threads" 32
    (Mma_atom.thread_count a)

let test_thread_count_sm90 () =
  let a = Mma_atom.sm90_64x64x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check int) "sm90 128 threads" 128
    (Mma_atom.thread_count a)

let test_thread_count_sm100 () =
  let a = Mma_atom.sm100_64x64x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check int) "sm100 128 threads" 128
    (Mma_atom.thread_count a)

(* ------------------------------------------------------------------ *)
(* is_wgmma                                                            *)
(* ------------------------------------------------------------------ *)

let test_is_wgmma_false () =
  let a = Mma_atom.sm80_16x8x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check bool) "sm80 not wgmma" false
    (Mma_atom.is_wgmma a)

let test_is_wgmma_true_sm90 () =
  let a = Mma_atom.sm90_64x64x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check bool) "sm90 is wgmma" true
    (Mma_atom.is_wgmma a)

let test_is_wgmma_true_sm100 () =
  let a = Mma_atom.sm100_64x64x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check bool) "sm100 is wgmma" true
    (Mma_atom.is_wgmma a)

let test_emit_sm80_tn () =
  (* ColMajor A, RowMajor B → TN in CUTLASS convention *)
  let a = Mma_atom.sm80_16x8x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check string) "sm80 TN"
    "SM80_16x8x16_F32F16F16F32_TN"
    (Mma_atom.emit_cpp a)

let test_emit_sm80_tt () =
  let a = Mma_atom.sm80_16x8x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.ColMajor in
  Alcotest.(check string) "sm80 TT"
    "SM80_16x8x16_F32F16F16F32_TT"
    (Mma_atom.emit_cpp a)

let test_emit_sm90_tn () =
  let a = Mma_atom.sm90_64x64x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check string) "sm90 TN"
    "SM90_64x64x16_F32F16F16F32_TN"
    (Mma_atom.emit_cpp a)

let test_emit_sm100_tn () =
  let a = Mma_atom.sm100_64x64x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check string) "sm100 TN"
    "SM100_64x64x16_F32F16F16F32_TN"
    (Mma_atom.emit_cpp a)

let test_emit_s32s8s8s32 () =
  let a = Mma_atom.sm80_16x8x32_s32s8s8s32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  Alcotest.(check string) "int8"
    "SM80_16x8x32_S32S8S8S32_TN"
    (Mma_atom.emit_cpp a)

let test_pp () =
  let a = Mma_atom.sm80_16x8x16_f32f16f16f32
            Mma_atom.ColMajor Mma_atom.RowMajor in
  let s = Stdlib.Format.asprintf "%a" Mma_atom.pp a in
  Alcotest.(check string) "pp"
    "SM80_16x8x16_F32F16F16F32_TN"
    s

let () =
  Alcotest.run "Mma_atom" [
    "shape",      [ Alcotest.test_case "sm80"  `Quick test_shape_sm80
                  ; Alcotest.test_case "sm90"  `Quick test_shape_sm90
                  ; Alcotest.test_case "sm100" `Quick test_shape_sm100 ];
    "threads",    [ Alcotest.test_case "sm80"  `Quick test_thread_count_sm80
                  ; Alcotest.test_case "sm90"  `Quick test_thread_count_sm90
                  ; Alcotest.test_case "sm100" `Quick test_thread_count_sm100 ];
    "is_wgmma",   [ Alcotest.test_case "false"     `Quick test_is_wgmma_false
                  ; Alcotest.test_case "sm90-true"  `Quick test_is_wgmma_true_sm90
                  ; Alcotest.test_case "sm100-true" `Quick test_is_wgmma_true_sm100 ];
    "emit_cpp",   [ Alcotest.test_case "sm80-tn"  `Quick test_emit_sm80_tn
                  ; Alcotest.test_case "sm80-tt"  `Quick test_emit_sm80_tt
                  ; Alcotest.test_case "sm90-tn"  `Quick test_emit_sm90_tn
                  ; Alcotest.test_case "sm100-tn" `Quick test_emit_sm100_tn
                  ; Alcotest.test_case "int8"     `Quick test_emit_s32s8s8s32 ];
    "pp",         [ Alcotest.test_case "basic" `Quick test_pp ];
  ]
