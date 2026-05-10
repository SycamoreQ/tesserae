open Tesserae

(* BM=128, BN=256, BK=64, bf16 (2 bytes), 2SM so B is halved
   tile_bytes = (128 + 128) * 64 * 2 = 32768 bytes *)
let p4 () = Pipeline.make 4 32768

let test_make_valid () =
  let p = p4 () in
  Alcotest.(check int) "depth" 4 p.Pipeline.depth;
  Alcotest.(check int) "tile"  32768 p.Pipeline.tile_bytes;
  Alcotest.(check int) "smem"  131072 p.Pipeline.smem_bytes

let test_make_invalid_depth () =
  Alcotest.check_raises "depth=0"
    (Invalid_argument "pipeline depth must be >= 1")
    (fun () -> ignore (Pipeline.make 0 32768))

let test_make_invalid_tile () =
  Alcotest.check_raises "tile=0"
    (Invalid_argument "tile_bytes must be > 0")
    (fun () -> ignore (Pipeline.make 4 0))

let test_stage_of () =
  Alcotest.(check int) "iter 0" 0 (Pipeline.stage_of 0 4);
  Alcotest.(check int) "iter 3" 3 (Pipeline.stage_of 3 4);
  Alcotest.(check int) "iter 4" 0 (Pipeline.stage_of 4 4);
  Alcotest.(check int) "iter 7" 3 (Pipeline.stage_of 7 4)

let test_phase_of () =
  Alcotest.(check int) "iter 0" 0 (Pipeline.phase_of 0 4);
  Alcotest.(check int) "iter 3" 0 (Pipeline.phase_of 3 4);
  Alcotest.(check int) "iter 4" 1 (Pipeline.phase_of 4 4);
  Alcotest.(check int) "iter 8" 0 (Pipeline.phase_of 8 4)

let test_smem_offset () =
  let p = p4 () in
  Alcotest.(check int) "stage 0" 0       (Pipeline.smem_offset_of p 0);
  Alcotest.(check int) "stage 1" 32768   (Pipeline.smem_offset_of p 1);
  Alcotest.(check int) "stage 3" 98304   (Pipeline.smem_offset_of p 3)

let test_a_offset () =
  let p = p4 () in
  (* A tile at start of stage slot *)
  Alcotest.(check int) "a stage 0" 0     (Pipeline.a_smem_offset_of p 0 128 64 2);
  Alcotest.(check int) "a stage 1" 32768 (Pipeline.a_smem_offset_of p 1 128 64 2)

let test_b_offset () =
  let p = p4 () in
  (* B follows A: offset = stage_offset + 128*64*2 = stage_offset + 16384 *)
  Alcotest.(check int) "b stage 0" 16384 (Pipeline.b_smem_offset_of p 0 128 64 2);
  Alcotest.(check int) "b stage 1" 49152 (Pipeline.b_smem_offset_of p 1 128 64 2)

let test_emit_full_mbar () =
  let p = p4 () in
  Alcotest.(check string) "full mbar"
    "__shared__ __align__(8) uint64_t full_mbar[4];"
    (Pipeline.emit_full_mbar "full_mbar" p)

let test_emit_empty_mbar () =
  let p = p4 () in
  Alcotest.(check string) "empty mbar"
    "__shared__ __align__(8) uint64_t empty_mbar[4];"
    (Pipeline.emit_empty_mbar "empty_mbar" p)

let test_emit_smem_buf () =
  let p = p4 () in
  Alcotest.(check string) "smem buf"
    "extern __shared__ char smem[];"
    (Pipeline.emit_smem_buf "smem" p)

let test_emit_advance_stage () =
  let p = p4 () in
  Alcotest.(check string) "advance"
    "stage = (stage + 1) % 4;"
    (Pipeline.emit_advance_stage "stage" p)

let test_emit_phase_toggle () =
  let p = p4 () in
  Alcotest.(check string) "phase toggle"
    "if (stage == 0) phase ^= 1;"
    (Pipeline.emit_phase_toggle "phase" "stage" p)

let () =
  Alcotest.run "Pipeline" [
    "make",    [ Alcotest.test_case "valid"   `Quick test_make_valid
               ; Alcotest.test_case "depth"   `Quick test_make_invalid_depth
               ; Alcotest.test_case "tile"    `Quick test_make_invalid_tile ];
    "stage",   [ Alcotest.test_case "stage"   `Quick test_stage_of
               ; Alcotest.test_case "phase"   `Quick test_phase_of ];
    "offset",  [ Alcotest.test_case "smem"    `Quick test_smem_offset
               ; Alcotest.test_case "a"       `Quick test_a_offset
               ; Alcotest.test_case "b"       `Quick test_b_offset ];
    "emit",    [ Alcotest.test_case "full"    `Quick test_emit_full_mbar
               ; Alcotest.test_case "empty"   `Quick test_emit_empty_mbar
               ; Alcotest.test_case "buf"     `Quick test_emit_smem_buf
               ; Alcotest.test_case "advance" `Quick test_emit_advance_stage
               ; Alcotest.test_case "toggle"  `Quick test_emit_phase_toggle ];
  ]
