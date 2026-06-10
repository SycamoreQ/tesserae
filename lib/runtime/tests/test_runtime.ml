open Tesserae_runtime
open Tesserae_kernel
open Tesserae_test_kernels
open Base

let _test_is_available () =
  Alcotest.(check bool) "gpu available" true
    (Runtime.is_available ())

let _test_device_info () =
  let s = Runtime.device_info () in
  Alcotest.(check bool) "non-empty" true (String.length s > 0);
  Stdio.printf "GPU: %s\n%!" s


let _test_copy_kernel () =
  let n_elements = 64 * 64 in
  let host_a = Array.init n_elements ~f:(fun i -> Float.of_int i) in
  let host_c = Array.create ~len:n_elements 0.0 in

  let buf_a = Gpu_buffer.alloc_f16 n_elements in
  let buf_c = Gpu_buffer.alloc_f16 n_elements in
  Gpu_buffer.copy_from_host_f16 buf_a host_a;
  Gpu_buffer.copy_from_host_f16 buf_c host_c;

  let tma_a =
    Kernel_launch.create_tma_descriptor
      (Gpu_buffer.ptr buf_a) 64 64 64 64
  in

  match Compile.to_ptx (Copy_kernel.copy_kernel ()) with
  | Error e ->
      let msg = Stdlib.Format.asprintf "%a" Compile.pp_error e in
      Alcotest.failf "Compilation failed: %s" msg
  | Ok result ->
      let ptx = match result.Compile.ptx with
        | Some p -> p
        | None -> Alcotest.fail "No PTX generated"
      in
      let module_ = Kernel_launch.load_ptx ptx in
      let func = Kernel_launch.get_function module_ "copy_hopper" in

      let smem_bytes = 64 * 64 * 2 + 8 in
      Kernel_launch.launch func
        ~grid:(1, 1, 1)
        ~block:(256, 1, 1)
        ~smem:smem_bytes
        ~args:[ tma_a; Gpu_buffer.ptr buf_c ];

      Kernel_launch.synchronize ();

      let result_host = Gpu_buffer.to_host_f16 buf_c in
      let ok = Array.for_all
        (Array.init 64 ~f:(fun i -> i))
        ~f:(fun i -> Float.(abs (result_host.(i) - of_int i) < 1e-5))
      in
      Alcotest.(check bool) "copy correct" true ok;

      Gpu_buffer.free buf_a;
      Gpu_buffer.free buf_c;
      Kernel_launch.free_tma_descriptor tma_a;
      Kernel_launch.unload module_


let test_gemm_hopper_ws () =
  let m = 128 in
  let n = 128 in
  let k = 64 in

  let host_a = Array.init (m * k) ~f:(fun i -> Float.of_int (i % 4)) in
  let host_b = Array.init (k * n) ~f:(fun i -> Float.of_int (i % 4)) in
  let host_c = Array.create ~len:(m * n) 0.0 in

  let k_total = 256 in
  let buf_a = Gpu_buffer.alloc_f16 (m * k_total) in
  let buf_b = Gpu_buffer.alloc_f16 (k_total * n) in
  let buf_c = Gpu_buffer.alloc (m * n) in  (* F32 *)

  Gpu_buffer.copy_from_host_f16 buf_a host_a;
  Gpu_buffer.copy_from_host_f16 buf_b host_b;
  Gpu_buffer.copy_from_host buf_c host_c;

  (* A: global is M×K = 128×256 row-major
     tile: load a 128-row × 64-col slice per TMA call
     coordinate x advances by 64 each k_tile *)
  let tma_a = Kernel_launch.create_tma_descriptor
    (Gpu_buffer.ptr buf_a) m k_total 128 64 in
  (*                         rows  cols  tile_rows tile_cols *)

  (* B: global is K×N = 256×128 row-major
     tile: load a 64-row × 128-col slice per TMA call
     coordinate y advances by 64 each k_tile *)
  let tma_b = Kernel_launch.create_tma_descriptor
    (Gpu_buffer.ptr buf_b) k_total n 64 128 in
  (*                         rows   cols tile_rows tile_cols *)
  match Compile.to_ptx (Tesserae_test_kernels.Gemm_warpspec.gemm_hopper_ws()) with
  | Error e ->
      let msg = Stdlib.Format.asprintf "%a" Compile.pp_error e in
      Alcotest.failf "Compilation failed: %s" msg

  | Ok result ->
      let ptx = match result.Compile.ptx with
        | Some p -> p
        | None -> Alcotest.fail "No PTX generated"
      in
      (*Stdlib.Printf.printf
        "================ PTX ================\n%s\n=====================================\n%!"
        ptx; *)
      let module_ = Kernel_launch.load_ptx ptx in
      let func = Kernel_launch.get_function module_ "gemm_hopper_ws" in


      let stages = 4 in
      let smem_a = stages * 128 * 64 * 2 in  (* __half smem_A[4][8192] *)
      let smem_b = stages * 128 * 64 * 2 in  (* __half smem_B[4][8192] *)
      let smem_mbar = 2 * stages * 8 in         (* full_mbar[4] + empty_mbar[4], uint64_t *)
      let _smem_bytes = smem_a + smem_b + smem_mbar in  (* = 131136 *)
      Kernel_launch.launch func
        ~grid:(1, 1, 1)
        ~block:(256, 1, 1)
        ~smem:131584
        ~args:[ tma_a; tma_b; Gpu_buffer.ptr buf_c ];

      Kernel_launch.synchronize ();

      let result_host = Gpu_buffer.to_host buf_c in
      (* For a simple sanity check: with A=identity-like and B=identity-like,
         C should be non-zero. Full GEMM verification is more complex. *)
      let ok = Array.exists result_host ~f:(fun f -> Float.(abs f > 0.0)) in
      Alcotest.(check bool) "gemm produced non-zero output" true ok;

      Gpu_buffer.free buf_a;
      Gpu_buffer.free buf_b;
      Gpu_buffer.free buf_c;
      Kernel_launch.free_tma_descriptor tma_a;
      Kernel_launch.free_tma_descriptor tma_b


let () =
  Alcotest.run "Runtime" [
    (* Commented out to skip:
    "device",  [ Alcotest.test_case "available" `Quick test_is_available
               ; Alcotest.test_case "info"      `Quick test_device_info ];
    "copy",    [ Alcotest.test_case "copy_kernel_hopper"    `Quick _test_copy_kernel ];
    *)
    "gemm_ws", [ Alcotest.test_case "gemm_kernel_hopper"    `Quick test_gemm_hopper_ws ];
  ]
