open Tesserae_runtime
open Tesserae_kernel
open Tesserae_test_kernels
open Base

let test_is_available () =
  Alcotest.(check bool) "gpu available" true
    (Runtime.is_available ())

let test_device_info () =
  let s = Runtime.device_info () in
  Alcotest.(check bool) "non-empty" true (String.length s > 0);
  Stdio.printf "GPU: %s\n%!" s

let test_run_ok () =
  Stdio.printf "starting runtime/run ok\n%!";
  let k = Kernel_ast.make
    ~name:"gemm_test"
    ~arch:Kernel_ast.SM90a
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ Kernel_ast.in_arg "A" Kernel_ast.F16
          ; Kernel_ast.in_arg "B" Kernel_ast.F16
          ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Seq []) in
  let result = Runtime.run k
      ~m:128 ~n:128 ~k_:32
      ~a:(Array.create ~len:(128 * 32) 1.0)
      ~b:(Array.create ~len:(32 * 128) 1.0)
    in
    match result with
    | Ok _ -> Alcotest.(check bool) "ok" true true
    | Error msg -> Alcotest.failf "Runtime.run failed with: %s" msg


let test_run_output_size () =
  Stdio.printf "starting runtime/run size\n%!";
  let k = Kernel_ast.make
    ~name:"gemm_size_test"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ Kernel_ast.in_arg "A" Kernel_ast.F16
          ; Kernel_ast.in_arg "B" Kernel_ast.F16
          ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Seq []) in
  match Runtime.run k
      ~m:128
      ~n:128
      ~k_:32
      ~a:(Array.create ~len:(128 * 32) 1.0)
      ~b:(Array.create ~len:(32 * 128) 1.0)
    with
    | Ok c    -> Alcotest.(check int) "output size" (128 * 128) (Array.length c)
    | Error e -> Alcotest.failf "failed: %s" e


let test_copy_kernel () =
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
        ~block:(64, 1, 1)
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


let () =
  Alcotest.run "Runtime" [
    "device",  [ Alcotest.test_case "available" `Quick test_is_available
               ; Alcotest.test_case "info"      `Quick test_device_info ];
    "run",     [ Alcotest.test_case "ok"        `Quick test_run_ok
               ; Alcotest.test_case "size"      `Quick test_run_output_size ];
    "copy",    [ Alcotest.test_case "kernel"    `Quick test_copy_kernel ];
  ]
