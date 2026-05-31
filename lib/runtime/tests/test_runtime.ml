open Tesserae_runtime
open Tesserae_kernel
open Tesserae_test_kernels
open Base

(* Runtime is the top-level user API.
   Functions needed:
   - run : Kernel_ast.kernel
       -> m:int -> n:int -> k:int
       -> a:float array -> b:float array
       -> (float array, string) Result.t
       Full pipeline: lower → emit → nvrtc → load → launch → copy back
   - run_exn : same but raises on error
   - device_info : unit -> string
       return GPU name and compute capability
   - is_available : unit -> bool
       true iff a CUDA device is present *)

let test_is_available () =
  Alcotest.(check bool) "gpu available" true
    (Runtime.is_available ())

let test_device_info () =
  let s = Runtime.device_info () in
  Alcotest.(check bool) "non-empty" true (String.length s > 0);
  Stdio.printf "GPU: %s\n%!" s  (* Base-compliant printing *)


let test_run_ok () =
  let k = Kernel_ast.make
    ~name:"gemm_test"
    ~arch:Kernel_ast.SM90a
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
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
  let k = Kernel_ast.make
    ~name:"gemm_size_test"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq []) in
  match Runtime.run k
      ~m:128
      ~n:128
      ~k_:32
      ~a:(Array.create ~len:(128 * 32) 1.0)  (* Array.make -> Array.create *)
      ~b:(Array.create ~len:(32 * 128) 1.0)  (* Array.make -> Array.create *)
    with
    | Ok c    -> Alcotest.(check int) "output size" (128 * 128) (Array.length c)
    | Error e -> Alcotest.failf "failed: %s" e


let test_copy_kernel () =
  let n_elements = 64 * 64 in
  let host_a = Array.init n_elements ~f:(fun i -> Float.of_int i) in
  let host_c = Array.create ~len:n_elements 0.0 in

  let buf_a = Gpu_buffer.alloc n_elements in
  let buf_c = Gpu_buffer.alloc n_elements in
  Gpu_buffer.copy_from_host buf_a host_a;
  Gpu_buffer.copy_from_host buf_c host_c;

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

    let smem_bytes = 64 * 8 * 2 * 2 in  (* M*K * elem_size * stages *)
    Kernel_launch.launch func
      ~grid:(1, 1, 1)
      ~block:(64, 1, 1)
      ~smem:smem_bytes
      ~args:[ Gpu_buffer.ptr buf_a
            ; Gpu_buffer.ptr buf_c ];

    Kernel_launch.synchronize ();

    let result_host = Gpu_buffer.to_host buf_c in
    let ok = Array.for_all
      (Array.init 64 ~f:(fun i -> i))
      ~f:(fun i -> Float.(abs (result_host.(i) - of_int i) < 1e-5))
    in
    Alcotest.(check bool) "copy correct" true ok;

    Gpu_buffer.free buf_a;
    Gpu_buffer.free buf_c;
    Kernel_launch.unload module_

let () =
  Alcotest.run "Runtime" [
    "device",  [ Alcotest.test_case "available" `Quick test_is_available
               ; Alcotest.test_case "info"      `Quick test_device_info ];
    "run",     [ Alcotest.test_case "ok"        `Quick test_run_ok
               ; Alcotest.test_case "size"      `Quick test_run_output_size ];
    "copy",    [ Alcotest.test_case "kernel"    `Quick test_copy_kernel ]; (* Added to test runner! *)
  ]
