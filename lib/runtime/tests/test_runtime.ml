open Tesserae

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

let small_identity_a () =
  (* 4x4 identity matrix, row major, as flat array *)
  Array.init 16 (fun i -> if i mod 5 = 0 then 1.0 else 0.0)

let small_b () =
  Array.init 16 (fun i -> float_of_int (i + 1))

(* ------------------------------------------------------------------ *)
(* device_info / is_available                                          *)
(* ------------------------------------------------------------------ *)

let test_is_available () =
  Alcotest.(check bool) "gpu available" true
    (Runtime.is_available ())

let test_device_info () =
  let s = Runtime.device_info () in
  Alcotest.(check bool) "non-empty" true (String.length s > 0);
  Printf.printf "GPU: %s\n%!" s

(* ------------------------------------------------------------------ *)
(* run — small matmul                                                  *)
(* ------------------------------------------------------------------ *)

let test_run_ok () =
  let k = Kernel_ast.make
    ~name:"gemm_test"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq []) in
  let result = Runtime.run k
    ~m:128 ~n:128 ~k:32
    ~a:(Array.make (128*32) 1.0)
    ~b:(Array.make (32*128) 1.0) in
  Alcotest.(check bool) "ok" true (Result.is_ok result)

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
    ~m:128 ~n:128 ~k:32
    ~a:(Array.make (128*32) 1.0)
    ~b:(Array.make (32*128) 1.0) with
  | Ok c  -> Alcotest.(check int) "output size" (128*128) (Array.length c)
  | Error e -> Alcotest.failf "failed: %s" e

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Runtime" [
    "device",  [ Alcotest.test_case "available" `Quick test_is_available
               ; Alcotest.test_case "info"      `Quick test_device_info ];
    "run",     [ Alcotest.test_case "ok"        `Quick test_run_ok
               ; Alcotest.test_case "size"      `Quick test_run_output_size ];
  ]
