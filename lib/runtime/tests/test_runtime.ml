open Tesserae_runtime
open Tesserae_kernel
open Base

module Hopper_copy = Tesserae_test_kernels.Copy_kernel
module Hopper_gemm = Tesserae_test_kernels.Gemm_warpspec
module Blackwell_copy = Tesserae_blackwell_kernels.Copy_kernel
module Blackwell_gemm = Tesserae_blackwell_kernels.Gemm_warpspec

let test_is_available () =
  Alcotest.(check bool) "gpu available" true
    (Runtime.is_available ())

let test_device_info () =
  let s = Runtime.device_info () in
  Alcotest.(check bool) "non-empty" true (String.length s > 0);
  Stdio.printf "GPU: %s\n%!" s

let _test_copy_kernel_hopper () =
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

  match Compile.to_ptx (Hopper_copy.copy_kernel ()) with
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
        ~args:[ tma_a; Gpu_buffer.ptr buf_c ] ();

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

let test_copy_kernel_blackwell () =
  let n_elements = 64 * 64 in
  let host_a = Array.init n_elements ~f:(fun i -> Float.of_int i) in
  let host_c = Array.create ~len:(n_elements * 2) 0.0 in

  let buf_a = Gpu_buffer.alloc_f16 n_elements in
  let buf_c = Gpu_buffer.alloc_f16 (n_elements * 2) in
  Gpu_buffer.copy_from_host_f16 buf_a host_a;
  Gpu_buffer.copy_from_host_f16 buf_c host_c;

  let tma_a =
    Kernel_launch.create_tma_descriptor
      (Gpu_buffer.ptr buf_a) 64 64 64 64
  in

  match Compile.to_ptx (Blackwell_copy.copy_kernel ()) with
  | Error e ->
      let msg = Stdlib.Format.asprintf "%a" Compile.pp_error e in
      Alcotest.failf "Compilation failed: %s" msg
  | Ok result ->
      let ptx = match result.Compile.ptx with
        | Some p -> p
        | None -> Alcotest.fail "No PTX generated"
      in
      let module_ = Kernel_launch.load_ptx ptx in
      let func = Kernel_launch.get_function module_ "copy_blackwell" in

      let smem_bytes = 64 * 64 * 2 + 8 in
      (* Blackwell SM100a uses clusterDim (2, 1, 1) and 192 threads (6 warps per CTA) *)
      Kernel_launch.launch func
        ~grid:(2, 1, 1)
        ~block:(192, 1, 1)
        ~smem:smem_bytes
        ~cluster:(2, 1, 1)
        ~args:[ tma_a; Gpu_buffer.ptr buf_c ] ();

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

let _test_gemm_hopper_ws () =
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

  let tma_a = Kernel_launch.create_tma_descriptor
    (Gpu_buffer.ptr buf_a) m k_total 128 64 in
  let tma_b = Kernel_launch.create_tma_descriptor
    (Gpu_buffer.ptr buf_b) k_total n 64 128 in

  match Compile.to_ptx (Hopper_gemm.gemm_hopper_ws ()) with
  | Error e ->
      let msg = Stdlib.Format.asprintf "%a" Compile.pp_error e in
      Alcotest.failf "Compilation failed: %s" msg

  | Ok result ->
      let ptx = match result.Compile.ptx with
        | Some p -> p
        | None -> Alcotest.fail "No PTX generated"
      in
      let module_ = Kernel_launch.load_ptx ptx in
      let func = Kernel_launch.get_function module_ "gemm_hopper_ws" in

      let stages = 4 in
      let smem_a = stages * 128 * 64 * 2 in
      let smem_b = stages * 128 * 64 * 2 in
      let smem_mbar = 2 * stages * 8 in
      let smem_bytes = smem_a + smem_b + smem_mbar in
      Kernel_launch.launch func
        ~grid:(1, 1, 1)
        ~block:(256, 1, 1)
        ~smem:smem_bytes
        ~args:[ tma_a; tma_b; Gpu_buffer.ptr buf_c ] ();

      Kernel_launch.synchronize ();

      let result_host = Gpu_buffer.to_host buf_c in
      let ok = Array.exists result_host ~f:(fun f -> Float.(abs f > 0.0)) in
      Alcotest.(check bool) "gemm produced non-zero output" true ok;

      Gpu_buffer.free buf_a;
      Gpu_buffer.free buf_b;
      Gpu_buffer.free buf_c;
      Kernel_launch.free_tma_descriptor tma_a;
      Kernel_launch.free_tma_descriptor tma_b;
      Kernel_launch.unload module_

(* How many times the GEMM is launched. A missing barrier or a race usually
   passes most runs, so repeat the launch + full check many times. *)
let gemm_iterations = 20

let test_gemm_blackwell_ws () =
  let m = 128 in
  let n = 128 in
  let k_total = 256 in
  (* A is [m][k], B is [n][k] (both K-major). Small integers: exact in f16,
     and the f32 sums are exact, so we can compare with Float.equal. *)
  let host_a = Array.init (m * k_total) ~f:(fun i ->
    let r = i / k_total and k = i % k_total in
    Float.of_int (((r * 3 + k * 5) % 7) - 3)) in
  let host_b = Array.init (n * k_total) ~f:(fun i ->
    let c = i / k_total and k = i % k_total in
    Float.of_int (((c * 2 + k * 3) % 7) - 3)) in
  (* two CTAs -> two output tiles; NaN so unwritten elements are visible *)
  let host_c = Array.create ~len:(m * n * 2) Float.nan in

  let buf_a = Gpu_buffer.alloc_f16 (m * k_total) in
  let buf_b = Gpu_buffer.alloc_f16 (n * k_total) in
  let buf_c = Gpu_buffer.alloc (m * n * 2) in  (* F32 *)

  Gpu_buffer.copy_from_host_f16 buf_a host_a;
  Gpu_buffer.copy_from_host_f16 buf_b host_b;
  Gpu_buffer.copy_from_host buf_c host_c;

  (* 128B-swizzled maps: the tcgen05 smem descriptor assumes SW128, K-major *)
  let tma_a = Kernel_launch.create_tma_descriptor_sw128
    (Gpu_buffer.ptr buf_a) m k_total 128 64 in
  let tma_b = Kernel_launch.create_tma_descriptor_sw128
    (Gpu_buffer.ptr buf_b) n k_total 128 64 in

  match Compile.to_ptx (Blackwell_gemm.gemm_blackwell_ws ~k_total ()) with
  | Error e ->
      let msg = Stdlib.Format.asprintf "%a" Compile.pp_error e in
      Alcotest.failf "Compilation failed: %s" msg

  | Ok result ->
      let ptx = match result.Compile.ptx with
        | Some p -> p
        | None -> Alcotest.fail "No PTX generated"
      in
      let module_ = Kernel_launch.load_ptx ptx in
      let func = Kernel_launch.get_function module_ "gemm_blackwell_ws" in

      let stages = 4 in
      let smem_a = stages * 128 * 64 * 2 in
      let smem_b = stages * 128 * 64 * 2 in
      let smem_mbar = 2 * stages * 8 + 256 in
      let smem_bytes = smem_a + smem_b + smem_mbar in

      let ref_ r c =
        let acc = ref 0.0 in
        for k = 0 to k_total - 1 do
          acc := !acc +. host_a.(r * k_total + k) *. host_b.(c * k_total + k)
        done;
        !acc
      in
      let expected = Array.init (m * n) ~f:(fun i -> ref_ (i / n) (i % n)) in

      let run_once () =
        (* reset C to NaN so results from the previous iteration can't mask a
           missing write or a race *)
        Gpu_buffer.copy_from_host buf_c host_c;
        Kernel_launch.launch func
          ~grid:(2, 1, 1)
          ~block:(192, 1, 1)
          ~smem:smem_bytes
          ~cluster:(2, 1, 1)
          ~args:[ tma_a; tma_b; Gpu_buffer.ptr buf_c ] ();
        Kernel_launch.synchronize ();

        let result_host = Gpu_buffer.to_host buf_c in
        let bad = ref 0 and first = ref "" in
        for r = 0 to 2 * m - 1 do
          for c = 0 to n - 1 do
            let got = result_host.(r * n + c)
            and exp = expected.((r % m) * n + c) in
            if not (Float.equal got exp) then begin
              if !bad = 0 then
                first := Printf.sprintf "r=%d c=%d got=%g exp=%g" r c got exp;
              Int.incr bad
            end
          done
        done;
        (!bad, !first)
      in

      let failed = ref 0 and first_fail = ref "" in
      for iter = 1 to gemm_iterations do
        let (bad, first) = run_once () in
        if bad > 0 then begin
          if !failed = 0 then
            first_fail :=
              Printf.sprintf "iteration %d: %d/%d mismatches, first %s"
                iter bad (2 * m * n) first;
          Int.incr failed
        end
      done;

      Gpu_buffer.free buf_a;
      Gpu_buffer.free buf_b;
      Gpu_buffer.free buf_c;
      Kernel_launch.free_tma_descriptor tma_a;
      Kernel_launch.free_tma_descriptor tma_b;
      Kernel_launch.unload module_;

      if !failed > 0 then
        Alcotest.failf "%d/%d iterations failed; %s"
          !failed gemm_iterations !first_fail


let gemm_blackwell_ws_with_k ~k_total =
  let m = 128 in
  let n = 128 in
  (* A is [m][k], B is [n][k] (both K-major). Small integers: exact in f16,
     and the f32 sums are exact, so we can compare with Float.equal. *)
  let host_a = Array.init (m * k_total) ~f:(fun i ->
    let r = i / k_total and k = i % k_total in
    Float.of_int (((r * 3 + k * 5) % 7) - 3)) in
  let host_b = Array.init (n * k_total) ~f:(fun i ->
    let c = i / k_total and k = i % k_total in
    Float.of_int (((c * 2 + k * 3) % 7) - 3)) in
  (* two CTAs -> two output tiles; NaN so unwritten elements are visible *)
  let host_c = Array.create ~len:(m * n * 2) Float.nan in

  let buf_a = Gpu_buffer.alloc_f16 (m * k_total) in
  let buf_b = Gpu_buffer.alloc_f16 (n * k_total) in
  let buf_c = Gpu_buffer.alloc (m * n * 2) in  (* F32 *)

  Gpu_buffer.copy_from_host_f16 buf_a host_a;
  Gpu_buffer.copy_from_host_f16 buf_b host_b;
  Gpu_buffer.copy_from_host buf_c host_c;

  (* 128B-swizzled maps: the tcgen05 smem descriptor assumes SW128, K-major *)
  let tma_a = Kernel_launch.create_tma_descriptor_sw128
    (Gpu_buffer.ptr buf_a) m k_total 128 64 in
  let tma_b = Kernel_launch.create_tma_descriptor_sw128
    (Gpu_buffer.ptr buf_b) n k_total 128 64 in

  match Compile.to_ptx (Blackwell_gemm.gemm_blackwell_ws ~k_total ()) with
  | Error e ->
      let msg = Stdlib.Format.asprintf "%a" Compile.pp_error e in
      Alcotest.failf "Compilation failed: %s" msg

  | Ok result ->
      let ptx = match result.Compile.ptx with
        | Some p -> p
        | None -> Alcotest.fail "No PTX generated"
      in
      let module_ = Kernel_launch.load_ptx ptx in
      let func = Kernel_launch.get_function module_ "gemm_blackwell_ws" in

      let stages = 4 in
      let smem_a = stages * 128 * 64 * 2 in
      let smem_b = stages * 128 * 64 * 2 in
      let smem_mbar = 2 * stages * 8 + 256 in
      let smem_bytes = smem_a + smem_b + smem_mbar in

      let ref_ r c =
        let acc = ref 0.0 in
        for k = 0 to k_total - 1 do
          acc := !acc +. host_a.(r * k_total + k) *. host_b.(c * k_total + k)
        done;
        !acc
      in
      let expected = Array.init (m * n) ~f:(fun i -> ref_ (i / n) (i % n)) in

      let run_once () =
        (* reset C to NaN so results from the previous iteration can't mask a
           missing write or a race *)
        Gpu_buffer.copy_from_host buf_c host_c;
        Kernel_launch.launch func
          ~grid:(2, 1, 1)
          ~block:(192, 1, 1)
          ~smem:smem_bytes
          ~cluster:(2, 1, 1)
          ~args:[ tma_a; tma_b; Gpu_buffer.ptr buf_c ] ();
        Kernel_launch.synchronize ();

        let result_host = Gpu_buffer.to_host buf_c in
        let bad = ref 0 and first = ref "" in
        for r = 0 to 2 * m - 1 do
          for c = 0 to n - 1 do
            let got = result_host.(r * n + c)
            and exp = expected.((r % m) * n + c) in
            if not (Float.equal got exp) then begin
              if !bad = 0 then
                first := Printf.sprintf "r=%d c=%d got=%g exp=%g" r c got exp;
              Int.incr bad
            end
          done
        done;
        (!bad, !first)
      in

      let failed = ref 0 and first_fail = ref "" in
      for iter = 1 to gemm_iterations do
        let (bad, first) = run_once () in
        if bad > 0 then begin
          if !failed = 0 then
            first_fail :=
              Printf.sprintf "iteration %d: %d/%d mismatches, first %s"
                iter bad (2 * m * n) first;
          Int.incr failed
        end
      done;

      Gpu_buffer.free buf_a;
      Gpu_buffer.free buf_b;
      Gpu_buffer.free buf_c;
      Kernel_launch.free_tma_descriptor tma_a;
      Kernel_launch.free_tma_descriptor tma_b;
      Kernel_launch.unload module_;

      if !failed > 0 then
        Alcotest.failf "K=%d: %d/%d iterations failed; %s"
          k_total !failed gemm_iterations !first_fail

(* K=1024 = 16 tiles over 4 stages, so every stage is refilled 3 times.
   Expected to fail (or hang) until the empty_mbar handshake exists. *)
let test_gemm_blackwell_ws_k1024 () = gemm_blackwell_ws_with_k ~k_total:1024

let () =
  Alcotest.run "Runtime" [
    "device",               [ Alcotest.test_case "available"                   `Quick test_is_available
                            ; Alcotest.test_case "info"                        `Quick test_device_info ];
    "copy_blackwell",       [ Alcotest.test_case "copy_kernel_blackwell"       `Quick test_copy_kernel_blackwell ];
    "gemm_blackwell",       [ Alcotest.test_case "gemm_kernel_blackwell"       `Quick test_gemm_blackwell_ws ];
    "gemm_blackwell_k1024", [ Alcotest.test_case "gemm_kernel_blackwell_k1024" `Quick test_gemm_blackwell_ws_k1024 ];
  ]
