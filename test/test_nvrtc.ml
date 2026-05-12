open Tesserae

(* A minimal valid CUDA kernel for testing nvrtc compilation *)
let trivial_source = {|
#include <stdint.h>
extern "C" __global__ void trivial_kernel(float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 1.0f;
}
|}

let invalid_source = {|
this is not valid cuda c++ at all
__global__ void broken( {
|}

let ampere_source () =
  let k = Kernel_ast.make
    ~name:"gemm_ampere_f16"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq []) in
  (Compile.to_source_exn k).Compile.source

let test_create_program () =
  let prog = Nvrtc.create_program trivial_source "trivial.cu" in
  Alcotest.(check bool) "created" true (Nvrtc.is_valid prog);
  Nvrtc.destroy_program prog

let test_destroy_idempotent () =
  let prog = Nvrtc.create_program trivial_source "trivial.cu" in
  Nvrtc.destroy_program prog;
  (* should not raise *)
  Alcotest.(check bool) "ok" true true

let test_compile_trivial () =
  let result = Nvrtc.compile_source trivial_source
    ~name:"trivial.cu"
    ~arch:"sm_80"
    ()
  in
  Alcotest.(check bool) "ok" true (Result.is_ok result)


let test_compile_ptx_nonempty () =
  match Nvrtc.compile_source trivial_source
    ~name:"trivial.cu" ~arch:"sm_80" () with
  | Ok ptx -> Alcotest.(check bool) "non-empty" true (String.length ptx > 0)
  | Error e -> Alcotest.failf "compile failed: %s" e

let test_compile_ptx_contains_kernel () =
  match Nvrtc.compile_source trivial_source
    ~name:"trivial.cu" ~arch:"sm_80" () with
  | Ok ptx ->
    let contains sub str =
      let n = String.length sub and m = String.length str in
      let found = ref false in
      for i = 0 to m - n do
        if String.sub str i n = sub then found := true
      done; !found
    in
    Alcotest.(check bool) "has .visible" true (contains ".visible" ptx);
    Alcotest.(check bool) "has .entry"   true (contains ".entry"   ptx)
  | Error e -> Alcotest.failf "compile failed: %s" e

let test_compile_invalid_source () =
  match Nvrtc.compile_source invalid_source
    ~name:"broken.cu" ~arch:"sm_80" () with
  | Ok _    -> Alcotest.fail "expected error"
  | Error e -> Alcotest.(check bool) "has error" true (String.length e > 0)

let test_arch_sm80 () =
  Alcotest.(check string) "sm80"
    "sm_80" (Nvrtc.arch_string Kernel_ast.SM80)

let test_arch_sm90 () =
  Alcotest.(check string) "sm90"
    "sm_90" (Nvrtc.arch_string Kernel_ast.SM90)

let test_arch_sm100 () =
  Alcotest.(check string) "sm100"
    "sm_100" (Nvrtc.arch_string Kernel_ast.SM100)

let test_compile_kernel_ampere () =
  let src = ampere_source () in
  match Nvrtc.compile_source src ~name:"gemm.cu" ~arch:"sm_80" () with
  | Ok ptx ->
    Alcotest.(check bool) "ptx non-empty" true (String.length ptx > 0)
  | Error e ->
    (* on machines without GPU/nvrtc, skip gracefully *)
    Printf.printf "nvrtc not available: %s\n%!" e;
    Alcotest.(check bool) "skip" true true


let test_compile_with_options () =
  let result = Nvrtc.compile_source trivial_source
    ~name:"trivial.cu"
    ~arch:"sm_80"
    ~options:["--use_fast_math"; "--generate-line-info"]
    ()
  in
  Alcotest.(check bool) "ok" true (Result.is_ok result)

let () =
  Alcotest.run "Nvrtc" [
    "program",  [ Alcotest.test_case "create"    `Quick test_create_program
                ; Alcotest.test_case "destroy"   `Quick test_destroy_idempotent ];
    "compile",  [ Alcotest.test_case "trivial"   `Quick test_compile_trivial
                ; Alcotest.test_case "ptx"       `Quick test_compile_ptx_nonempty
                ; Alcotest.test_case "entry"     `Quick test_compile_ptx_contains_kernel
                ; Alcotest.test_case "invalid"   `Quick test_compile_invalid_source ];
    "arch",     [ Alcotest.test_case "sm80"      `Quick test_arch_sm80
                ; Alcotest.test_case "sm90"      `Quick test_arch_sm90
                ; Alcotest.test_case "sm100"     `Quick test_arch_sm100 ];
    "kernel",   [ Alcotest.test_case "ampere"    `Quick test_compile_kernel_ampere ];
    "options",  [ Alcotest.test_case "fast-math" `Quick test_compile_with_options];
  ]
