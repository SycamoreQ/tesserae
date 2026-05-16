open Tesserae
open Stdio

let ampere () =
  Kernel_ast.make
    ~name:"gemm_ampere"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])

let hopper () =
  Kernel_ast.make
    ~name:"gemm_hopper"
    ~arch:Kernel_ast.SM90
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 64 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("B", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32,  Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])

let blackwell () =
  Kernel_ast.make
    ~name:"gemm_blackwell"
    ~arch:Kernel_ast.SM100
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 256; k = 64 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("B", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32,  Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])

let contains sub str =
  let n = String.length sub and m = String.length str in
  let found = ref false in
  for i = 0 to m - n do
    if String.sub str i n = sub then found := true
  done; !found

(* ------------------------------------------------------------------ *)
(* to_source                                                           *)
(* ------------------------------------------------------------------ *)

let test_to_source_ok () =
  Alcotest.(check bool) "ok" true
    (Result.is_ok (Compile.to_source (ampere ())))

let test_to_source_name () =
  let r = Compile.to_source_exn (ampere ()) in
  Alcotest.(check string) "name" "gemm_ampere" r.Compile.kernel_name

let test_to_source_has_source () =
  let r = Compile.to_source_exn (ampere ()) in
  Alcotest.(check bool) "non-empty" true
    (String.length r.Compile.source > 0)

let test_to_source_no_ptx () =
  let r = Compile.to_source_exn (ampere ()) in
  Alcotest.(check bool) "no ptx" true
    (Option.is_none r.Compile.ptx)

let test_to_source_duration () =
  let r = Compile.to_source_exn (ampere ()) in
  Alcotest.(check bool) "duration >= 0" true
    (r.Compile.duration_ms >= 0.0)

(* ------------------------------------------------------------------ *)
(* source content                                                      *)
(* ------------------------------------------------------------------ *)

let test_source_pragma () =
  let r = Compile.to_source_exn (ampere ()) in
  Alcotest.(check bool) "pragma" true
    (contains "#pragma once" r.Compile.source)

let test_source_global () =
  let r = Compile.to_source_exn (ampere ()) in
  Alcotest.(check bool) "__global__" true
    (contains "__global__" r.Compile.source)

let test_source_kernel_name () =
  let r = Compile.to_source_exn (ampere ()) in
  Alcotest.(check bool) "kernel name" true
    (contains "gemm_ampere" r.Compile.source)

let test_source_hopper_tma () =
  let r = Compile.to_source_exn (hopper ()) in
  Alcotest.(check bool) "tma" true
    (contains "tma" r.Compile.source)

let test_source_blackwell_tcgen05 () =
  let r = Compile.to_source_exn (blackwell ()) in
  Alcotest.(check bool) "tcgen05" true
    (contains "tcgen05" r.Compile.source)

(* ------------------------------------------------------------------ *)
(* to_ptx (stub)                                                       *)
(* ------------------------------------------------------------------ *)

let test_to_ptx_ok () =
  match Compile.to_ptx (ampere ()) with
  | Ok _    -> Alcotest.(check bool) "ok" true true
  | Error e ->
    let msg = match e with
      | Compile.NvrtcError s  -> "NvrtcError: " ^ s
      | Compile.LowerError _  -> "LowerError"
      | Compile.LaunchError s -> "LaunchError: " ^ s
    in
    Printf.printf "COMPILE ERROR: %s\n%!" msg;
    Alcotest.(check bool) "ok" true false

let test_to_ptx_has_ptx () =
  match Compile.to_ptx (ampere ()) with
  | Ok r -> Alcotest.(check bool) "some" true (Option.is_some r.Compile.ptx)
  | Error _ -> Alcotest.fail "expected ok"

(* ------------------------------------------------------------------ *)
(* write_source                                                        *)
(* ------------------------------------------------------------------ *)

let test_write_source () =
  let path = "/tmp/tesserae_test_gemm.cuh" in
  Compile.write_source (ampere ()) path;
  let content = In_channel.read_all path in
  Alcotest.(check bool) "written" true
    (contains "__global__" content)

(* ------------------------------------------------------------------ *)
(* all three archs                                                     *)
(* ------------------------------------------------------------------ *)

let test_all_archs () =
  let kernels = [ampere (); hopper (); blackwell ()] in
  List.iter (fun k ->
    let r = Compile.to_source_exn k in
    Alcotest.(check bool) "non-empty" true
      (String.length r.Compile.source > 0)
  ) kernels

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Compile" [
    "to_source", [ Alcotest.test_case "ok"       `Quick test_to_source_ok
                 ; Alcotest.test_case "name"     `Quick test_to_source_name
                 ; Alcotest.test_case "source"   `Quick test_to_source_has_source
                 ; Alcotest.test_case "no-ptx"   `Quick test_to_source_no_ptx
                 ; Alcotest.test_case "duration" `Quick test_to_source_duration ];
    "content",   [ Alcotest.test_case "pragma"   `Quick test_source_pragma
                 ; Alcotest.test_case "global"   `Quick test_source_global
                 ; Alcotest.test_case "name"     `Quick test_source_kernel_name
                 ; Alcotest.test_case "hopper"   `Quick test_source_hopper_tma
                 ; Alcotest.test_case "blkwll"   `Quick test_source_blackwell_tcgen05 ];
    "to_ptx",    [ Alcotest.test_case "ok"       `Quick test_to_ptx_ok
                 ; Alcotest.test_case "has-ptx"  `Quick test_to_ptx_has_ptx ];
    "write",     [ Alcotest.test_case "file"     `Quick test_write_source ];
    "all_archs", [ Alcotest.test_case "three"    `Quick test_all_archs ];
  ]
