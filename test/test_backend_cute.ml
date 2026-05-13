open Tesserae

let ampere_kernel () =
  Kernel_ast.make
    ~name:"gemm_ampere_f16"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])

let hopper_kernel () =
  Kernel_ast.make
    ~name:"gemm_hopper_bf16"
    ~arch:Kernel_ast.SM90
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 64 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("B", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32,  Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])

let blackwell_kernel () =
  Kernel_ast.make
    ~name:"gemm_blackwell_bf16"
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

let test_includes_pragma () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "pragma" true
    (contains "#pragma once" (Backend_cute.emit_includes desc))

let test_includes_cute () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "cute" true
    (contains "tensor.hpp" (Backend_cute.emit_includes desc))

let test_includes_bf16 () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "bf16" true
    (contains "cuda_bf16" (Backend_cute.emit_includes desc))

let test_helpers_smem_desc_tma () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "smem_desc" true
    (contains "make_smem_desc" (Backend_cute.emit_helpers desc))

let test_helpers_tma_wrapper () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "tma_2d" true
    (contains "tma_2d_gmem2smem" (Backend_cute.emit_helpers desc))

let test_helpers_multicast () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "multicast" true
    (contains "multicast" (Backend_cute.emit_helpers desc))

let test_helpers_empty_ampere () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  let s = Backend_cute.emit_helpers desc in
  Alcotest.(check bool) "no tma" false (contains "tma_2d_gmem2smem" s)

let test_smem_struct () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "struct" true
    (contains "struct" (Backend_cute.emit_shared_storage desc))

let test_smem_buf_a () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "smem_A" true
    (contains "smem_A" (Backend_cute.emit_shared_storage desc))

let test_smem_buf_b () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "smem_B" true
    (contains "smem_B" (Backend_cute.emit_shared_storage desc))

let test_smem_mbar_tma () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "mbar" true
    (contains "mbar" (Backend_cute.emit_shared_storage desc))

let test_smem_tmem_blackwell () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "tmem_addr" true
    (contains "tmem_addr" (Backend_cute.emit_shared_storage desc))

let test_producer_loop () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "for" true
    (contains "for" (Backend_cute.emit_producer_body desc))

let test_producer_cp_async () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "cp.async" true
    (contains "cp.async" (Backend_cute.emit_producer_body desc))

let test_producer_tma () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "tma" true
    (contains "tma" (Backend_cute.emit_producer_body desc))

let test_producer_multicast () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "multicast" true
    (contains "multicast" (Backend_cute.emit_producer_body desc))

let test_consumer_mma_sm80 () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "mma" true
    (contains "mma" (Backend_cute.emit_consumer_body desc))

let test_consumer_wgmma_sm90 () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "wgmma" true
    (contains "wgmma" (Backend_cute.emit_consumer_body desc))

let test_consumer_tcgen05_sm100 () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "tcgen05" true
    (contains "tcgen05" (Backend_cute.emit_consumer_body desc))

let test_consumer_loop () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "for" true
    (contains "for" (Backend_cute.emit_consumer_body desc))

let test_epilogue_store () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "store" true
    (contains "store" (Backend_cute.emit_epilogue_body desc))

let test_epilogue_predicate () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "predicate" true
    (contains "< M" (Backend_cute.emit_epilogue_body desc))

let test_epilogue_tcgen05_ld () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "tcgen05.ld" true
    (contains "tcgen05.ld" (Backend_cute.emit_epilogue_body desc))

let test_kernel_global () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "__global__" true
    (contains "__global__" (Backend_cute.emit_kernel_func desc))

let test_kernel_name () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "kernel name" true
    (contains "gemm_ampere_f16" (Backend_cute.emit_kernel_func desc))

let test_kernel_warp_id () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "warp_id" true
    (contains "warp_id" (Backend_cute.emit_kernel_func desc))

let test_kernel_smem () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "SharedStorage" true
    (contains "SharedStorage" (Backend_cute.emit_kernel_func desc))

let test_kernel_tmem_alloc () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "alloc" true
    (contains "alloc" (Backend_cute.emit_kernel_func desc))

let test_host_dim3 () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "dim3" true
    (contains "dim3" (Backend_cute.emit_host_launcher desc))

let test_host_launch () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "triple chevron launch" true
    (contains "<<<" (Backend_cute.emit_host_launcher desc))

let test_host_tmap () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "CUtensorMap" true
    (contains "CUtensorMap" (Backend_cute.emit_host_launcher desc))

let test_host_cluster () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "cluster" true
    (contains "cluster" (Backend_cute.emit_host_launcher desc))

let test_emit_full_ampere () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  let out = Backend_cute.emit desc in
  Alcotest.(check bool) "pragma"    true (contains "#pragma once"    out.Backend_cute.full_source);
  Alcotest.(check bool) "global"    true (contains "__global__"      out.Backend_cute.full_source);
  Alcotest.(check bool) "kernel nm" true (contains "gemm_ampere_f16" out.Backend_cute.full_source)

let test_emit_full_blackwell () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  let out = Backend_cute.emit desc in
  Alcotest.(check bool) "tcgen05"   true (contains "tcgen05"   out.Backend_cute.full_source);
  Alcotest.(check bool) "multicast" true (contains "multicast" out.Backend_cute.full_source)

let test_emit_sections_nonempty () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  let out = Backend_cute.emit desc in
  Alcotest.(check bool) "includes" true (String.length out.Backend_cute.includes > 0);
  Alcotest.(check bool) "kernel"   true (String.length out.Backend_cute.kernel_func > 0);
  Alcotest.(check bool) "producer" true (String.length out.Backend_cute.producer_body > 0);
  Alcotest.(check bool) "consumer" true (String.length out.Backend_cute.consumer_body > 0)


let () =
  Alcotest.run "Backend_cute" [
    "includes", [ Alcotest.test_case "pragma"   `Quick test_includes_pragma
                ; Alcotest.test_case "cute"     `Quick test_includes_cute
                ; Alcotest.test_case "bf16"     `Quick test_includes_bf16 ];
    "helpers",  [ Alcotest.test_case "smem_desc"`Quick test_helpers_smem_desc_tma
                ; Alcotest.test_case "tma_wrap" `Quick test_helpers_tma_wrapper
                ; Alcotest.test_case "multicast"`Quick test_helpers_multicast
                ; Alcotest.test_case "no-tma"   `Quick test_helpers_empty_ampere ];
    "smem",     [ Alcotest.test_case "struct"   `Quick test_smem_struct
                ; Alcotest.test_case "buf-a"    `Quick test_smem_buf_a
                ; Alcotest.test_case "buf-b"    `Quick test_smem_buf_b
                ; Alcotest.test_case "mbar"     `Quick test_smem_mbar_tma
                ; Alcotest.test_case "tmem"     `Quick test_smem_tmem_blackwell ];
    "producer", [ Alcotest.test_case "loop"     `Quick test_producer_loop
                ; Alcotest.test_case "cp"       `Quick test_producer_cp_async
                ; Alcotest.test_case "tma"      `Quick test_producer_tma
                ; Alcotest.test_case "mcast"    `Quick test_producer_multicast ];
    "consumer", [ Alcotest.test_case "sm80"     `Quick test_consumer_mma_sm80
                ; Alcotest.test_case "sm90"     `Quick test_consumer_wgmma_sm90
                ; Alcotest.test_case "sm100"    `Quick test_consumer_tcgen05_sm100
                ; Alcotest.test_case "loop"     `Quick test_consumer_loop ];
    "epilogue", [ Alcotest.test_case "store"    `Quick test_epilogue_store
                ; Alcotest.test_case "pred"     `Quick test_epilogue_predicate
                ; Alcotest.test_case "ld"       `Quick test_epilogue_tcgen05_ld ];
    "kernel",   [ Alcotest.test_case "global"   `Quick test_kernel_global
                ; Alcotest.test_case "name"     `Quick test_kernel_name
                ; Alcotest.test_case "warp_id"  `Quick test_kernel_warp_id
                ; Alcotest.test_case "smem"     `Quick test_kernel_smem
                ; Alcotest.test_case "alloc"    `Quick test_kernel_tmem_alloc ];
    "host",     [ Alcotest.test_case "dim3"     `Quick test_host_dim3
                ; Alcotest.test_case "launch"   `Quick test_host_launch
                ; Alcotest.test_case "tmap"     `Quick test_host_tmap
                ; Alcotest.test_case "cluster"  `Quick test_host_cluster ];
    "emit",     [ Alcotest.test_case "ampere"   `Quick test_emit_full_ampere
                ; Alcotest.test_case "blackwell"`Quick test_emit_full_blackwell
                ; Alcotest.test_case "sections" `Quick test_emit_sections_nonempty ];
  ]
