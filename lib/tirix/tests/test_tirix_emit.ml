open Tesserae_core
open Tesserae_atoms
open Tesserae_pipeline
open Tesserae_kernel
open Tesserae_tirix

(* ------------------------------------------------------------------ *)
(* helpers                                                             *)
(* ------------------------------------------------------------------ *)

let ampere_kernel () =
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

let hopper_kernel () =
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

let blackwell_kernel () =
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

let lower_and_emit k =
  let (Lower.Pack desc) = Lower.lower_exn k in
  let tir = Kernel_desc_to_tir.lower desc in
  Tir_emit.emit tir

let contains sub str =
  let n = String.length sub and m = String.length str in
  let found = ref false in
  for i = 0 to m - n do
    if String.sub str i n = sub then found := true
  done; !found

(* ------------------------------------------------------------------ *)
(* emit_scalar_ty                                                      *)
(* ------------------------------------------------------------------ *)

let test_scalar_u8 () =
  Alcotest.(check string) "u8" "uint8_t"
    (Tir_emit.emit_scalar_ty Tir.U8)

let test_scalar_s32 () =
  Alcotest.(check string) "s32" "int32_t"
    (Tir_emit.emit_scalar_ty Tir.S32)

let test_scalar_f16 () =
  Alcotest.(check string) "f16" "__half"
    (Tir_emit.emit_scalar_ty Tir.F16)

let test_scalar_f32 () =
  Alcotest.(check string) "f32" "float"
    (Tir_emit.emit_scalar_ty Tir.F32)

let test_scalar_bf16 () =
  Alcotest.(check string) "bf16" "__nv_bfloat16"
    (Tir_emit.emit_scalar_ty Tir.BF16)

let test_scalar_bool () =
  Alcotest.(check string) "bool" "bool"
    (Tir_emit.emit_scalar_ty Tir.Bool)

(* ------------------------------------------------------------------ *)
(* emit_expr                                                           *)
(* ------------------------------------------------------------------ *)

let test_expr_const_s32 () =
  Alcotest.(check string) "const s32" "42"
    (Tir_emit.emit_expr (Tir.Const (Tir.S32, 42l)))

let test_expr_const_bool_true () =
  Alcotest.(check string) "true" "true"
    (Tir_emit.emit_expr (Tir.Const (Tir.Bool, true)))

let test_expr_builtin_threadidx () =
  Alcotest.(check string) "threadIdx.x" "threadIdx.x"
    (Tir_emit.emit_expr (Tir.Builtin (Tir.ThreadIdx Tir.X)))

let test_expr_builtin_warpid () =
  Alcotest.(check string) "warp_id" "(threadIdx.x / 32)"
    (Tir_emit.emit_expr (Tir.Builtin Tir.WarpId))

let test_expr_binop_add () =
  let e = Tir.Binop (Tir.Add,
    Tir.Const (Tir.S32, 1l),
    Tir.Const (Tir.S32, 2l)) in
  Alcotest.(check string) "add" "(1 + 2)"
    (Tir_emit.emit_expr e)

let test_expr_binop_logical_and () =
  let e = Tir.Binop (Tir.And,
    Tir.Const (Tir.Bool, true),
    Tir.Const (Tir.Bool, false)) in
  Alcotest.(check bool) "logical and has &&" true
    (contains "&&" (Tir_emit.emit_expr e))

let test_expr_binop_bitand_not_logical () =
  let e = Tir.Binop (Tir.BitAnd,
    Tir.Const (Tir.U32, 0xffl),
    Tir.Const (Tir.U32, 0x0fl)) in
  let s = Tir_emit.emit_expr e in
  Alcotest.(check bool) "bitand has single &" true (contains "&" s);
  Alcotest.(check bool) "bitand not &&" false (contains "&&" s)

let test_expr_cast () =
  let e = Tir.Cast (Tir.U32, Tir.Const (Tir.S32, 42l)) in
  Alcotest.(check bool) "cast has type" true
    (contains "uint32_t" (Tir_emit.emit_expr e))

let test_expr_addr_conv () =
  let e = Tir.AddrConv (Tir.GenericToShared,
    Tir.Const (Tir.U64, 0L)) in
  Alcotest.(check bool) "addr conv" true
    (contains "__cvta_generic_to_shared" (Tir_emit.emit_expr e))

(* ------------------------------------------------------------------ *)
(* emit_barrier                                                        *)
(* ------------------------------------------------------------------ *)

let test_barrier_cta_sync () =
  Alcotest.(check bool) "__syncthreads" true
    (contains "__syncthreads"
      (Tir_emit.emit_barrier Tir.CtaSync))

let test_barrier_mbar_init () =
  let v = { Tir.var_name="full_mbar"; var_id=0
          ; var_type=Tir.Scalar Tir.U64; var_mutable=false } in
  Alcotest.(check bool) "mbarrier.init" true
    (contains "mbarrier.init"
      (Tir_emit.emit_barrier (Tir.MbarInit { mbar=v; count=1 })))

let test_barrier_cp_async_wait () =
  Alcotest.(check bool) "cp.async.wait_all" true
    (contains "cp.async.wait_all"
      (Tir_emit.emit_barrier Tir.CpAsyncWaitAll))

let test_barrier_cluster_arrive () =
  Alcotest.(check bool) "cluster arrive" true
    (contains "barrier.cluster.arrive"
      (Tir_emit.emit_barrier Tir.ClusterArrive))

(* ------------------------------------------------------------------ *)
(* emit_stmt                                                           *)
(* ------------------------------------------------------------------ *)

let test_stmt_slet () =
  let v = { Tir.var_name="x"; var_id=0
          ; var_type=Tir.Scalar Tir.S32; var_mutable=false } in
  let s = Tir_emit.emit_stmt
    (Tir.SLet (v, Tir.Expr (Tir.Const (Tir.S32, 0l)))) in
  Alcotest.(check bool) "const decl" true (contains "const" s);
  Alcotest.(check bool) "var name"   true (contains "x" s)

let test_stmt_sletmut () =
  let v = { Tir.var_name="y"; var_id=0
          ; var_type=Tir.Scalar Tir.S32; var_mutable=true } in
  let s = Tir_emit.emit_stmt
    (Tir.SLetMut (v, Tir.Expr (Tir.Const (Tir.S32, 1l)))) in
  Alcotest.(check bool) "no const" false (contains "const" s);
  Alcotest.(check bool) "var name"  true  (contains "y" s)

let test_stmt_sfor () =
  let v = { Tir.var_name="i"; var_id=0
          ; var_type=Tir.Scalar Tir.S32; var_mutable=true } in
  let s = Tir_emit.emit_stmt (Tir.SFor {
    var    = v;
    start  = Tir.Const (Tir.S32, 0l);
    stop   = Tir.Const (Tir.S32, 4l);
    step   = Tir.Const (Tir.S32, 1l);
    dir    = Tir.Upto;
    unroll = false;
    body   = [];
  }) in
  Alcotest.(check bool) "for loop" true (contains "for" s);
  Alcotest.(check bool) "var i"    true (contains "i" s)

let test_stmt_sfor_unroll () =
  let v = { Tir.var_name="i"; var_id=0
          ; var_type=Tir.Scalar Tir.S32; var_mutable=true } in
  let s = Tir_emit.emit_stmt (Tir.SFor {
    var    = v;
    start  = Tir.Const (Tir.S32, 0l);
    stop   = Tir.Const (Tir.S32, 4l);
    step   = Tir.Const (Tir.S32, 1l);
    dir    = Tir.Upto;
    unroll = true;
    body   = [];
  }) in
  Alcotest.(check bool) "pragma unroll" true
    (contains "#pragma unroll" s)

let test_stmt_sif () =
  let s = Tir_emit.emit_stmt
    (Tir.SIf (Tir.Const (Tir.Bool, true), [], [])) in
  Alcotest.(check bool) "if stmt" true (contains "if" s)

let test_stmt_pipeline () =
  let s = Tir_emit.emit_stmt (Tir.SPipeline {
    stages   = 4;
    prologue = [];
    mainloop = [];
    epilogue = [];
  }) in
  Alcotest.(check bool) "pipeline comment" true
    (contains "pipeline" s)

let test_stmt_warp_group_producer () =
  let s = Tir_emit.emit_stmt
    (Tir.SWarpGroup (Cluster.Producer, [])) in
  Alcotest.(check bool) "producer comment" true
    (contains "producer" s)

(* ------------------------------------------------------------------ *)
(* emit_shared_storage                                                 *)
(* ------------------------------------------------------------------ *)

let test_shared_storage_struct () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "SharedStorage" true
    (contains "SharedStorage" out.Backend_cute.shared_storage)

let test_shared_storage_smem_a () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "smem_A" true
    (contains "smem_A" out.Backend_cute.shared_storage)

let test_shared_storage_smem_b () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "smem_B" true
    (contains "smem_B" out.Backend_cute.shared_storage)

let test_shared_storage_mbar_tma () =
  let out = lower_and_emit (hopper_kernel ()) in
  Alcotest.(check bool) "mbar" true
    (contains "mbar" out.Backend_cute.shared_storage)

let test_shared_storage_tmem_blackwell () =
  let out = lower_and_emit (blackwell_kernel ()) in
  Alcotest.(check bool) "tmem_addr" true
    (contains "tmem_addr" out.Backend_cute.shared_storage)

(* ------------------------------------------------------------------ *)
(* emit_kernel_func                                                    *)
(* ------------------------------------------------------------------ *)

let test_kernel_func_global () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "__global__" true
    (contains "__global__" out.Backend_cute.kernel_func)

let test_kernel_func_name () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "kernel name" true
    (contains "gemm_ampere" out.Backend_cute.kernel_func)

let test_kernel_func_warp_id () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "warp_id" true
    (contains "warp_id" out.Backend_cute.kernel_func)

let test_kernel_func_shared_storage () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "SharedStorage" true
    (contains "SharedStorage" out.Backend_cute.kernel_func)

let test_kernel_func_cluster_attr_blackwell () =
  let out = lower_and_emit (blackwell_kernel ()) in
  Alcotest.(check bool) "cluster attr" true
    (contains "__cluster_dims__" out.Backend_cute.kernel_func)

let test_kernel_func_tmem_alloc_blackwell () =
  let out = lower_and_emit (blackwell_kernel ()) in
  Alcotest.(check bool) "tcgen05.alloc" true
    (contains "tcgen05.alloc" out.Backend_cute.kernel_func)

(* ------------------------------------------------------------------ *)
(* emit_host_launcher                                                  *)
(* ------------------------------------------------------------------ *)

let test_host_launcher_dim3 () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "dim3" true
    (contains "dim3" out.Backend_cute.host_launcher)

let test_host_launcher_chevron_ampere () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "<<<" true
    (contains "<<<" out.Backend_cute.host_launcher)

let test_host_launcher_kernel_ex_blackwell () =
  let out = lower_and_emit (blackwell_kernel ()) in
  Alcotest.(check bool) "cudaLaunchKernelEx" true
    (contains "cudaLaunchKernelEx" out.Backend_cute.host_launcher)

let test_host_launcher_tma_map () =
  let out = lower_and_emit (hopper_kernel ()) in
  Alcotest.(check bool) "CUtensorMap" true
    (contains "CUtensorMap" out.Backend_cute.host_launcher)

(* ------------------------------------------------------------------ *)
(* emit (full output)                                                  *)
(* ------------------------------------------------------------------ *)

let test_emit_full_ampere () =
  let out = lower_and_emit (ampere_kernel ()) in
  Alcotest.(check bool) "pragma"     true
    (contains "#pragma once"  out.Backend_cute.full_source);
  Alcotest.(check bool) "global"     true
    (contains "__global__"    out.Backend_cute.full_source);
  Alcotest.(check bool) "kernel nm"  true
    (contains "gemm_ampere"   out.Backend_cute.full_source)

let test_emit_full_blackwell () =
  let out = lower_and_emit (blackwell_kernel ()) in
  Alcotest.(check bool) "tcgen05"    true
    (contains "tcgen05"    out.Backend_cute.full_source);
  Alcotest.(check bool) "multicast"  true
    (contains "multicast"  out.Backend_cute.full_source)

let test_emit_sections_nonempty () =
  let out = lower_and_emit (hopper_kernel ()) in
  Alcotest.(check bool) "includes" true
    (String.length out.Backend_cute.includes > 0);
  Alcotest.(check bool) "kernel"   true
    (String.length out.Backend_cute.kernel_func > 0);
  Alcotest.(check bool) "launcher" true
    (String.length out.Backend_cute.host_launcher > 0)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Tir_emit" [
    "scalar",   [ Alcotest.test_case "u8"        `Quick test_scalar_u8
                ; Alcotest.test_case "s32"       `Quick test_scalar_s32
                ; Alcotest.test_case "f16"       `Quick test_scalar_f16
                ; Alcotest.test_case "f32"       `Quick test_scalar_f32
                ; Alcotest.test_case "bf16"      `Quick test_scalar_bf16
                ; Alcotest.test_case "bool"      `Quick test_scalar_bool ];
    "expr",     [ Alcotest.test_case "const-s32" `Quick test_expr_const_s32
                ; Alcotest.test_case "const-bool"`Quick test_expr_const_bool_true
                ; Alcotest.test_case "threadidx" `Quick test_expr_builtin_threadidx
                ; Alcotest.test_case "warpid"    `Quick test_expr_builtin_warpid
                ; Alcotest.test_case "binop-add" `Quick test_expr_binop_add
                ; Alcotest.test_case "logical-and"`Quick test_expr_binop_logical_and
                ; Alcotest.test_case "bitand"    `Quick test_expr_binop_bitand_not_logical
                ; Alcotest.test_case "cast"      `Quick test_expr_cast
                ; Alcotest.test_case "addrconv"  `Quick test_expr_addr_conv ];
    "barrier",  [ Alcotest.test_case "cta-sync"  `Quick test_barrier_cta_sync
                ; Alcotest.test_case "mbar-init" `Quick test_barrier_mbar_init
                ; Alcotest.test_case "cp-wait"   `Quick test_barrier_cp_async_wait
                ; Alcotest.test_case "cluster"   `Quick test_barrier_cluster_arrive ];
    "stmt",     [ Alcotest.test_case "slet"      `Quick test_stmt_slet
                ; Alcotest.test_case "sletmut"   `Quick test_stmt_sletmut
                ; Alcotest.test_case "sfor"      `Quick test_stmt_sfor
                ; Alcotest.test_case "unroll"    `Quick test_stmt_sfor_unroll
                ; Alcotest.test_case "sif"       `Quick test_stmt_sif
                ; Alcotest.test_case "pipeline"  `Quick test_stmt_pipeline
                ; Alcotest.test_case "warpgroup" `Quick test_stmt_warp_group_producer ];
    "smem",     [ Alcotest.test_case "struct"    `Quick test_shared_storage_struct
                ; Alcotest.test_case "smem-a"    `Quick test_shared_storage_smem_a
                ; Alcotest.test_case "smem-b"    `Quick test_shared_storage_smem_b
                ; Alcotest.test_case "mbar"      `Quick test_shared_storage_mbar_tma
                ; Alcotest.test_case "tmem"      `Quick test_shared_storage_tmem_blackwell ];
    "kernel",   [ Alcotest.test_case "global"    `Quick test_kernel_func_global
                ; Alcotest.test_case "name"      `Quick test_kernel_func_name
                ; Alcotest.test_case "warp-id"   `Quick test_kernel_func_warp_id
                ; Alcotest.test_case "shared"    `Quick test_kernel_func_shared_storage
                ; Alcotest.test_case "cluster"   `Quick test_kernel_func_cluster_attr_blackwell
                ; Alcotest.test_case "tmem-alloc"`Quick test_kernel_func_tmem_alloc_blackwell ];
    "host",     [ Alcotest.test_case "dim3"      `Quick test_host_launcher_dim3
                ; Alcotest.test_case "chevron"   `Quick test_host_launcher_chevron_ampere
                ; Alcotest.test_case "kernel-ex" `Quick test_host_launcher_kernel_ex_blackwell
                ; Alcotest.test_case "tma-map"   `Quick test_host_launcher_tma_map ];
    "emit",     [ Alcotest.test_case "ampere"    `Quick test_emit_full_ampere
                ; Alcotest.test_case "blackwell" `Quick test_emit_full_blackwell
                ; Alcotest.test_case "sections"  `Quick test_emit_sections_nonempty ];
  ]
