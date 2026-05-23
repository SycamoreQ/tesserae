open Tesserae_core
open Tesserae_pipeline
open Tesserae_kernel
open Tesserae_tirix
open Tirix

let contains sub str =
  let n = String.length sub and m = String.length str in
  let found = ref false in
  for i = 0 to m - n do
    if String.sub str i n = sub then found := true
  done; !found

let fresh_var name ty =
  { var_name = name; var_id = 0
  ; var_type = Scalar ty; var_mutable = false }

let fresh_mut_var name ty =
  { var_name = name; var_id = 0
  ; var_type = Scalar ty; var_mutable = true }

let flat_layout () = Layout.make (Modes.Int 1) (Modes.Int 1)

let make_tensor name elem space =
  Tensor {
    tensor_name      = name
  ; tensor_id        = Type_id.create ()
  ; tensor_elem_type = elem
  ; tensor_memspace  = space
  ; tensor_layout    = flat_layout ()
  ; tensor_swizzle   = Swizzle.make 0 0 0
  }

let minimal_cluster () =
  Cluster.make { Cluster.x=1; y=1; z=1 } 4
    [ (0, Cluster.Producer); (1, Cluster.Consumer)
    ; (2, Cluster.Epilogue); (3, Cluster.Epilogue) ]

let minimal_tirix () : tirix =
  { name           = "test_kernel"
  ; family         = Kernel_desc.Ampere
  ; params         = []
  ; tensors        = []
  ; smem_bytes     = 0
  ; pipeline_depth = 4
  ; bm             = 128
  ; bn             = 128
  ; bk             = 32
  ; cluster        = minimal_cluster ()
  ; body           = []
  ; helpers        = []
  }

(* ------------------------------------------------------------------ *)
(* pp_scalar_ty                                                        *)
(* ------------------------------------------------------------------ *)

let test_pp_scalar_u8   () = Alcotest.(check string) "u8"   "uint8_t"       (Tirix_pp.pp_scalar_ty U8)
let test_pp_scalar_u32  () = Alcotest.(check string) "u32"  "uint32_t"      (Tirix_pp.pp_scalar_ty U32)
let test_pp_scalar_s32  () = Alcotest.(check string) "s32"  "int32_t"       (Tirix_pp.pp_scalar_ty S32)
let test_pp_scalar_u64  () = Alcotest.(check string) "u64"  "uint64_t"      (Tirix_pp.pp_scalar_ty U64)
let test_pp_scalar_f16  () = Alcotest.(check string) "f16"  "__half"        (Tirix_pp.pp_scalar_ty F16)
let test_pp_scalar_f32  () = Alcotest.(check string) "f32"  "float"         (Tirix_pp.pp_scalar_ty F32)
let test_pp_scalar_bf16 () = Alcotest.(check string) "bf16" "__nv_bfloat16" (Tirix_pp.pp_scalar_ty BF16)
let test_pp_scalar_bool () = Alcotest.(check string) "bool" "bool"          (Tirix_pp.pp_scalar_ty Bool)
let test_pp_scalar_ptr  () = Alcotest.(check string) "ptr"  "uint64_t"      (Tirix_pp.pp_scalar_ty Ptr)

(* ------------------------------------------------------------------ *)
(* pp_arith_op                                                         *)
(* ------------------------------------------------------------------ *)

let test_pp_arith_add () =
  Alcotest.(check string) "add" "+" (Tirix_pp.pp_arith_op Arith.Add)

let test_pp_arith_sub () =
  Alcotest.(check string) "sub" "-" (Tirix_pp.pp_arith_op Arith.Sub)

let test_pp_arith_mul () =
  Alcotest.(check string) "mul" "*" (Tirix_pp.pp_arith_op Arith.Mul)

let test_pp_arith_div () =
  Alcotest.(check string) "div" "/" (Tirix_pp.pp_arith_op Arith.Div)

let test_pp_arith_mod () =
  Alcotest.(check string) "mod" "%" (Tirix_pp.pp_arith_op Arith.Mod)

(* ------------------------------------------------------------------ *)
(* pp_cmp_op                                                           *)
(* ------------------------------------------------------------------ *)

let test_pp_cmp_eq () =
  Alcotest.(check string) "eq" "==" (Tirix_pp.pp_cmp_op Cmp.Eq)

let test_pp_cmp_ne () =
  Alcotest.(check string) "ne" "!=" (Tirix_pp.pp_cmp_op Cmp.Ne)

let test_pp_cmp_lt () =
  Alcotest.(check string) "lt" "<" (Tirix_pp.pp_cmp_op Cmp.Lt)

let test_pp_cmp_le () =
  Alcotest.(check string) "le" "<=" (Tirix_pp.pp_cmp_op Cmp.Le)

let test_pp_cmp_gt () =
  Alcotest.(check string) "gt" ">" (Tirix_pp.pp_cmp_op Cmp.Gt)

let test_pp_cmp_ge () =
  Alcotest.(check string) "ge" ">=" (Tirix_pp.pp_cmp_op Cmp.Ge)

(* ------------------------------------------------------------------ *)
(* pp_logic_op                                                         *)
(* ------------------------------------------------------------------ *)

let test_pp_logic_and () =
  Alcotest.(check string) "and" "&&" (Tirix_pp.pp_logic_op Logic.And)

let test_pp_logic_or () =
  Alcotest.(check string) "or" "||" (Tirix_pp.pp_logic_op Logic.Or)

(* ------------------------------------------------------------------ *)
(* pp_unop                                                             *)
(* ------------------------------------------------------------------ *)

let test_pp_unop_neg () =
  Alcotest.(check string) "neg" "-" (Tirix_pp.pp_unop Unop.Neg)

let test_pp_unop_not () =
  Alcotest.(check string) "not" "!" (Tirix_pp.pp_unop Unop.Not)

let test_pp_unop_bitnot () =
  Alcotest.(check string) "bitnot" "~" (Tirix_pp.pp_unop Unop.BitNot)

(* ------------------------------------------------------------------ *)
(* pp_expr                                                             *)
(* ------------------------------------------------------------------ *)

let test_pp_expr_const_s32 () =
  Alcotest.(check bool) "42" true
    (contains "42" (Tirix_pp.pp_expr (Const (S32, 42l))))

let test_pp_expr_const_bool_true () =
  Alcotest.(check bool) "true" true
    (contains "true" (Tirix_pp.pp_expr (Const (Bool, true))))

let test_pp_expr_const_f32 () =
  Alcotest.(check bool) "f32" true
    (contains "f" (Tirix_pp.pp_expr (Const (F32, 1.0))))

let test_pp_expr_var () =
  let v = fresh_var "warp_id" S32 in
  Alcotest.(check bool) "var name" true
    (contains "warp_id" (Tirix_pp.pp_expr (Var v)))

let test_pp_expr_builtin_threadidx_x () =
  Alcotest.(check bool) "threadIdx.x" true
    (contains "threadIdx.x" (Tirix_pp.pp_expr (Builtin (ThreadIdx X))))

let test_pp_expr_builtin_threadidx_y () =
  Alcotest.(check bool) "threadIdx.y" true
    (contains "threadIdx.y" (Tirix_pp.pp_expr (Builtin (ThreadIdx Y))))

let test_pp_expr_builtin_blockidx_x () =
  Alcotest.(check bool) "blockIdx.x" true
    (contains "blockIdx.x" (Tirix_pp.pp_expr (Builtin (BlockIdx X))))

let test_pp_expr_builtin_warpid () =
  Alcotest.(check bool) "warp_id" true
    (contains "warp_id" (Tirix_pp.pp_expr (Builtin WarpId)))

let test_pp_expr_builtin_laneid () =
  Alcotest.(check bool) "lane_id" true
    (contains "lane_id" (Tirix_pp.pp_expr (Builtin LaneId)))

let test_pp_expr_arith_add () =
  let e = Arith (Add, Const (S32, 1l), Const (S32, 2l)) in
  let s = Tirix_pp.pp_expr e in
  Alcotest.(check bool) "+" true (contains "+" s);
  Alcotest.(check bool) "1" true (contains "1" s);
  Alcotest.(check bool) "2" true (contains "2" s)

let test_pp_expr_arith_mod () =
  let e = Arith (Mod, Const (S32, 5l), Const (S32, 4l)) in
  Alcotest.(check bool) "%" true
    (contains "%" (Tirix_pp.pp_expr e))

let test_pp_expr_cmp_eq () =
  let e = Cmp (Eq, Const (S32, 0l), Const (S32, 1l)) in
  Alcotest.(check bool) "==" true
    (contains "==" (Tirix_pp.pp_expr e))

let test_pp_expr_cmp_lt () =
  let e = Cmp (Lt, Const (S32, 0l), Const (S32, 10l)) in
  Alcotest.(check bool) "<" true
    (contains "<" (Tirix_pp.pp_expr e))

let test_pp_expr_logic_and () =
  let e = Logic (And, Const (Bool, true), Const (Bool, false)) in
  Alcotest.(check bool) "&&" true
    (contains "&&" (Tirix_pp.pp_expr e))

let test_pp_expr_logic_or () =
  let e = Logic (Or, Const (Bool, true), Const (Bool, false)) in
  Alcotest.(check bool) "||" true
    (contains "||" (Tirix_pp.pp_expr e))

let test_pp_expr_cast () =
  let e = Cast (U32, Const (S32, 42l)) in
  Alcotest.(check bool) "uint32_t" true
    (contains "uint32_t" (Tirix_pp.pp_expr e))

let test_pp_expr_addrconv_shared () =
  let e = AddrConv (GenericToShared, Const (U64, 0L)) in
  Alcotest.(check bool) "cvta_generic_to_shared" true
    (contains "__cvta_generic_to_shared" (Tirix_pp.pp_expr e))

(* ------------------------------------------------------------------ *)
(* pp_barrier                                                          *)
(* ------------------------------------------------------------------ *)

let test_pp_barrier_cta_sync () =
  Alcotest.(check bool) "__syncthreads" true
    (contains "__syncthreads" (Tirix_pp.pp_barrier CtaSync))

let test_pp_barrier_warp_sync () =
  Alcotest.(check bool) "__syncwarp" true
    (contains "__syncwarp" (Tirix_pp.pp_barrier WarpSync))

let test_pp_barrier_mem_fence () =
  Alcotest.(check bool) "__threadfence" true
    (contains "__threadfence" (Tirix_pp.pp_barrier MemFence))

let test_pp_barrier_mbar_init () =
  let v = fresh_var "mbar" U64 in
  Alcotest.(check bool) "mbarrier.init" true
    (contains "mbarrier.init"
      (Tirix_pp.pp_barrier (MbarInit { mbar = v; count = 1 })))

let test_pp_barrier_mbar_arrive_expect () =
  let v = fresh_var "mbar" U64 in
  Alcotest.(check bool) "arrive.expect_tx" true
    (contains "arrive.expect_tx"
      (Tirix_pp.pp_barrier
        (MbarArriveExpect { mbar = v; bytes = Const (S32, 128l) })))

let test_pp_barrier_mbar_wait_parity () =
  let v = fresh_var "mbar" U64 in
  Alcotest.(check bool) "wait.parity" true
    (contains "wait.parity"
      (Tirix_pp.pp_barrier
        (MbarWaitParity { mbar = v; phase = Const (S32, 0l) })))

let test_pp_barrier_mbar_arrive () =
  let v = fresh_var "mbar" U64 in
  Alcotest.(check bool) "mbarrier.arrive" true
    (contains "mbarrier.arrive"
      (Tirix_pp.pp_barrier (MbarArrive { mbar = v })))

let test_pp_barrier_cluster_arrive () =
  Alcotest.(check bool) "cluster.arrive" true
    (contains "barrier.cluster.arrive"
      (Tirix_pp.pp_barrier ClusterArrive))

let test_pp_barrier_cluster_wait () =
  Alcotest.(check bool) "cluster.wait" true
    (contains "barrier.cluster.wait"
      (Tirix_pp.pp_barrier ClusterWait))

let test_pp_barrier_cp_async_commit () =
  Alcotest.(check bool) "commit_group" true
    (contains "commit_group"
      (Tirix_pp.pp_barrier CpAsyncCommitGroup))

let test_pp_barrier_cp_async_wait_all () =
  Alcotest.(check bool) "wait_all" true
    (contains "wait_all"
      (Tirix_pp.pp_barrier CpAsyncWaitAll))

let test_pp_barrier_tcgen05_wait () =
  Alcotest.(check bool) "tcgen05.wait" true
    (contains "tcgen05.wait"
      (Tirix_pp.pp_barrier Tcgen05Wait))

let test_pp_barrier_tcgen05_fence () =
  Alcotest.(check bool) "tcgen05.fence" true
    (contains "tcgen05.fence"
      (Tirix_pp.pp_barrier Tcgen05Fence))

(* ------------------------------------------------------------------ *)
(* pp_op                                                               *)
(* ------------------------------------------------------------------ *)

let test_pp_op_copy_cp_async () =
  let src = make_tensor "A"      Elemtype.Float16 Memspace.Global in
  let dst = make_tensor "smem_A" Elemtype.Float16 Memspace.Shared in
  let s = Tirix_pp.pp_op (Copy {
    copy_kind  = CpAsync
  ; src_tensor = src; dst_tensor = dst
  ; pred_expr  = None; mbar_var  = None
  }) in
  Alcotest.(check bool) "cp.async"  true (contains "cp.async" s);
  Alcotest.(check bool) "src A"     true (contains "A"        s);
  Alcotest.(check bool) "dst smem_A" true (contains "smem_A"  s)

let test_pp_op_copy_tma () =
  let src = make_tensor "A"      Elemtype.Float16 Memspace.Global in
  let dst = make_tensor "smem_A" Elemtype.Float16 Memspace.Shared in
  let s = Tirix_pp.pp_op (Copy {
    copy_kind  = TmaLoad
  ; src_tensor = src; dst_tensor = dst
  ; pred_expr  = None; mbar_var  = None
  }) in
  Alcotest.(check bool) "tma.load" true (contains "tma" s)

let test_pp_op_copy_multicast () =
  let src = make_tensor "A"      Elemtype.Float16 Memspace.Global in
  let dst = make_tensor "smem_A" Elemtype.Float16 Memspace.Shared in
  let s = Tirix_pp.pp_op (Copy {
    copy_kind  = TmaMulticast
  ; src_tensor = src; dst_tensor = dst
  ; pred_expr  = None; mbar_var  = None
  }) in
  Alcotest.(check bool) "multicast" true (contains "multicast" s)

let test_pp_op_mma_sm80 () =
  let a = make_tensor "smem_A" Elemtype.Float16 Memspace.Shared in
  let b = make_tensor "smem_B" Elemtype.Float16 Memspace.Shared in
  let c = make_tensor "acc"    Elemtype.Float32 Memspace.Register in
  let s = Tirix_pp.pp_op (Mma {
    mma_kind = Sm80Mma; tensor_a = a; tensor_b = b; tensor_c = c
  ; smem_desc_a = None; smem_desc_b = None; accum_flag = true
  }) in
  Alcotest.(check bool) "SM80" true (contains "SM80" s)

let test_pp_op_mma_wgmma () =
  let a = make_tensor "smem_A" Elemtype.Bfloat16 Memspace.Shared in
  let b = make_tensor "smem_B" Elemtype.Bfloat16 Memspace.Shared in
  let c = make_tensor "acc"    Elemtype.Float32  Memspace.Register in
  let s = Tirix_pp.pp_op (Mma {
    mma_kind = Sm90Wgmma; tensor_a = a; tensor_b = b; tensor_c = c
  ; smem_desc_a = None; smem_desc_b = None; accum_flag = true
  }) in
  Alcotest.(check bool) "SM90" true (contains "SM90" s)

let test_pp_op_mma_tcgen05 () =
  let a = make_tensor "smem_A" Elemtype.Bfloat16 Memspace.Shared in
  let b = make_tensor "smem_B" Elemtype.Bfloat16 Memspace.Shared in
  let c = make_tensor "acc"    Elemtype.Float32  Memspace.Register in
  let s = Tirix_pp.pp_op (Mma {
    mma_kind = Sm100Tcgen05; tensor_a = a; tensor_b = b; tensor_c = c
  ; smem_desc_a = None; smem_desc_b = None; accum_flag = true
  }) in
  Alcotest.(check bool) "SM100" true (contains "SM100" s)

let test_pp_op_barrier_cta () =
  let s = Tirix_pp.pp_op (Barrier CtaSync) in
  Alcotest.(check bool) "syncthreads" true (contains "__syncthreads" s)

let test_pp_op_tmem_alloc () =
  let v = fresh_var "tmem_addr" U32 in
  let s = Tirix_pp.pp_op (TmemAlloc { addr_var = v; col_count = 256 }) in
  Alcotest.(check bool) "tmem.alloc" true (contains "tmem.alloc" s);
  Alcotest.(check bool) "256"        true (contains "256" s)

let test_pp_op_tmem_dealloc () =
  let v = fresh_var "tmem_addr" U32 in
  let s = Tirix_pp.pp_op (TmemDealloc { addr_var = v; col_count = 128 }) in
  Alcotest.(check bool) "tmem.dealloc" true (contains "tmem.dealloc" s)

let test_pp_op_tmem_commit () =
  let v = fresh_var "mbar" U64 in
  let s = Tirix_pp.pp_op (TmemCommit { mbar_var = v; cta_mask = Some 3 }) in
  Alcotest.(check bool) "tmem.commit" true (contains "tmem.commit" s)

let test_pp_op_smem_desc_init () =
  let dv = fresh_var "desc" U64 in
  let s = Tirix_pp.pp_op (SmemDescInit {
    desc_var    = dv
  ; ptr_expr    = Const (U64, 0L)
  ; leading_dim = 128
  ; stride      = 64
  ; swizzle     = Swizzle.make 3 4 3
  }) in
  Alcotest.(check bool) "desc var" true (contains "desc" s)

(* ------------------------------------------------------------------ *)
(* pp_stmt                                                             *)
(* ------------------------------------------------------------------ *)

let test_pp_stmt_slet () =
  let v = fresh_var "x" S32 in
  let s = Tirix_pp.pp_stmt
    (SLet (v, Expr (Const (S32, 0l)))) in
  Alcotest.(check bool) "let"    true (contains "let"    s);
  Alcotest.(check bool) "x"      true (contains "x"      s);
  Alcotest.(check bool) "int32"  true (contains "int32"  s)

let test_pp_stmt_sletmut () =
  let v = fresh_mut_var "k" S32 in
  let s = Tirix_pp.pp_stmt
    (SLetMut (v, Expr (Const (S32, 0l)))) in
  Alcotest.(check bool) "mut"    true (contains "mut" s);
  Alcotest.(check bool) "k"      true (contains "k"   s)

let test_pp_stmt_sassign () =
  let v = fresh_mut_var "x" S32 in
  let s = Tirix_pp.pp_stmt
    (SAssign (v, Expr (Const (S32, 1l)))) in
  Alcotest.(check bool) "x"      true (contains "x"   s);
  Alcotest.(check bool) "="      true (contains "="   s)

let test_pp_stmt_sfor () =
  let v = fresh_mut_var "i" S32 in
  let s = Tirix_pp.pp_stmt (SFor {
    var = v; start = Const (S32, 0l); stop = Const (S32, 4l)
  ; step = Const (S32, 1l); dir = Upto; unroll = false; body = []
  }) in
  Alcotest.(check bool) "for" true (contains "for" s);
  Alcotest.(check bool) "i"   true (contains "i"   s);
  Alcotest.(check bool) "4"   true (contains "4"   s)

let test_pp_stmt_sfor_unroll () =
  let v = fresh_mut_var "i" S32 in
  let s = Tirix_pp.pp_stmt (SFor {
    var = v; start = Const (S32, 0l); stop = Const (S32, 4l)
  ; step = Const (S32, 1l); dir = Upto; unroll = true; body = []
  }) in
  Alcotest.(check bool) "unroll" true (contains "unroll" s)

let test_pp_stmt_sif_then () =
  let s = Tirix_pp.pp_stmt
    (SIf (Const (Bool, true), [SEmpty], [])) in
  Alcotest.(check bool) "if"   true (contains "if"   s);
  Alcotest.(check bool) "true" true (contains "true" s)

let test_pp_stmt_sif_else () =
  let s = Tirix_pp.pp_stmt
    (SIf (Const (Bool, true), [], [SOp (Barrier CtaSync)])) in
  Alcotest.(check bool) "else" true (contains "else" s)

let test_pp_stmt_spipeline () =
  let s = Tirix_pp.pp_stmt (SPipeline {
    stages = 4; prologue = []; mainloop = []; epilogue = []
  }) in
  Alcotest.(check bool) "pipeline" true (contains "pipeline" s);
  Alcotest.(check bool) "4"        true (contains "4"        s)

let test_pp_stmt_warp_group_producer () =
  let s = Tirix_pp.pp_stmt (SWarpGroup (Cluster.Producer, [])) in
  Alcotest.(check bool) "Producer" true (contains "Producer" s)

let test_pp_stmt_warp_group_consumer () =
  let s = Tirix_pp.pp_stmt (SWarpGroup (Cluster.Consumer, [])) in
  Alcotest.(check bool) "Consumer" true (contains "Consumer" s)

let test_pp_stmt_warp_group_epilogue () =
  let s = Tirix_pp.pp_stmt (SWarpGroup (Cluster.Epilogue, [])) in
  Alcotest.(check bool) "Epilogue" true (contains "Epilogue" s)

let test_pp_stmt_warp_group_scheduler () =
  let s = Tirix_pp.pp_stmt (SWarpGroup (Cluster.Scheduler, [])) in
  Alcotest.(check bool) "Scheduler" true (contains "Scheduler" s)

let test_pp_stmt_pragma () =
  let s = Tirix_pp.pp_stmt (SPragma ("unroll", [])) in
  Alcotest.(check bool) "#pragma" true (contains "#pragma" s);
  Alcotest.(check bool) "unroll"  true (contains "unroll"  s)

let test_pp_stmt_sempty () =
  let s = Tirix_pp.pp_stmt SEmpty in
  Alcotest.(check bool) "empty" true (contains "empty" s)

let test_pp_stmt_depth_increases_indent () =
  let v = fresh_var "x" S32 in
  let s0 = Tirix_pp.pp_stmt ~depth:0
    (SLet (v, Expr (Const (S32, 0l)))) in
  let s1 = Tirix_pp.pp_stmt ~depth:1
    (SLet (v, Expr (Const (S32, 0l)))) in
  Alcotest.(check bool) "depth 1 longer" true
    (String.length s1 > String.length s0)

(* ------------------------------------------------------------------ *)
(* pp_helper                                                           *)
(* ------------------------------------------------------------------ *)

let test_pp_helper_name () =
  let h = {
    hf_name     = "make_smem_desc"
  ; hf_params   = []
  ; hf_ret_type = Scalar U64
  ; hf_body     = []
  } in
  let s = Tirix_pp.pp_helper h in
  Alcotest.(check bool) "name"           true (contains "make_smem_desc" s);
  Alcotest.(check bool) "__forceinline__" true (contains "__forceinline__" s)

let test_pp_helper_params () =
  let h = {
    hf_name     = "foo"
  ; hf_params   = [ fresh_var "ptr" U64; fresh_var "len" U32 ]
  ; hf_ret_type = Scalar U32
  ; hf_body     = []
  } in
  let s = Tirix_pp.pp_helper h in
  Alcotest.(check bool) "ptr" true (contains "ptr" s);
  Alcotest.(check bool) "len" true (contains "len" s)

let test_pp_helper_ret_type () =
  let h = {
    hf_name     = "foo"
  ; hf_params   = []
  ; hf_ret_type = Scalar F32
  ; hf_body     = []
  } in
  let s = Tirix_pp.pp_helper h in
  Alcotest.(check bool) "float" true (contains "float" s)

(* ------------------------------------------------------------------ *)
(* pp_tirix                                                            *)
(* ------------------------------------------------------------------ *)

let test_pp_tirix_name () =
  let k = { (minimal_tirix ()) with name = "my_kernel" } in
  let s = Tirix_pp.pp_tirix k in
  Alcotest.(check bool) "kernel name" true (contains "my_kernel" s)

let test_pp_tirix_global () =
  let k = minimal_tirix () in
  let s = Tirix_pp.pp_tirix k in
  Alcotest.(check bool) "__global__" true (contains "__global__" s)

let test_pp_tirix_with_param () =
  let flat = Layout.make (Modes.Int 1) (Modes.Int 1) in
  let p = {
    param_name   = "A"
  ; param_tensor = Tensor {
      tensor_name      = "A"
    ; tensor_id        = Type_id.create ()
    ; tensor_elem_type = Elemtype.Float16
    ; tensor_memspace  = Memspace.Global
    ; tensor_layout    = flat
    ; tensor_swizzle   = Swizzle.make 0 0 0
    }
  ; param_is_tma = false
  } in
  let k = { (minimal_tirix ()) with params = [p] } in
  let s = Tirix_pp.pp_tirix k in
  Alcotest.(check bool) "param A" true (contains "A" s)

let test_pp_tirix_with_body () =
  let v = fresh_var "x" S32 in
  let k = { (minimal_tirix ()) with
    body = [ SLet (v, Expr (Const (S32, 0l))) ] } in
  let s = Tirix_pp.pp_tirix k in
  Alcotest.(check bool) "body stmt" true (contains "x" s)

let test_pp_tirix_with_helper () =
  let h = {
    hf_name     = "make_smem_desc"
  ; hf_params   = []
  ; hf_ret_type = Scalar U64
  ; hf_body     = []
  } in
  let k = { (minimal_tirix ()) with helpers = [h] } in
  let s = Tirix_pp.pp_tirix k in
  Alcotest.(check bool) "helper name" true (contains "make_smem_desc" s)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Tirix_pp" [
    "scalar",   [ Alcotest.test_case "u8"          `Quick test_pp_scalar_u8
                ; Alcotest.test_case "u32"         `Quick test_pp_scalar_u32
                ; Alcotest.test_case "s32"         `Quick test_pp_scalar_s32
                ; Alcotest.test_case "u64"         `Quick test_pp_scalar_u64
                ; Alcotest.test_case "f16"         `Quick test_pp_scalar_f16
                ; Alcotest.test_case "f32"         `Quick test_pp_scalar_f32
                ; Alcotest.test_case "bf16"        `Quick test_pp_scalar_bf16
                ; Alcotest.test_case "bool"        `Quick test_pp_scalar_bool
                ; Alcotest.test_case "ptr"         `Quick test_pp_scalar_ptr ];
    "arith",    [ Alcotest.test_case "add"         `Quick test_pp_arith_add
                ; Alcotest.test_case "sub"         `Quick test_pp_arith_sub
                ; Alcotest.test_case "mul"         `Quick test_pp_arith_mul
                ; Alcotest.test_case "div"         `Quick test_pp_arith_div
                ; Alcotest.test_case "mod"         `Quick test_pp_arith_mod ];
    "cmp",      [ Alcotest.test_case "eq"          `Quick test_pp_cmp_eq
                ; Alcotest.test_case "ne"          `Quick test_pp_cmp_ne
                ; Alcotest.test_case "lt"          `Quick test_pp_cmp_lt
                ; Alcotest.test_case "le"          `Quick test_pp_cmp_le
                ; Alcotest.test_case "gt"          `Quick test_pp_cmp_gt
                ; Alcotest.test_case "ge"          `Quick test_pp_cmp_ge ];
    "logic",    [ Alcotest.test_case "and"         `Quick test_pp_logic_and
                ; Alcotest.test_case "or"          `Quick test_pp_logic_or ];
    "unop",     [ Alcotest.test_case "neg"         `Quick test_pp_unop_neg
                ; Alcotest.test_case "not"         `Quick test_pp_unop_not
                ; Alcotest.test_case "bitnot"      `Quick test_pp_unop_bitnot ];
    "expr",     [ Alcotest.test_case "const-s32"   `Quick test_pp_expr_const_s32
                ; Alcotest.test_case "bool-true"   `Quick test_pp_expr_const_bool_true
                ; Alcotest.test_case "f32"         `Quick test_pp_expr_const_f32
                ; Alcotest.test_case "var"         `Quick test_pp_expr_var
                ; Alcotest.test_case "threadidx-x" `Quick test_pp_expr_builtin_threadidx_x
                ; Alcotest.test_case "threadidx-y" `Quick test_pp_expr_builtin_threadidx_y
                ; Alcotest.test_case "blockidx-x"  `Quick test_pp_expr_builtin_blockidx_x
                ; Alcotest.test_case "warpid"      `Quick test_pp_expr_builtin_warpid
                ; Alcotest.test_case "laneid"      `Quick test_pp_expr_builtin_laneid
                ; Alcotest.test_case "arith-add"   `Quick test_pp_expr_arith_add
                ; Alcotest.test_case "arith-mod"   `Quick test_pp_expr_arith_mod
                ; Alcotest.test_case "cmp-eq"      `Quick test_pp_expr_cmp_eq
                ; Alcotest.test_case "cmp-lt"      `Quick test_pp_expr_cmp_lt
                ; Alcotest.test_case "logic-and"   `Quick test_pp_expr_logic_and
                ; Alcotest.test_case "logic-or"    `Quick test_pp_expr_logic_or
                ; Alcotest.test_case "cast"        `Quick test_pp_expr_cast
                ; Alcotest.test_case "addrconv"    `Quick test_pp_expr_addrconv_shared ];
    "barrier",  [ Alcotest.test_case "cta-sync"    `Quick test_pp_barrier_cta_sync
                ; Alcotest.test_case "warp-sync"   `Quick test_pp_barrier_warp_sync
                ; Alcotest.test_case "mem-fence"   `Quick test_pp_barrier_mem_fence
                ; Alcotest.test_case "mbar-init"   `Quick test_pp_barrier_mbar_init
                ; Alcotest.test_case "arrive-exp"  `Quick test_pp_barrier_mbar_arrive_expect
                ; Alcotest.test_case "wait-parity" `Quick test_pp_barrier_mbar_wait_parity
                ; Alcotest.test_case "mbar-arrive" `Quick test_pp_barrier_mbar_arrive
                ; Alcotest.test_case "cl-arrive"   `Quick test_pp_barrier_cluster_arrive
                ; Alcotest.test_case "cl-wait"     `Quick test_pp_barrier_cluster_wait
                ; Alcotest.test_case "cp-commit"   `Quick test_pp_barrier_cp_async_commit
                ; Alcotest.test_case "cp-wait-all" `Quick test_pp_barrier_cp_async_wait_all
                ; Alcotest.test_case "tc-wait"     `Quick test_pp_barrier_tcgen05_wait
                ; Alcotest.test_case "tc-fence"    `Quick test_pp_barrier_tcgen05_fence ];
    "op",       [ Alcotest.test_case "copy-cp"     `Quick test_pp_op_copy_cp_async
                ; Alcotest.test_case "copy-tma"    `Quick test_pp_op_copy_tma
                ; Alcotest.test_case "copy-mc"     `Quick test_pp_op_copy_multicast
                ; Alcotest.test_case "mma-sm80"    `Quick test_pp_op_mma_sm80
                ; Alcotest.test_case "mma-wgmma"   `Quick test_pp_op_mma_wgmma
                ; Alcotest.test_case "mma-tcgen05" `Quick test_pp_op_mma_tcgen05
                ; Alcotest.test_case "barrier"     `Quick test_pp_op_barrier_cta
                ; Alcotest.test_case "tmem-alloc"  `Quick test_pp_op_tmem_alloc
                ; Alcotest.test_case "tmem-dealloc"`Quick test_pp_op_tmem_dealloc
                ; Alcotest.test_case "tmem-commit" `Quick test_pp_op_tmem_commit
                ; Alcotest.test_case "smem-desc"   `Quick test_pp_op_smem_desc_init ];
    "stmt",     [ Alcotest.test_case "slet"        `Quick test_pp_stmt_slet
                ; Alcotest.test_case "sletmut"     `Quick test_pp_stmt_sletmut
                ; Alcotest.test_case "sassign"     `Quick test_pp_stmt_sassign
                ; Alcotest.test_case "sfor"        `Quick test_pp_stmt_sfor
                ; Alcotest.test_case "unroll"      `Quick test_pp_stmt_sfor_unroll
                ; Alcotest.test_case "sif-then"    `Quick test_pp_stmt_sif_then
                ; Alcotest.test_case "sif-else"    `Quick test_pp_stmt_sif_else
                ; Alcotest.test_case "pipeline"    `Quick test_pp_stmt_spipeline
                ; Alcotest.test_case "producer"    `Quick test_pp_stmt_warp_group_producer
                ; Alcotest.test_case "consumer"    `Quick test_pp_stmt_warp_group_consumer
                ; Alcotest.test_case "epilogue"    `Quick test_pp_stmt_warp_group_epilogue
                ; Alcotest.test_case "scheduler"   `Quick test_pp_stmt_warp_group_scheduler
                ; Alcotest.test_case "pragma"      `Quick test_pp_stmt_pragma
                ; Alcotest.test_case "empty"       `Quick test_pp_stmt_sempty
                ; Alcotest.test_case "depth"       `Quick test_pp_stmt_depth_increases_indent ];
    "helper",   [ Alcotest.test_case "name"        `Quick test_pp_helper_name
                ; Alcotest.test_case "params"      `Quick test_pp_helper_params
                ; Alcotest.test_case "ret-type"    `Quick test_pp_helper_ret_type ];
    "tirix",    [ Alcotest.test_case "name"        `Quick test_pp_tirix_name
                ; Alcotest.test_case "global"      `Quick test_pp_tirix_global
                ; Alcotest.test_case "param"       `Quick test_pp_tirix_with_param
                ; Alcotest.test_case "body"        `Quick test_pp_tirix_with_body
                ; Alcotest.test_case "helper"      `Quick test_pp_tirix_with_helper ];
  ]
