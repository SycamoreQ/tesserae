open Tesserae_core
open Tesserae_pipeline
open Tesserae_kernel
open Tesserae_tirix
open Tirix
open Base


let fresh_var name ty =
  { var_name = name
  ; var_id = 0
  ; var_type = Scalar ty
  ; var_mutable = false
  }

let fresh_mut_var name ty =
  { var_name = name
  ; var_id = 0
  ; var_type = Scalar ty
  ; var_mutable = true
  }

let flat_layout () =
  Layout.make (Modes.Int 1) (Modes.Int 1)

let make_tensor name elem space =
  Tensor {
    tensor_name = name
  ; tensor_id = Type_id.create ()
  ; tensor_elem_type = elem
  ; tensor_memspace = space
  ; tensor_layout = flat_layout ()
  ; tensor_swizzle = Swizzle.make 0 0 0
  }

let minimal_cluster () =
  Cluster.make
    { Cluster.x = 1; y = 1; z = 1 } 4
    [ (0, Cluster.Producer); (1, Cluster.Consumer)
    ; (2, Cluster.Epilogue); (3, Cluster.Epilogue) ]

let minimal_Tirix () =
  { name = "test_kernel"
  ; family = Kernel_desc.Ampere
  ; params = []
  ; tensors = []
  ; bm = 128
  ; bn = 128
  ; bk = 32
  ; smem_bytes = 0
  ; pipeline_depth = 4
  ; cluster = minimal_cluster ()
  ; body = []
  ; helpers = []
  }


let test_type_id_create () =
  let id = Type_id.create () in
  ignore id;
  Alcotest.(check bool) "create" true true

let test_type_id_equal_same () =
  let id : int Type_id.t = Type_id.create () in
  match Type_id.equal id id with
  | Some Type_id.Refl -> Alcotest.(check bool) "same" true true
  | None -> Alcotest.fail "expected Refl"

let test_type_id_equal_different () =
  let id1 : int Type_id.t = Type_id.create () in
  let id2 : int Type_id.t = Type_id.create () in
  match Type_id.equal id1 id2 with
  | None -> Alcotest.(check bool) "different" true true
  | Some _ -> Alcotest.fail "expected None"


let test_scalar_ty_u8 () =
  let v = fresh_var "x" U8 in
  match v.var_type with
  | Scalar U8 -> Alcotest.(check bool) "u8" true true
  | _ -> Alcotest.fail "wrong type"

let test_scalar_ty_f32 () =
  let v = fresh_var "x" F32 in
  match v.var_type with
  | Scalar F32 -> Alcotest.(check bool) "f32" true true
  | _ -> Alcotest.fail "wrong type"

let test_scalar_ty_bool () =
  let v = fresh_var "x" Bool in
  match v.var_type with
  | Scalar Bool -> Alcotest.(check bool) "bool" true true
  | _ -> Alcotest.fail "wrong type"


let test_tensor_name () =
  let (Tensor t) = make_tensor "smem_A" Elemtype.Float16 Memspace.Shared in
  Alcotest.(check string) "name" "smem_A" t.tensor_name

let test_tensor_memspace_shared () =
  let (Tensor t) = make_tensor "smem_A" Elemtype.Float16 Memspace.Shared in
  Alcotest.(check string) "shared" "shared"
    (Memspace.name t.tensor_memspace)

let test_tensor_memspace_global () =
  let (Tensor t) = make_tensor "A" Elemtype.Float16 Memspace.Global in
  Alcotest.(check string) "global" "global"
    (Memspace.name t.tensor_memspace)

let test_tensor_layout_size () =
  let (Tensor t) = make_tensor "A" Elemtype.Float16 Memspace.Global in
  Alcotest.(check int) "size" 1 (Layout.size t.tensor_layout)

let test_packed_tensor_wraps () =
  let pt = make_tensor "A" Elemtype.Float16 Memspace.Global in
  match pt with
  | Tensor t -> Alcotest.(check string) "packed" "A" t.tensor_name

let test_var_name () =
  let v = fresh_var "warp_id" S32 in
  Alcotest.(check string) "name" "warp_id" v.var_name

let test_var_immutable () =
  let v = fresh_var "x" S32 in
  Alcotest.(check bool) "immutable" false v.var_mutable

let test_var_mutable () =
  let v = fresh_mut_var "k" S32 in
  Alcotest.(check bool) "mutable" true v.var_mutable


let test_expr_const_s32 () =
  let e = Const (S32, 42l) in
  match e with
  | Const (S32, v) -> Alcotest.(check int) "42" 42 (Int32.to_int_exn v)
  | _ -> Alcotest.fail "wrong"

let test_expr_var () =
  let v = fresh_var "x" S32 in
  let e : int32 expr = Var v in
  match e with
  | Var vv -> Alcotest.(check string) "var name" "x" vv.var_name
  | _ -> Alcotest.fail "wrong"

let test_expr_builtin_threadidx () =
  let e = Builtin (ThreadIdx X) in
  match e with
  | Builtin (ThreadIdx X) -> Alcotest.(check bool) "threadIdx.x" true true
  | _ -> Alcotest.fail "wrong"

let test_expr_binop () =
  let e = Arith (Add, Const (S32, 1l), Const (S32, 2l)) in
  match e with
  | Arith (Add, _, _) -> Alcotest.(check bool) "binop add" true true
  | _ -> Alcotest.fail "wrong"

let test_expr_cast () =
  let e = Cast (U32, Const (S32, 1l)) in
  match e with
  | Cast (U32, _) -> Alcotest.(check bool) "cast" true true
  | _ -> Alcotest.fail "wrong"

let test_expr_addrconv () =
  let e = AddrConv (GenericToShared, Const (U64, 0L)) in
  match e with
  | AddrConv (GenericToShared, _) ->
    Alcotest.(check bool) "addrconv" true true
  | _ -> Alcotest.fail "wrong"


let test_barrier_cta_sync () =
  let b = CtaSync in
  match b with
  | CtaSync -> Alcotest.(check bool) "cta sync" true true
  | _ -> Alcotest.fail "wrong"

let test_barrier_mbar_init () =
  let v = fresh_var "mbar" U64 in
  let b = MbarInit { mbar = v; count = 1 } in
  match b with
  | MbarInit { count; _ } ->
    Alcotest.(check int) "count" 1 count
  | _ -> Alcotest.fail "wrong"

let test_barrier_mbar_wait_parity () =
  let v = fresh_var "mbar" U64 in
  let b = MbarWaitParity { mbar = v; phase = Const (S32, 0l) } in
  match b with
  | MbarWaitParity _ -> Alcotest.(check bool) "wait parity" true true
  | _ -> Alcotest.fail "wrong"


let test_op_copy_cp_async () =
  let src = make_tensor "A"      Elemtype.Float16 Memspace.Global in
  let dst = make_tensor "smem_A" Elemtype.Float16 Memspace.Shared in
  let op  = Copy {
    copy_kind  = CpAsync
  ; src_tensor = src
  ; dst_tensor = dst
  ; pred_expr  = None
  ; mbar_var   = None
  } in
  match op with
  | Copy c -> Alcotest.(check bool) "cp async" true
      (match c.copy_kind with CpAsync -> true | _ -> false)
  | _ -> Alcotest.fail "wrong"

let test_op_mma_sm80 () =
  let a   = make_tensor "smem_A" Elemtype.Float16 Memspace.Shared in
  let b   = make_tensor "smem_B" Elemtype.Float16 Memspace.Shared in
  let c   = make_tensor "acc"    Elemtype.Float32 Memspace.Register in
  let op  = Mma {
  mma_kind    = Sm80Mma
  ; mma_atom = default_atom_for_kind Sm80Mma
  ; tensor_a    = a
  ; tensor_b    = b
  ; tensor_c    = c
  ; smem_desc_a = None
  ; smem_desc_b = None
  ; accum_flag  = true
  } in
  match op with
  | Mma m -> Alcotest.(check bool) "sm80 mma" true
      (match m.mma_kind with Sm80Mma -> true | _ -> false)
  | _ -> Alcotest.fail "wrong"

let test_op_barrier () =
  let op = Barrier CtaSync in
  match op with
  | Barrier CtaSync -> Alcotest.(check bool) "barrier" true true
  | _ -> Alcotest.fail "wrong"

let test_op_tmem_alloc () =
  let v  = fresh_var "tmem_addr" U32 in
  let op = TmemAlloc { addr_var = v; col_count = 256 } in
  match op with
  | TmemAlloc { col_count; _ } ->
    Alcotest.(check int) "col_count" 256 col_count
  | _ -> Alcotest.fail "wrong"

let test_stmt_slet () =
  let v = fresh_var "x" S32 in
  let s = SLet (v, Expr (Const (S32, 0l))) in
  match s with
  | SLet (vv, _) -> Alcotest.(check string) "slet" "x" vv.var_name
  | _ -> Alcotest.fail "wrong"

let test_stmt_sfor_stages () =
  let v = fresh_mut_var "i" S32 in
  let s = SFor {
    var    = v
  ; start  = Const (S32, 0l)
  ; stop   = Const (S32, 4l)
  ; step   = Const (S32, 1l)
  ; dir    = Upto
  ; unroll = false
  ; body   = []
  } in
  match s with
  | SFor { stop = Const (S32, n); _ } ->
    Alcotest.(check int) "stop=4" 4 (Int32.to_int_exn n)
  | _ -> Alcotest.fail "wrong"

let test_stmt_spipeline_stages () =
  let s = SPipeline {
    stages   = 4
  ; prologue = []
  ; mainloop = []
  ; epilogue = []
  } in
  match s with
  | SPipeline { stages; _ } ->
    Alcotest.(check int) "stages" 4 stages
  | _ -> Alcotest.fail "wrong"

let test_stmt_swarp_group_producer () =
  let s = SWarpGroup (Cluster.Producer, [SEmpty]) in
  match s with
  | SWarpGroup (Cluster.Producer, body) ->
    Alcotest.(check int) "body len" 1 (List.length body)
  | _ -> Alcotest.fail "wrong"

let test_stmt_swarp_group_consumer () =
  let s = SWarpGroup (Cluster.Consumer, []) in
  match s with
  | SWarpGroup (Cluster.Consumer, _) ->
    Alcotest.(check bool) "consumer" true true
  | _ -> Alcotest.fail "wrong"

let test_stmt_sif () =
  let s = SIf (Const (Bool, true), [SEmpty], []) in
  match s with
  | SIf (Const (Bool, true), _, _) ->
    Alcotest.(check bool) "sif" true true
  | _ -> Alcotest.fail "wrong"


let test_helper_func_name () =
  let h = {
    hf_name     = "make_smem_desc"
  ; hf_params   = []
  ; hf_ret_type = Scalar U64
  ; hf_body     = []
  } in
  Alcotest.(check string) "name" "make_smem_desc" h.hf_name

let test_helper_func_ret_type () =
  let h = {
    hf_name     = "foo"
  ; hf_params   = [ fresh_var "x" U32 ]
  ; hf_ret_type = Scalar F32
  ; hf_body     = []
  } in
  match h.hf_ret_type with
  | Scalar F32 -> Alcotest.(check bool) "f32 ret" true true
  | _ -> Alcotest.fail "wrong"

(* ------------------------------------------------------------------ *)
(* param                                                               *)
(* ------------------------------------------------------------------ *)

let test_param_not_tma () =
  let p = {
    param_name   = "A"
  ; param_tensor = make_tensor "A" Elemtype.Float16 Memspace.Global
  ; param_is_tma = false
  } in
  Alcotest.(check bool) "not tma" false p.param_is_tma

let test_param_tma () =
  let p = {
    param_name   = "A"
  ; param_tensor = make_tensor "A" Elemtype.Float16 Memspace.Global
  ; param_is_tma = true
  } in
  Alcotest.(check bool) "is tma" true p.param_is_tma

let test_Tirix_name () =
  let k = minimal_Tirix () in
  Alcotest.(check string) "name" "test_kernel" k.name

let test_Tirix_family_ampere () =
  let k = minimal_Tirix () in
  match k.family with
  | Kernel_desc.Ampere -> Alcotest.(check bool) "ampere" true true
  | _ -> Alcotest.fail "wrong"

let test_Tirix_empty_body () =
  let k = minimal_Tirix () in
  Alcotest.(check int) "empty body" 0 (List.length k.body)

let test_Tirix_smem_bytes () =
  let k = minimal_Tirix () in
  Alcotest.(check int) "smem_bytes" 0 k.smem_bytes

let test_Tirix_pipeline_depth () =
  let k = minimal_Tirix () in
  Alcotest.(check int) "depth" 4 k.pipeline_depth


let test_Tirix_cluster_warps () =
  let k = minimal_Tirix () in
  Alcotest.(check int) "warps" 4 k.cluster.Cluster.num_warps

let test_Tirix_tensors_empty () =
  let k = minimal_Tirix () in
  Alcotest.(check int) "no tensors" 0 (List.length k.tensors)

let test_Tirix_tensors_nonempty () =
  let k = { (minimal_Tirix ()) with tensors =
    [ ("smem_A", make_tensor "smem_A" Elemtype.Float16 Memspace.Shared) ] }
  in
  Alcotest.(check int) "one tensor" 1 (List.length k.tensors)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Tirix" [
    "type_id",  [ Alcotest.test_case "create"     `Quick test_type_id_create
                ; Alcotest.test_case "eq-same"    `Quick test_type_id_equal_same
                ; Alcotest.test_case "eq-diff"    `Quick test_type_id_equal_different ];
    "scalar",   [ Alcotest.test_case "u8"         `Quick test_scalar_ty_u8
                ; Alcotest.test_case "f32"        `Quick test_scalar_ty_f32
                ; Alcotest.test_case "bool"       `Quick test_scalar_ty_bool ];
    "tensor",   [ Alcotest.test_case "name"       `Quick test_tensor_name
                ; Alcotest.test_case "shared"     `Quick test_tensor_memspace_shared
                ; Alcotest.test_case "global"     `Quick test_tensor_memspace_global
                ; Alcotest.test_case "size"       `Quick test_tensor_layout_size
                ; Alcotest.test_case "packed"     `Quick test_packed_tensor_wraps ];
    "var",      [ Alcotest.test_case "name"       `Quick test_var_name
                ; Alcotest.test_case "immutable"  `Quick test_var_immutable
                ; Alcotest.test_case "mutable"    `Quick test_var_mutable ];
    "expr",     [ Alcotest.test_case "const"      `Quick test_expr_const_s32
                ; Alcotest.test_case "var"        `Quick test_expr_var
                ; Alcotest.test_case "builtin"    `Quick test_expr_builtin_threadidx
                ; Alcotest.test_case "binop"      `Quick test_expr_binop
                ; Alcotest.test_case "cast"       `Quick test_expr_cast
                ; Alcotest.test_case "addrconv"   `Quick test_expr_addrconv ];
    "barrier",  [ Alcotest.test_case "cta-sync"   `Quick test_barrier_cta_sync
                ; Alcotest.test_case "mbar-init"  `Quick test_barrier_mbar_init
                ; Alcotest.test_case "wait-parity"`Quick test_barrier_mbar_wait_parity ];
    "op",       [ Alcotest.test_case "copy"       `Quick test_op_copy_cp_async
                ; Alcotest.test_case "mma-sm80"   `Quick test_op_mma_sm80
                ; Alcotest.test_case "barrier"    `Quick test_op_barrier
                ; Alcotest.test_case "tmem-alloc" `Quick test_op_tmem_alloc ];
    "stmt",     [ Alcotest.test_case "slet"       `Quick test_stmt_slet
                ; Alcotest.test_case "sfor"       `Quick test_stmt_sfor_stages
                ; Alcotest.test_case "pipeline"   `Quick test_stmt_spipeline_stages
                ; Alcotest.test_case "producer"   `Quick test_stmt_swarp_group_producer
                ; Alcotest.test_case "consumer"   `Quick test_stmt_swarp_group_consumer
                ; Alcotest.test_case "sif"        `Quick test_stmt_sif ];
    "helper",   [ Alcotest.test_case "name"       `Quick test_helper_func_name
                ; Alcotest.test_case "ret-type"   `Quick test_helper_func_ret_type ];
    "param",    [ Alcotest.test_case "not-tma"    `Quick test_param_not_tma
                ; Alcotest.test_case "tma"        `Quick test_param_tma ];
    "Tirix",      [ Alcotest.test_case "name"       `Quick test_Tirix_name
                ; Alcotest.test_case "family"     `Quick test_Tirix_family_ampere
                ; Alcotest.test_case "empty-body" `Quick test_Tirix_empty_body
                ; Alcotest.test_case "smem-bytes" `Quick test_Tirix_smem_bytes
                ; Alcotest.test_case "depth"      `Quick test_Tirix_pipeline_depth
                ; Alcotest.test_case "cluster"    `Quick test_Tirix_cluster_warps
                ; Alcotest.test_case "tensors-0"  `Quick test_Tirix_tensors_empty
                ; Alcotest.test_case "tensors-1"  `Quick test_Tirix_tensors_nonempty ];
  ]
