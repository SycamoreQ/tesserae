open Base
open Tesserae_core
open Tesserae_atoms
open Tesserae_pipeline
open Tesserae_kernel
open Tesserae_Tirix

let var_counter = ref 0

let fresh_id () =
  let id = !var_counter in
  var_counter := id + 1;
  id

let mk_var name ty ?(mut = false) () = {
  var_name = name;
  var_id  = fresh_id ();
  var_type = Scalar ty;
  var_mutable = mut;
}

let i32 n = Const (S32, Int32.of_int_exn n)

let elem_type_of (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  desc.Kernel_desc.tile_io
    .Tile_io.tiled_copy_a
    .Tiled_copy.atom
    .Copy_atom.elem_type

let is_tma (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  Tile_io.is_tma desc.Kernel_desc.tile_io

let is_blackwell (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  match desc.Kernel_desc.family with
  | Kernel_desc.Blackwell -> true
  | _ -> false

let bn_smem (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  match desc.Kernel_desc.family with
  | Kernel_desc.Blackwell -> desc.Kernel_desc.bn / 2
  | _ -> desc.Kernel_desc.bn

let flat_layout () =
  Layout.make (Modes.Int 1) (Modes.Int 1)

let make_global_tensor name elem_type =
  Tensor {
    tensor_name = name;
    tensor_id = Type_id.create ();
    tensor_elem_type = elem_type;
    tensor_memspace = Memspace.Global;
    tensor_layout = flat_layout ();
    tensor_swizzle = Swizzle.make 0 0 0;
  }

let construct_smem_tensors (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  let i n = Modes.Int n in
  let tup ts = Modes.Tuple ts in
  let lay s d = Layout.make s d in
  let elem = elem_type_of desc in
  let bm = desc.Kernel_desc.bm in
  let bk = desc.Kernel_desc.bk in
  let bn_s = bn_smem desc in
  let d = desc.Kernel_desc.pipeline.Pipeline.depth in
  let sw = Swizzle.smem_selector elem bm bk in
  let smem_a = Tensor {
    tensor_name  = "smem_A";
    tensor_id = Type_id.create ();
    tensor_elem_type = elem;
    tensor_memspace = Memspace.Shared;
    tensor_layout = lay
      (tup [i d; i bm; i bk])
      (tup [i (bm * bk); i 1; i bm]);
    tensor_swizzle = sw;
  } in
  let smem_b = Tensor {
    tensor_name = "smem_B";
    tensor_id = Type_id.create ();
    tensor_elem_type = elem;
    tensor_memspace = Memspace.Shared;
    tensor_layout = lay
      (tup [i d; i bn_s; i bk])
      (tup [i (bn_s * bk); i 1; i bn_s]);
    tensor_swizzle   = sw;
  } in
  [ ("smem_A", smem_a); ("smem_B", smem_b) ]


let construct_vars (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  let warp_id  = mk_var "warp_id"  S32 () in
  let lane_id  = mk_var "lane_id"  S32 () in
  let block_m  = mk_var "block_m"  S32 () in
  let block_n  = mk_var "block_n"  S32 () in
  let row = mk_var "row" S32 () in
  let col = mk_var "col" S32 () in
  let k_loop = mk_var "k" S32 ~mut:true () in
  let stage = mk_var "stage" S32 ~mut:true () in
  let phase = mk_var "phase" S32 ~mut:true () in
  let base_vars =
    [ warp_id; lane_id; block_m; block_n; row; col; k_loop; stage; phase ]
  in
  if is_tma desc then
    let full_mbar  = mk_var "full_mbar"  U64 () in
    let empty_mbar = mk_var "empty_mbar" U64 () in
    base_vars @ [ full_mbar; empty_mbar ]
  else if is_blackwell desc then
    let tmem_addr = mk_var "tmem_addr" U32 () in
    let cta_rank  = mk_var "cta_rank"  S32 () in
    base_vars @ [ tmem_addr; cta_rank ]
  else
    base_vars

let construct_params (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  let elem = elem_type_of desc in
  let tma = is_tma desc in
  let a_param = {
    param_name = "A";
    param_tensor = make_global_tensor "A" elem;
    param_is_tma = tma;
  } in
  let b_param = {
    param_name = "B";
    param_tensor = make_global_tensor "B" elem;
    param_is_tma = tma;
  } in
  let c_param = {
    param_name = "C";
    param_tensor = make_global_tensor "C" Elemtype.Float32;
    param_is_tma = false;
  } in
  (* M, N, K are scalar ints — we represent them as flat global tensors
     of size 1. tirix_emit will special-case params named M/N/K to emit
     them as `int M` rather than a pointer. *)
  let m_param = {
    param_name = "M";
    param_tensor = make_global_tensor "M" Elemtype.Int32;
    param_is_tma = false;
  } in
  let n_param = {
    param_name = "N";
    param_tensor = make_global_tensor "N" Elemtype.Int32;
    param_is_tma = false;
  } in
  let k_param = {
    param_name = "K";
    param_tensor = make_global_tensor "K" Elemtype.Int32;
    param_is_tma = false;
  } in
  [ a_param; b_param; c_param; m_param; n_param; k_param ]


let construct_helpers (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  let make_smem_desc_fn = {
    hf_name     = "make_smem_desc";
    hf_params   = [ mk_var "ptr"     U64 ()
                  ; mk_var "leading" U32 ()
                  ; mk_var "stride"  U32 ()
                  ; mk_var "sw"      U32 () ];
    hf_ret_type = Scalar U64;
    hf_body     = [
      SLet (mk_var "addr" U32 (), Expr (AddrConv (GenericToShared,
        Var (mk_var "ptr" U64 ()))));
      (* body emits the descriptor bit-packing — tirix_emit handles the
         actual asm string from SmemDescInit op *)
      SOp (SmemDescInit {
        desc_var = mk_var "desc" U64 ~mut:true ();
        ptr_expr = Var (mk_var "ptr" U64 ());
        leading_dim = 0;
        stride = 0;
        swizzle = Swizzle.make 0 0 0;
      });
    ];
  } in
  let tma_load_fn = {
    hf_name = "tma_2d_gmem2smem";
    hf_params = [ mk_var "smem" U64 ()
                  ; mk_var "tmap" U64 ()
                  ; mk_var "x" S32 ()
                  ; mk_var "y" S32 ()
                  ; mk_var "mbar" U64 () ];
    hf_ret_type = Scalar U32;
    hf_body = [
      SOp (Copy {
        copy_kind = TmaLoad;
        src_tensor = make_global_tensor "gmem" (elem_type_of desc);
        dst_tensor = Tensor {
          tensor_name = "smem";
          tensor_id = Type_id.create ();
          tensor_elem_type = elem_type_of desc;
          tensor_memspace = Memspace.Shared;
          tensor_layout = flat_layout ();
          tensor_swizzle = Swizzle.make 0 0 0;
        };
        pred_expr = None;
        mbar_var= Some (mk_var "mbar" U64 ());
      });
    ];
  } in
  let tma_multicast_fn = {
    hf_name     = "tma_2d_gmem2smem_multicast";
    hf_params   = [ mk_var "smem" U64 ()
                  ; mk_var "tmap" U64 ()
                  ; mk_var "x" S32 ()
                  ; mk_var "y" S32 ()
                  ; mk_var "mbar" U64 ()
                  ; mk_var "mask" U32 () ];
    hf_ret_type = Scalar U32;
    hf_body     = [
      SOp (Copy {
        copy_kind  = TmaMulticast;
        src_tensor = make_global_tensor "gmem" (elem_type_of desc);
        dst_tensor = Tensor {
          tensor_name = "smem";
          tensor_id = Type_id.create ();
          tensor_elem_type = elem_type_of desc;
          tensor_memspace = Memspace.Shared;
          tensor_layout = flat_layout ();
          tensor_swizzle = Swizzle.make 0 0 0;
        };
        pred_expr = None;
        mbar_var = Some (mk_var "mbar" U64 ());
      });
    ];
  } in
  match is_blackwell desc, is_tma desc with
  | true,  true -> [ make_smem_desc_fn; tma_load_fn; tma_multicast_fn ]
  | false, true -> [ make_smem_desc_fn; tma_load_fn ]
  | _,     false -> []

let construct_producer_body
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list)
    (smem : (string * packed_tensor) list) =
  let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
  let k_var = find "k" in
  let stage_var = find "stage" in
  let bk = desc.Kernel_desc.bk in
  let depth = desc.Kernel_desc.pipeline.Pipeline.depth in
  let _, smem_a = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_A") in
  let _, smem_b = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_B") in
  let a_tensor = make_global_tensor "A" (elem_type_of desc) in
  let b_tensor = make_global_tensor "B" (elem_type_of desc) in
  let let_stage =
    SLet (stage_var,
      Expr (Binop (Mod, Var k_var, i32 depth)))
  in
  let copy_ops = match is_tma desc, is_blackwell desc with
    | true, true ->
      let mbar = find "full_mbar" in
      [ SOp (Barrier (MbarArriveExpect {
            mbar  = mbar;
            bytes = i32 ((desc.Kernel_desc.bm + bn_smem desc)
                         * bk
                         * Elemtype.byte_width (elem_type_of desc)) }))
      ; SOp (Copy {
            copy_kind = TmaMulticast;
            src_tensor = a_tensor;
            dst_tensor = smem_a;
            pred_expr = None;
            mbar_var = Some mbar; })
      ; SOp (Copy {
            copy_kind = TmaMulticast;
            src_tensor = b_tensor;
            dst_tensor = smem_b;
            pred_expr = None;
            mbar_var = Some mbar; }) ]
    | true, false ->
      let mbar = find "full_mbar" in
      [ SOp (Barrier (MbarArriveExpect {
            mbar  = mbar;
            bytes = i32 ((desc.Kernel_desc.bm + bn_smem desc)
                         * bk
                         * Elemtype.byte_width (elem_type_of desc)) }))
      ; SOp (Copy {
            copy_kind = TmaLoad;
            src_tensor = a_tensor;
            dst_tensor = smem_a;
            pred_expr = None;
            mbar_var = Some mbar; })
      ; SOp (Copy {
            copy_kind = TmaLoad;
            src_tensor = b_tensor;
            dst_tensor = smem_b;
            pred_expr = None;
            mbar_var = Some mbar; }) ]
    | false, _ ->
      [ SOp (Copy {
            copy_kind = CpAsync;
            src_tensor = a_tensor;
            dst_tensor = smem_a;
            pred_expr = None;
            mbar_var = None; })
      ; SOp (Copy {
            copy_kind = CpAsync;
            src_tensor = b_tensor;
            dst_tensor = smem_b;
            pred_expr = None;
            mbar_var = None; })
      ; SOp (Barrier CpAsyncCommitGroup) ]
  in
  let loop_body = let_stage :: copy_ops in
  SWarpGroup (Cluster.Producer, [
    SFor {
      var    = k_var;
      start  = i32 0;
      stop   = i32 (Layout.size
                (Layout.make (Modes.Int (desc.Kernel_desc.bk)) (Modes.Int 1)));
      step   = i32 1;
      dir    = Upto;
      unroll = false;
      body   = loop_body;
    }
  ])

let construct_consumer_body
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list)
    (smem : (string * packed_tensor) list) =
  let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
  let k_var = find "k" in
  let stage_var = find "stage" in
  let phase_var = find "phase" in
  let bk = desc.Kernel_desc.bk in
  let depth = desc.Kernel_desc.pipeline.Pipeline.depth in
  let _, smem_a = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_A") in
  let _, smem_b = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_B") in
  let acc = make_global_tensor "acc" Elemtype.Float32 in
  let mma_kind = match desc.Kernel_desc.family with
    | Kernel_desc.Ampere -> Sm80Mma
    | Kernel_desc.Hopper -> Sm90Wgmma
    | Kernel_desc.Blackwell -> Sm100Tcgen05
  in
  let let_stage =
    SLet (stage_var, Expr (Binop (Mod, Var k_var, i32 depth)))
  in
  let let_phase =
    SLet (phase_var,
      Expr (Binop (Mod,
        Binop (Div, Var k_var, i32 depth),
        i32 2)))
  in
  let wait_op = if is_tma desc then
    let mbar = find "full_mbar" in
    SOp (Barrier (MbarWaitParity { mbar; phase = Var phase_var }))
  else
    SOp (Barrier CpAsyncWaitAll)
  in
  let mma_op = SOp (Mma {
    mma_kind;
    tensor_a  = smem_a;
    tensor_b  = smem_b;
    tensor_c  = acc;
    smem_desc_a = None;
    smem_desc_b = None;
    accum_flag = true;
  }) in
  let arrive_op = if is_tma desc then
    let mbar = find "empty_mbar" in
    [ SOp (Barrier (MbarArrive { mbar })) ]
  else [] in
  let commit_op = if is_blackwell desc then
    let mbar = find "full_mbar" in
    [ SOp (TmemCommit { mbar_var = mbar; cta_mask = Some 0b11 }) ]
  else [] in
  let loop_body =
    [ let_stage; let_phase; wait_op; mma_op ]
    @ arrive_op
    @ commit_op
  in
  SWarpGroup (Cluster.Consumer, [
    SFor {
      var    = k_var;
      start  = i32 0;
      stop   = i32 bk;
      step   = i32 1;
      dir    = Upto;
      unroll = false;
      body   = loop_body;
    }
  ])

let construct_epilogue_body
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list) =
  let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
  match desc.Kernel_desc.family with
  | Kernel_desc.Blackwell ->
    let tmem_addr = find "tmem_addr" in
    let n_loads   = desc.Kernel_desc.bn / 8 in
    let dst_vars  = List.init n_loads ~f:(fun i ->
      mk_var (Printf.sprintf "reg_%d" i) F32 ()) in
    SWarpGroup (Cluster.Epilogue, [
      SOp (TmemLoad {
        dst_vars;
        src_addr   = Cast (U64, Var tmem_addr);
        col_offset = 0;
      });
    ])
  | _ ->
    let acc = make_global_tensor "acc" Elemtype.Float32 in
    let c   = make_global_tensor "C"   Elemtype.Float32 in
    SWarpGroup (Cluster.Epilogue, [
      SOp (Mma {
        mma_kind    = Sm80Mma;
        tensor_a    = acc;
        tensor_b    = acc;
        tensor_c    = c;
        smem_desc_a = None;
        smem_desc_b = None;
        accum_flag  = false;
      });
    ])

let construct_mbar_init
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list) =
  if not (is_tma desc) then []
  else
    let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
    let warp_id = find "warp_id" in
    let lane_id = find "lane_id" in
    let full_m  = find "full_mbar" in
    let empty_m = find "empty_mbar" in
    let depth = desc.Kernel_desc.pipeline.Pipeline.depth in
    let i_var = mk_var "i" S32 ~mut:true () in
    let init_loop = SFor {
      var = i_var;
      start = i32 0;
      stop = i32 depth;
      step = i32 1;
      dir = Upto;
      unroll = false;
      body = [
        SOp (Barrier (MbarInit { mbar = full_m;  count = 1 }));
        SOp (Barrier (MbarInit { mbar = empty_m; count = 1 }));
      ];
    } in
    [ SIf (
        Binop (And,
          Binop (Eq, Var warp_id, i32 0),
          Binop (Eq, Var lane_id, i32 0)),
        [ init_loop ],
        []
      )
    ; SOp (Barrier CtaSync) ]

let tmem_alloc_op
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list) =
  if not (is_blackwell desc) then None
  else
    let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
    let addr = find "tmem_addr" in
    Some (SOp (TmemAlloc { addr_var = addr; col_count = desc.Kernel_desc.bn }))

let tmem_dealloc_op
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list) =
  if not (is_blackwell desc) then None
  else
    let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
    let addr = find "tmem_addr" in
    Some (SOp (TmemDealloc { addr_var = addr; col_count = desc.Kernel_desc.bn }))


let lower (desc : (_, _, _, _, _, _) Kernel_desc.t) : tirix =
  let tensors  = construct_smem_tensors desc in
  let vars     = construct_vars desc in
  let params   = construct_params desc in
  let helpers  = construct_helpers desc in
  let mbar_init = construct_mbar_init desc vars in
  let producer = construct_producer_body desc vars tensors in
  let consumer = construct_consumer_body desc vars tensors in
  let epilogue = construct_epilogue_body desc vars in
  let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
  let warp_id = find "warp_id" in
  let prod_warp = Option.value ~default:0
    (Cluster.producer_warp desc.Kernel_desc.cluster) in
  let cons_warp = Option.value ~default:1
    (Cluster.consumer_warp desc.Kernel_desc.cluster) in
  let warp_dispatch =
    SIf (Binop (Eq, Var warp_id, i32 prod_warp),
      [ producer ],
      [ SIf (Binop (Eq, Var warp_id, i32 cons_warp),
          [ consumer ],
          [ epilogue ]) ])
  in
  let var_decls = List.map vars ~f:(fun v ->
    SLet (v, Expr (Const (S32, 0l)))) in
  let alloc   = Option.to_list (tmem_alloc_op   desc vars) in
  let dealloc = Option.to_list (tmem_dealloc_op desc vars) in
  let body =
    var_decls
    @ alloc
    @ mbar_init
    @ [ warp_dispatch ]
    @ dealloc
  in
  { name  = desc.Kernel_desc.name
  ; family = desc.Kernel_desc.family
  ; params
  ; tensors
  ; smem_bytes = Kernel_desc.smem_bytes desc
  ; cluster = desc.Kernel_desc.cluster
  ; body
  ; helpers
  }
