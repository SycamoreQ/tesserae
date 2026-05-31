open Base
open Tesserae_core
open Tesserae_atoms
open Tesserae_pipeline
open Tesserae_kernel
open Tirix

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

let packed_atom_of_desc (desc : (_, _, _, _, _, _) Kernel_desc.t) : Tirix.packed_mma_atom =
  match desc.Kernel_desc.family with
  | Kernel_desc.Ampere ->
    Tirix.Atom80 (Mma_atom.sm80_16x8x16_f32f16f16f32
                    Mma_atom.ColMajor Mma_atom.RowMajor)
  | Kernel_desc.Hopper ->
    Tirix.Atom90 (Mma_atom.sm90_64x128x16_f32f16f16f32
                    Mma_atom.ColMajor Mma_atom.RowMajor)
  | Kernel_desc.Blackwell ->
    Tirix.Atom100 (Mma_atom.sm100_128x128x16_f32f16f16f32
                     Mma_atom.ColMajor Mma_atom.RowMajor)


let make_global_tensor name elem_type =
  Tensor {
    tensor_name = name;
    tensor_id = Type_id.create ();
    tensor_elem_type = elem_type;
    tensor_memspace = Memspace.Global;
    tensor_layout = flat_layout ();
    tensor_swizzle = Swizzle.make 0 0 0;
  }

let make_shared_tensor name elem_type bm bk depth =
  let i n = Modes.Int n in
  let tup ts = Modes.Tuple ts in
  let lay s d = Layout.make s d in
  let sw = Swizzle.smem_selector elem_type bm bk in
  Tensor {
    tensor_name  = name;
    tensor_id = Type_id.create ();
    tensor_elem_type = elem_type;
    tensor_memspace = Memspace.Shared;
    tensor_layout = lay
      (tup [i depth; i bm; i bk])
      (tup [i (bm * bk); i 1; i bm]);
    tensor_swizzle = sw;
  }

let construct_smem_tensors (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  let elem = elem_type_of desc in
  let bm = desc.Kernel_desc.bm in
  let bk = desc.Kernel_desc.bk in
  let bn_s = bn_smem desc in
  let d = desc.Kernel_desc.pipeline.Pipeline.depth in
  let smem_a = make_shared_tensor "smem_A" elem bm bk d in
  let smem_b = make_shared_tensor "smem_B" elem bn_s bk d in
  let tensors = [ ("smem_A", smem_a); ("smem_B", smem_b) ] in
  if is_tma desc then
    let mbar_full = Tensor {
      tensor_name = "full_mbar";
      tensor_id = Type_id.create ();
      tensor_elem_type = Elemtype.Int32;
      tensor_memspace = Memspace.Shared;
      tensor_layout = flat_layout ();
      tensor_swizzle = Swizzle.make 0 0 0;
    } in
    let mbar_empty = Tensor {
      tensor_name  = "empty_mbar";
      tensor_id    = Type_id.create ();
      tensor_elem_type = Elemtype.Int32;
      tensor_memspace = Memspace.Shared;
      tensor_layout = flat_layout ();
      tensor_swizzle = Swizzle.make 0 0 0;
    } in
    tensors @ [ ("full_mbar", mbar_full); ("empty_mbar", mbar_empty) ]
  else
    tensors


let construct_vars (desc : (_, _, _, _, _, _) Kernel_desc.t) =
  let warp_id = mk_var "warp_id" S32 () in
  let lane_id = mk_var "lane_id" S32 () in
  let block_m = mk_var "block_m" S32 () in
  let block_n = mk_var "block_n" S32 () in
  let row = mk_var "row" S32 () in
  let col = mk_var "col" S32 () in
  let base_vars = [ warp_id; lane_id; block_m; block_n; row; col ] in

  let tma_vars =
    if is_tma desc then
      let full_mbar  = mk_var "full_mbar"  U64 () in
      let empty_mbar = mk_var "empty_mbar" U64 () in
      [ full_mbar; empty_mbar ]
    else
      []
  in
  let arch_vars =
    if is_blackwell desc then
      let tmem_addr = mk_var "tmem_addr" U32 ~mut:true () in
      let cta_rank  = mk_var "cta_rank"  S32 () in
      let n_loads = desc.Kernel_desc.bn / 8 in
      let regs = List.init n_loads ~f:(fun i ->
        mk_var (Printf.sprintf "reg_%d" i) F32 ())
      in
      [ tmem_addr; cta_rank ] @ regs
    else
      []
  in
  let coord_k = mk_var "coord_k" S32 ~mut:true () in
  let coord_m = mk_var "coord_m" S32 ~mut:true () in
  let coord_n = mk_var "coord_n" S32 ~mut:true () in
  base_vars @ tma_vars @ arch_vars @ [coord_k; coord_m; coord_n]

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
        dst_tensor = make_shared_tensor "smem" (elem_type_of desc) 128 32 4;
        pred_expr = None;
        mbar_var= Some (mk_var "mbar" U64 ());
        tma_coord_k = None;
        tma_coord_m = None;
        tma_coord_n = None;
        stage_var = None;
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
        dst_tensor = make_shared_tensor "smem" (elem_type_of desc) 128 32 4;
        pred_expr = None;
        mbar_var = Some (mk_var "mbar" U64 ());
        tma_coord_k = None;
        tma_coord_m = None;
        tma_coord_n = None;
        stage_var = None;
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
  let k_var = mk_var "k" S32 ~mut:true () in
  let stage_var = mk_var "stage" S32 ~mut:true () in
  let bk = desc.Kernel_desc.bk in
  let depth = desc.Kernel_desc.pipeline.Pipeline.depth in
  let _, smem_a = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_A") in
  let _, smem_b = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_B") in
  let a_tensor = make_global_tensor "A" (elem_type_of desc) in
  let b_tensor = make_global_tensor "B" (elem_type_of desc) in
  let atom_k = match packed_atom_of_desc desc with
    | Atom90 a -> let (_, _, k) = Mma_atom.shape a in k
    | Atom80 a -> let (_, _, k) = Mma_atom.shape a in k
    | Atom100 a -> let (_, _, k) = Mma_atom.shape a in k
  in
  let let_stage = SAssign (stage_var,
    Expr (Arith (Mod,
      Arith (Div, Var k_var, i32 atom_k),
      i32 depth))) in

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
            mbar_var = Some mbar;
            tma_coord_k = Some (Var k_var);
            tma_coord_m = Some (Var (find "row"));
            tma_coord_n = None;
            stage_var = Some stage_var; })
      ; SOp (Copy {
            copy_kind = TmaMulticast;
            src_tensor = b_tensor;
            dst_tensor = smem_b;
            pred_expr = None;
            mbar_var = Some mbar;
            tma_coord_k = Some (Var k_var);
            tma_coord_m = None;
            tma_coord_n = Some (Var (find "col"));
            stage_var = Some stage_var; }) ]

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
            mbar_var = Some mbar;
            tma_coord_k = Some (Var k_var);
            tma_coord_m = Some (Var (find "row"));
            tma_coord_n = None;
            stage_var = Some stage_var; })
      ; SOp (Copy {
            copy_kind = TmaLoad;
            src_tensor = b_tensor;
            dst_tensor = smem_b;
            pred_expr = None;
            mbar_var = Some mbar;
            tma_coord_k = Some (Var k_var);
            tma_coord_m = None;
            tma_coord_n = Some (Var (find "col"));
            stage_var = Some stage_var; }) ]

    | false, _ ->
      [ SOp (Copy {
            copy_kind = CpAsync;
            src_tensor = a_tensor;
            dst_tensor = smem_a;
            pred_expr = None;
            mbar_var = None;
            tma_coord_k = None;
            tma_coord_m = None;
            tma_coord_n = None;
            stage_var = None; })
      ; SOp (Copy {
            copy_kind = CpAsync;
            src_tensor = b_tensor;
            dst_tensor = smem_b;
            pred_expr = None;
            mbar_var = None;
            tma_coord_k = None;
            tma_coord_m = None;
            tma_coord_n = None;
            stage_var = None; })
      ; SOp (Barrier CpAsyncCommitGroup) ]
  in

  let loop_body = let_stage :: copy_ops in
  SWarpGroup (Cluster.Producer, [
    SLetMut (stage_var, Expr (Const (S32, 0l)));
    SFor {
      var = k_var;
      start = i32 0;
      stop = i32 bk;
      step = i32 atom_k;
      dir = Upto;
      unroll = false;
      body = loop_body;
    }
  ])


let construct_consumer_body
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list)
    (smem : (string * packed_tensor) list) =
  let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
  let k_var   = mk_var "k"     S32 ~mut:true () in
  let stage_var = mk_var "stage" S32 ~mut:true () in
  let phase_var = mk_var "phase" S32 ~mut:true () in
  let _, smem_a = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_A") in
  let _, smem_b = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_B") in
  let acc = make_global_tensor "acc" Elemtype.Float32 in
  let mma_kind = match desc.Kernel_desc.family with
    | Kernel_desc.Ampere   -> Sm80Mma
    | Kernel_desc.Hopper   -> Sm90Wgmma
    | Kernel_desc.Blackwell -> Sm100Tcgen05
  in
  (* Step by atom_k, not 1 — avoids full unroll of 32 iterations *)
  let atom_k = match packed_atom_of_desc desc with
    | Atom90 a -> let (_, _, k) = Mma_atom.shape a in k
    | Atom80 a -> let (_, _, k) = Mma_atom.shape a in k
    | Atom100 a -> let (_, _, k) = Mma_atom.shape a in k
  in
  let bk    = desc.Kernel_desc.bk in
  let depth = desc.Kernel_desc.pipeline.Pipeline.depth in
  let let_stage = SAssign (stage_var,
    Expr (Arith (Mod,
      Arith (Div, Var k_var, i32 atom_k),
      i32 depth))) in

  let let_phase = SAssign (phase_var,
    Expr (Arith (Mod,
      Arith (Div, Var k_var, i32 (atom_k * depth)),
      i32 2))) in

  let wait_op =
    if is_tma desc then
      let mbar = find "full_mbar" in
      SOp (Barrier (MbarWaitParity { mbar; phase = Var phase_var }))
    else
      SOp (Barrier CpAsyncWaitAll)
  in
  let mma_op = SOp (Mma {
    mma_kind;
    mma_atom    = packed_atom_of_desc desc;
    tensor_a    = smem_a;
    tensor_b    = smem_b;
    tensor_c    = acc;
    smem_desc_a = None;
    smem_desc_b = None;
    accum_flag  = true;
  }) in
  let arrive_op =
    if is_tma desc then
      let mbar = find "empty_mbar" in
      [ SOp (Barrier (MbarArrive { mbar })) ]
    else []
  in
  let commit_op =
    if is_blackwell desc then
      let mbar = find "full_mbar" in
      [ SOp (TmemCommit { mbar_var = mbar; cta_mask = Some 0b11 }) ]
    else []
  in
  let loop_body =
    [ let_stage; let_phase; wait_op; mma_op ]
    @ arrive_op
    @ commit_op
  in
  SWarpGroup (Cluster.Consumer, [
    SLetMut (stage_var, Expr (Const (S32, 0l)));
    SLetMut (phase_var, Expr (Const (S32, 0l)));
    SFor {
      var   = k_var;
      start = i32 0;
      stop  = i32 bk;
      step  = i32 atom_k;   (* KEY: step by 16, not 1 *)
      dir   = Upto;
      unroll = false;
      body  = loop_body;
    }
  ])


let construct_epilogue_body
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list)
    (smem : (string * packed_tensor) list) =
  let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
  (* Get the shared memory tensors *)
  let _, smem_a = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_A") in
  let _, smem_b = List.find_exn smem ~f:(fun (n,_) -> String.equal n "smem_B") in
  match desc.Kernel_desc.family with
  | Kernel_desc.Blackwell ->
    let tmem_addr = find "tmem_addr" in
    let n_loads   = desc.Kernel_desc.bn / 8 in
    let dst_vars  = List.init n_loads ~f:(fun i ->
      mk_var (Printf.sprintf "reg_%d" i) F32 ~mut:true ()) in
    SWarpGroup (Cluster.Epilogue, [
      SOp (TmemLoad {
        dst_vars;
        src_addr   = Cast (U64, Var tmem_addr);
        col_offset = 0;
      });
    ])
  | Kernel_desc.Hopper ->
    (* For Hopper/SM90, WGMMA accumulates directly to acc_raw in the consumer.
       The epilogue just needs to handle storing results, not compute MMA.
       For now, we skip the epilogue MMA entirely. *)
    SWarpGroup (Cluster.Epilogue, [
      (* TODO: Add store to global memory here when ready *)
      (* For now, just a placeholder - the accumulator is already in acc_frag *)
    ])
  | Kernel_desc.Ampere ->
    (* For Ampere/SM80, the epilogue does the final MMA accumulation *)
    let c = make_global_tensor "C" Elemtype.Float32 in
    SWarpGroup (Cluster.Epilogue, [
      SOp (Mma {
        mma_kind = Sm80Mma;
        mma_atom = Tirix.default_atom_for_kind Sm80Mma;
        tensor_a = smem_a;
        tensor_b = smem_b;
        tensor_c = c;
        smem_desc_a = None;
        smem_desc_b = None;
        accum_flag = false;
      });
    ])

let construct_mbar_init (desc : (_, _, _, _, _, _) Kernel_desc.t) (_vars : var list) =
  if not (is_tma desc) then [] else []

let tmem_alloc_op
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list) =
  if not (is_blackwell desc) then None
  else
    let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
    let addr = find "tmem_addr" in
    Some (SOp (TmemAlloc { addr_var = addr; col_count = desc.Kernel_desc.bn / 2 }))

let tmem_dealloc_op
    (desc : (_, _, _, _, _, _) Kernel_desc.t)
    (vars : var list) =
  if not (is_blackwell desc) then None
  else
    let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
    let addr = find "tmem_addr" in
    Some (SOp (TmemDealloc { addr_var = addr; col_count = desc.Kernel_desc.bn / 2 }))


let lower (desc : (_, _, _, _, _, _) Kernel_desc.t) : tirix =
  let acc_tensor = make_global_tensor "acc" Elemtype.Float32 in
  let tensors = ("acc"  , acc_tensor):: construct_smem_tensors desc in
  let vars = construct_vars desc in
  let params = construct_params desc in
  let helpers  = construct_helpers desc in
  let mbar_init = construct_mbar_init desc vars in
  let producer = construct_producer_body desc vars tensors in
  let consumer = construct_consumer_body desc vars tensors in
  let epilogue = construct_epilogue_body desc vars tensors in
  let find v = List.find_exn vars ~f:(fun x -> String.equal x.var_name v) in
  let warp_id = find "warp_id" in

  (* Get warp IDs from cluster *)
  let prod_warp = Option.value ~default:0
    (Cluster.producer_warp desc.Kernel_desc.cluster) in
  let cons_warp = Option.value ~default:1
    (Cluster.consumer_warp desc.Kernel_desc.cluster) in
  let epi_warps = Cluster.epilogue_warps desc.Kernel_desc.cluster in
  let sched_warp = Cluster.scheduler_warp desc.Kernel_desc.cluster in

  let is_producer = Cmp (Eq, Var warp_id, i32 prod_warp) in
  let is_consumer = Cmp (Eq, Var warp_id, i32 cons_warp) in
  let is_epilogue = List.fold epi_warps ~init:(Const (Bool, false))
    ~f:(fun acc w -> Logic (Or, acc, Cmp (Eq, Var warp_id, i32 w))) in

  let warp_dispatch =
    match sched_warp with
    | Some sched_id ->
      (* Blackwell: include scheduler warp *)
      let is_scheduler = Cmp (Eq, Var warp_id, i32 sched_id) in
      let scheduler_body = SWarpGroup (Cluster.Scheduler, [SEmpty]) in
      SIf (is_producer,
        [ producer ],
        [ SIf (is_consumer,
            [ consumer ],
            [ SIf (is_epilogue,
                [ epilogue ],
                [ SIf (is_scheduler,
                    [ scheduler_body ],
                    [ SEmpty ])])])])
    | None ->
      (* Ampere/Hopper: no scheduler warp *)
      SIf (is_producer,
        [ producer ],
        [ SIf (is_consumer,
            [ consumer ],
            [ SIf (is_epilogue,
                [ epilogue ],
                [ SEmpty ])])])
  in

  let var_decls = List.map vars ~f:(fun v ->
    let init = match v.var_type with
      | Scalar S32 -> Expr (Const (S32, 0l))
      | Scalar U32 -> Expr (Const (U32, 0l))
      | Scalar U64 -> Expr (Const (U64, 0L))
      | Scalar F16 -> Expr (Const (F16, 0.0))
      | Scalar F32 -> Expr (Const (F32, 0.0))
      | Scalar Bool -> Expr (Const (Bool, false))
      | _ -> Expr (Const (S32, 0l))  (* fallback *)
    in
    if v.var_mutable then
      SLetMut (v, init)
    else
      SLet (v, init)
  ) in


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
  ; bm = desc.Kernel_desc.bm
  ; bn = desc.Kernel_desc.bn
  ; bk = desc.Kernel_desc.bk
  ; smem_bytes = Kernel_desc.smem_bytes desc
  ; cluster = desc.Kernel_desc.cluster
  ; pipeline_depth = desc.Kernel_desc.pipeline.Pipeline.depth
  ; body
  ; helpers
  }
