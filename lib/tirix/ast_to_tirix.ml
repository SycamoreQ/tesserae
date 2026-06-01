open Base
open Tesserae_kernel
open Tesserae_pipeline
open Tesserae_core
open Tirix


type ctx = {
  arch : Kernel_ast.arch;
  tensors : (string * Tirix.packed_tensor) list;
  var_counter : int ref;
  full_mbar : Tirix.var option;
  empty_mbar : Tirix.var option;
  pipeline_depth : int;
  (* tile geometry kept for mbarrier byte computation *)
  bm : int;
  bk : int;
  elem_bytes : int;
}

type loop_ctx = {
  k_var : var option;
  m_var : var option;
  n_var : var option;
  stage_var : var option;
}

let empty_loop_ctx = { k_var = None; m_var = None; n_var = None; stage_var = None }

type packed_space = Space : 'a Memspace.space -> packed_space
type packed_elem = Elem : 'a Elemtype.t -> packed_elem

let fresh_id (ctx : ctx) : int =
  let id = !(ctx.var_counter) in
  ctx.var_counter := id + 1;
  id

let lower_space = function
  | Kernel_ast.Global -> Space Memspace.Global
  | Kernel_ast.Shared -> Space Memspace.Shared
  | Kernel_ast.Register -> Space Memspace.Register
  | Kernel_ast.TensorMem -> Space Memspace.Tensor

let lower_elem = function
  | Kernel_ast.F16 -> Elem Elemtype.Float16
  | Kernel_ast.BF16 -> Elem Elemtype.Bfloat16
  | Kernel_ast.F32 -> Elem Elemtype.Float32
  | Kernel_ast.S8 -> Elem Elemtype.Int8
  | Kernel_ast.S32 -> Elem Elemtype.Int32

let elem_byte_width = function
  | Kernel_ast.F16 -> 2
  | Kernel_ast.BF16 -> 2
  | Kernel_ast.F32 -> 4
  | Kernel_ast.S8 -> 1
  | Kernel_ast.S32 -> 4

let make_tensor name (elem : Kernel_ast.elem) (space : Kernel_ast.space)
    layout swizzle : Tirix.packed_tensor =
  let Elem e = lower_elem elem in
  let Space m = lower_space space in
  Tirix.Tensor {
    tensor_name      = name
  ; tensor_id        = Tirix.Type_id.create ()
  ; tensor_elem_type = e
  ; tensor_memspace  = m
  ; tensor_layout    = layout
  ; tensor_swizzle   = swizzle
  }

let rec lookup_tensor (ctx : ctx) (expr : Kernel_ast.tensor_expr)
    : Tirix.packed_tensor =
  let flat = Layout.make (Modes.Int 1) (Modes.Int 1) in
  let sw   = Swizzle.make 0 0 0 in
  match expr with
  | Kernel_ast.Arg (name, elem, space) ->
    (match List.find ctx.tensors ~f:(fun (n,_) -> String.equal n name) with
     | Some (_, t) -> t
     | None -> make_tensor name elem space flat sw)
  | Kernel_ast.Smem (name, elem, shape) ->
    let layout = Layout.make
      (Modes.Tuple [Modes.Int shape.Kernel_ast.m; Modes.Int shape.Kernel_ast.k])
      (Modes.Tuple [Modes.Int 1; Modes.Int shape.Kernel_ast.m])
    in
    (match List.find ctx.tensors ~f:(fun (n,_) -> String.equal n name) with
     | Some (_, t) -> t
     | None -> make_tensor name elem Kernel_ast.Shared layout sw)
  | Kernel_ast.Tile (inner, _)      -> lookup_tensor ctx inner
  | Kernel_ast.LocalTile (inner, _) -> lookup_tensor ctx inner

let memspace_name (Tirix.Tensor t) = Memspace.name t.tensor_memspace

let infer_copy_kind (ctx : ctx)
    (src : Tirix.packed_tensor) (dst : Tirix.packed_tensor) : Tirix.copy_kind =
  match memspace_name src, memspace_name dst with
  | "global", "shared" ->
    (match ctx.arch with
     | Kernel_ast.SM80 -> Tirix.CpAsync
     | Kernel_ast.SM90a -> Tirix.TmaLoad
     | Kernel_ast.SM100a -> Tirix.TmaMulticast)
  | "shared", "global" -> Tirix.SmemToGlobal
  | "shared", "register" -> Tirix.SmemToReg
  | "register", "shared" -> Tirix.RegToSmem
  | "register", _  -> Tirix.RegToSmem
  | _ -> Tirix.CpAsync

let infer_mma_kind (ctx : ctx) : Tirix.mma_kind =
  match ctx.arch with
  | Kernel_ast.SM80    -> Tirix.Sm80Mma
  | Kernel_ast.SM90a   -> Tirix.Sm90Wgmma
  | Kernel_ast.SM100a  -> Tirix.Sm100Tcgen05

let lower_mask (m : Kernel_ast.mask) : bool Tirix.expr =
  let v = { Tirix.var_name = m.Kernel_ast.coord_var
           ; var_id = 0; var_type = Tirix.Scalar Tirix.S32; var_mutable = false } in
  List.foldi m.Kernel_ast.bounds ~init:(Tirix.Const (Tirix.Bool, true))
    ~f:(fun _i acc bound ->
      Logic (And, acc,
        Cmp (Lt, Tirix.Var v, Tirix.Const (S32, Int32.of_int_exn bound))))

let lower_barrier = function
  | Kernel_ast.ThreadSync  -> Tirix.CtaSync
  | Kernel_ast.ClusterSync -> Tirix.ClusterArrive
  | Kernel_ast.MbarFull  var ->
    let v = { Tirix.var_name=var; var_id=0
            ; var_type=Tirix.Scalar Tirix.S32; var_mutable=false } in
    Tirix.MbarWaitParity { mbar=v; phase=Tirix.Const (Tirix.S32, 0l) }
  | Kernel_ast.MbarEmpty var ->
    let v = { Tirix.var_name=var; var_id=0
            ; var_type=Tirix.Scalar Tirix.S32; var_mutable=false } in
    Tirix.MbarArrive { mbar=v }

let lower_pred (p : Kernel_ast.pred_expr) : bool Tirix.expr =
  match p with
  | Kernel_ast.WarpIs n ->
    Cmp (Eq, Tirix.Builtin Tirix.WarpId, Const (Tirix.S32, Int32.of_int_exn n))
  | Kernel_ast.WarpIn ns ->
    List.fold ns ~init:(Tirix.Const (Tirix.Bool, false))
      ~f:(fun acc n ->
        Logic (Or, acc,
          Cmp (Eq, Tirix.Builtin Tirix.WarpId,
               Tirix.Const (Tirix.S32, Int32.of_int_exn n))))
  | Kernel_ast.InBounds (var, bounds) ->
    lower_mask { Kernel_ast.coord_var = var; bounds }

let lower_params (ctx : ctx) (k : Kernel_ast.kernel) : Tirix.param list =
  let flat = Layout.make (Modes.Int 1) (Modes.Int 1) in
  let sw   = Swizzle.make 0 0 0 in
  List.map k.Kernel_ast.args ~f:(fun (name, elem, space) ->
    let is_tma = match ctx.arch, space with
      | (Kernel_ast.SM90a | Kernel_ast.SM100a), Kernel_ast.Global -> true
      | _ -> false
    in
    { Tirix.param_name   = name
    ; param_tensor = make_tensor name elem space flat sw
    ; param_is_tma = is_tma
    })

let lower_family = function
  | Kernel_ast.SM80    -> Kernel_desc.Ampere
  | Kernel_ast.SM90a   -> Kernel_desc.Hopper
  | Kernel_ast.SM100a  -> Kernel_desc.Blackwell

let rec collect_tensors (stmt : Kernel_ast.stmt)
    (acc : (string * Tirix.packed_tensor) list)
    : (string * Tirix.packed_tensor) list =
  let flat = Layout.make (Modes.Int 1) (Modes.Int 1) in
  let sw   = Swizzle.make 0 0 0 in
  match stmt with
  | Kernel_ast.Load (src, dst, _) | Kernel_ast.Store (src, dst, _) ->
    let tensors = [] in
    let tensors = match src with
      | Kernel_ast.Arg (name, elem, space) ->
        (name, make_tensor name elem space flat sw) :: tensors
      | Kernel_ast.Smem (name, elem, shape) ->
        let layout = Layout.make
          (Modes.Tuple [Modes.Int shape.Kernel_ast.m; Modes.Int shape.Kernel_ast.k])
          (Modes.Tuple [Modes.Int 1; Modes.Int shape.Kernel_ast.m]) in
        (name, make_tensor name elem Kernel_ast.Shared layout sw) :: tensors
      | _ -> tensors
    in
    let tensors = match dst with
      | Kernel_ast.Arg (name, elem, space)
        when not (List.exists tensors ~f:(fun (n,_) -> String.equal n name)) ->
        (name, make_tensor name elem space flat sw) :: tensors
      | Kernel_ast.Smem (name, elem, shape)
        when not (List.exists tensors ~f:(fun (n,_) -> String.equal n name)) ->
        let layout = Layout.make
          (Modes.Tuple [Modes.Int shape.Kernel_ast.m; Modes.Int shape.Kernel_ast.k])
          (Modes.Tuple [Modes.Int 1; Modes.Int shape.Kernel_ast.m]) in
        (name, make_tensor name elem Kernel_ast.Shared layout sw) :: tensors
      | _ -> tensors
    in
    tensors @ acc
  | Kernel_ast.Mma (a, b, c) ->
    let collect_one expr acc = match expr with
      | Kernel_ast.Arg (name, elem, space)
        when not (List.exists acc ~f:(fun (n,_) -> String.equal n name)) ->
        (name, make_tensor name elem space flat sw) :: acc
      | Kernel_ast.Smem (name, elem, shape)
        when not (List.exists acc ~f:(fun (n,_) -> String.equal n name)) ->
        let layout = Layout.make
          (Modes.Tuple [Modes.Int shape.Kernel_ast.m; Modes.Int shape.Kernel_ast.k])
          (Modes.Tuple [Modes.Int 1; Modes.Int shape.Kernel_ast.m]) in
        (name, make_tensor name elem Kernel_ast.Shared layout sw) :: acc
      | _ -> acc
    in
    collect_one c (collect_one b (collect_one a acc))
  | Kernel_ast.For (_, _, _, body) ->
    List.fold body ~init:acc ~f:(fun acc s -> collect_tensors s acc)
  | Kernel_ast.Pipeline (_, stmts) ->
    List.fold stmts ~init:acc ~f:(fun acc s -> collect_tensors s acc)
  | Kernel_ast.If (_, thn, els) ->
    let acc = List.fold thn ~init:acc ~f:(fun acc s -> collect_tensors s acc) in
    List.fold els ~init:acc ~f:(fun acc s -> collect_tensors s acc)
  | Kernel_ast.Seq stmts ->
    List.fold stmts ~init:acc ~f:(fun acc s -> collect_tensors s acc)
  | Kernel_ast.Barrier _ -> acc

let rec has_tma_load (arch : Kernel_ast.arch) (stmt : Kernel_ast.stmt) : bool =
  match arch with
  | Kernel_ast.SM80 -> false
  | Kernel_ast.SM90a | Kernel_ast.SM100a ->
    match stmt with
    | Kernel_ast.Load (Kernel_ast.Arg (_, _, Kernel_ast.Global),
                       Kernel_ast.Smem _, _) -> true
    | Kernel_ast.For (_, _, _, body) -> List.exists body ~f:(has_tma_load arch)
    | Kernel_ast.Pipeline (_, stmts) -> List.exists stmts ~f:(has_tma_load arch)
    | Kernel_ast.If (_, thn, els) ->
      List.exists thn ~f:(has_tma_load arch) ||
      List.exists els ~f:(has_tma_load arch)
    | Kernel_ast.Seq stmts -> List.exists stmts ~f:(has_tma_load arch)
    | _ -> false

let make_mbar_tensors (depth : int) : (string * Tirix.packed_tensor) list =
  (* Layout size = depth so emit_shared_storage emits uint64_t full_mbar[depth] *)
  let flat = Layout.make (Modes.Int depth) (Modes.Int 1) in
  let sw   = Swizzle.make 0 0 0 in
  let make name = Tirix.Tensor {
    tensor_name = name
  ; tensor_id = Tirix.Type_id.create ()
  ; tensor_elem_type = Elemtype.Int32
  ; tensor_memspace = Memspace.Shared
  ; tensor_layout = flat
  ; tensor_swizzle = sw
  } in
  [ ("full_mbar",  make "full_mbar")
  ; ("empty_mbar", make "empty_mbar") ]

let make_mbar_var name : Tirix.var =
  { Tirix.var_name = name; var_id = 0
  ; var_type = Tirix.Scalar Tirix.U64; var_mutable = false }


let rec convert_stmt (ctx : ctx) (loop_ctx : loop_ctx)
    (stmt : Kernel_ast.stmt) : Tirix.stmt =
  match stmt with

  | Kernel_ast.Load (src_expr, dst_expr, mask_opt) ->
    let src  = lookup_tensor ctx src_expr in
    let dst  = lookup_tensor ctx dst_expr in
    let kind = infer_copy_kind ctx src dst in
    let pred = Option.map mask_opt ~f:lower_mask in
    let is_tma = match kind with
      | Tirix.TmaLoad | Tirix.TmaMulticast -> true
      | _ -> false
    in
    let mbar_var = if is_tma then ctx.full_mbar else None in


    let arrive_stmts = match mbar_var with
      | None -> []
      | Some mbar ->
        let bytes = ctx.bm * ctx.bk * ctx.elem_bytes in
        [ Tirix.SOp (Tirix.Barrier (Tirix.MbarArriveExpect {
            mbar
          ; bytes = Tirix.Const (Tirix.S32, Int32.of_int_exn bytes)
          })) ]
    in

    let copy_stmt = Tirix.SOp (Tirix.Copy {
        copy_kind   = kind
      ; src_tensor  = src
      ; dst_tensor  = dst
      ; pred_expr   = pred
      ; mbar_var
      ; tma_coord_k = Option.map loop_ctx.k_var   ~f:(fun v -> Tirix.Var v)
      ; tma_coord_m = Option.map loop_ctx.m_var   ~f:(fun v -> Tirix.Var v)
      ; tma_coord_n = Option.map loop_ctx.n_var   ~f:(fun v -> Tirix.Var v)
      ; stage_var   = loop_ctx.stage_var
      }) in


    let wait_stmts = match mbar_var with
      | None -> []
      | Some mbar ->
        [ Tirix.SOp (Tirix.Barrier (Tirix.MbarWaitParity {
            mbar; phase = Tirix.Const (Tirix.S32, 0l) })) ]
    in

    (match arrive_stmts @ [copy_stmt] @ wait_stmts with
     | [s] -> s
     | many -> Tirix.SSeq many)

  | Kernel_ast.Store (src_expr, dst_expr, mask_opt) ->
    let src  = lookup_tensor ctx src_expr in
    let dst  = lookup_tensor ctx dst_expr in
    let kind = infer_copy_kind ctx src dst in
    let pred = Option.map mask_opt ~f:lower_mask in

    let copy_stmt = Tirix.SOp (Tirix.Copy {
        copy_kind   = kind
      ; src_tensor  = src
      ; dst_tensor  = dst
      ; pred_expr   = pred
      ; mbar_var    = None
      ; tma_coord_k = Option.map loop_ctx.k_var ~f:(fun v -> Tirix.Var v)
      ; tma_coord_m = Option.map loop_ctx.m_var ~f:(fun v -> Tirix.Var v)
      ; tma_coord_n = Option.map loop_ctx.n_var ~f:(fun v -> Tirix.Var v)
      ; stage_var   = loop_ctx.stage_var
      }) in

    let arrive_empty = match ctx.empty_mbar with
      | Some mbar when (match kind with Tirix.SmemToGlobal -> true | _ -> false) ->
        [ Tirix.SOp (Tirix.Barrier (Tirix.MbarArrive { mbar })) ]
      | _ -> []
    in

    (match [copy_stmt] @ arrive_empty with
     | [s] -> s
     | many -> Tirix.SSeq many)

  | Kernel_ast.Mma (a_expr, b_expr, c_expr) ->
    let kind = infer_mma_kind ctx in
    Tirix.SOp (Tirix.Mma {
      mma_kind    = kind
    ; mma_atom    = Tirix.default_atom_for_kind kind
    ; tensor_a    = lookup_tensor ctx a_expr
    ; tensor_b    = lookup_tensor ctx b_expr
    ; tensor_c    = lookup_tensor ctx c_expr
    ; smem_desc_a = None
    ; smem_desc_b = None
    ; accum_flag  = true
    })

  | Kernel_ast.For (var_name, start_val, stop_val, body_stmts) ->
    let loop_var = {
      Tirix.var_name  = var_name
    ; var_id          = fresh_id ctx
    ; var_type        = Tirix.Scalar Tirix.S32
    ; var_mutable     = true
    } in
    let new_loop_ctx = match var_name with
      | "k" | "k_loop"             -> { loop_ctx with k_var     = Some loop_var }
      | "m" | "row"   | "block_m"  -> { loop_ctx with m_var     = Some loop_var }
      | "n" | "col"   | "block_n"  -> { loop_ctx with n_var     = Some loop_var }
      | "stage" | "s"              -> { loop_ctx with stage_var = Some loop_var }
      | _                          -> loop_ctx
    in
    Tirix.SFor {
      var    = loop_var
    ; start  = Tirix.Const (Tirix.S32, Int32.of_int_exn start_val)
    ; stop   = Tirix.Const (Tirix.S32, Int32.of_int_exn stop_val)
    ; step   = Tirix.Const (Tirix.S32, 1l)
    ; dir    = Tirix.Upto
    ; unroll = false
    ; body   = List.map body_stmts ~f:(convert_stmt ctx new_loop_ctx)
    }

  | Kernel_ast.Pipeline (desc, stmts) ->
    Tirix.SPipeline {
      stages   = desc.Kernel_ast.stages
    ; prologue = []
    ; mainloop = List.map stmts ~f:(convert_stmt ctx loop_ctx)
    ; epilogue = []
    }

  | Kernel_ast.Barrier b ->
    Tirix.SOp (Tirix.Barrier (lower_barrier b))

  | Kernel_ast.If (pred, thn, els) ->
    Tirix.SIf (
      lower_pred pred,
      List.map thn ~f:(convert_stmt ctx loop_ctx),
      List.map els ~f:(convert_stmt ctx loop_ctx))

  | Kernel_ast.Seq stmts ->
    Tirix.SSeq (List.map stmts ~f:(convert_stmt ctx loop_ctx))


let lower (k : Kernel_ast.kernel) : Tirix.tirix =
  let tensors       = collect_tensors k.Kernel_ast.body [] in
  let needs_tma     = has_tma_load k.Kernel_ast.arch k.Kernel_ast.body in
  let pipeline_depth = k.Kernel_ast.stages in

  let (full_mbar_var, empty_mbar_var, mbar_tensors) =
    if needs_tma then
      (Some (make_mbar_var "full_mbar"),
       Some (make_mbar_var "empty_mbar"),
       make_mbar_tensors pipeline_depth)
    else
      (None, None, [])
  in

  let ctx = {
    arch           = k.Kernel_ast.arch
  ; tensors        = tensors @ mbar_tensors
  ; var_counter    = ref 0
  ; full_mbar      = full_mbar_var
  ; empty_mbar     = empty_mbar_var
  ; pipeline_depth
  ; bm             = k.Kernel_ast.tile.Kernel_ast.m
  ; bk             = k.Kernel_ast.tile.Kernel_ast.k
  ; elem_bytes     = elem_byte_width k.Kernel_ast.elem
  } in

  (* __syncthreads() after mbarrier init loop (warp 0 lane 0) so that
     all warps see initialized mbar state before issuing cp.async.bulk. *)
  let sync_after_init =
    if needs_tma then [ Tirix.SOp (Tirix.Barrier Tirix.CtaSync) ]
    else []
  in

  let body_stmt = convert_stmt ctx empty_loop_ctx k.Kernel_ast.body in

  let cluster = match k.Kernel_ast.arch with
    | Kernel_ast.SM80 ->
      Cluster.make { Cluster.x=1; y=1; z=1 } 4
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Epilogue); (3, Cluster.Epilogue) ]
    | Kernel_ast.SM90a ->
      Cluster.make { Cluster.x=2; y=1; z=1 } 8
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Consumer); (3, Cluster.Consumer)
        ; (4, Cluster.Consumer); (5, Cluster.Epilogue)
        ; (6, Cluster.Epilogue); (7, Cluster.Epilogue) ]
    | Kernel_ast.SM100a ->
      Cluster.make { Cluster.x=2; y=1; z=1 } 6
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Epilogue); (3, Cluster.Epilogue)
        ; (4, Cluster.Epilogue); (5, Cluster.Scheduler) ]
  in
  { Tirix.name           = k.Kernel_ast.name
  ; family               = lower_family k.Kernel_ast.arch
  ; params               = lower_params ctx k
  ; tensors              = ctx.tensors
  ; smem_bytes           = 0
  ; pipeline_depth
  ; bm                   = k.Kernel_ast.tile.Kernel_ast.m
  ; bn                   = k.Kernel_ast.tile.Kernel_ast.n
  ; bk                   = k.Kernel_ast.tile.Kernel_ast.k
  ; cluster
  ; body                 = sync_after_init @ [ body_stmt ]
  ; helpers              = []
  }
