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
  bm : int;
  bn : int;
  bk : int;
  elem_bytes : int;
  roles : (int * Cluster.warp_role) list;
  cta_count : int;
  (* Whether the kernel takes scalar M / N parameters (needed by the
     persistent pair-tile decomposition). *)
  has_m : bool;
  has_n : bool;
  (* The single decision point for 2-CTA TMA multicast. When false every CTA
     loads its own tiles (TmaLoad, no multicast commit). Flip it when the 2-CTA
     sharing optimisation is implemented. *)
  multicast : bool;
}

type loop_ctx = {
  k_var : var option;
  (* Trip count of the innermost k loop, needed to make stage/parity
     continuous across tile loops: global k = tile * k_stop + k. *)
  k_stop : int option;
  m_var : var option;
  n_var : var option;
  stage_var : var option;
  tile_var : var option;
  in_warp_group : bool;
}

let empty_loop_ctx =
  { k_var = None; k_stop = None; m_var = None; n_var = None; stage_var = None
  ; tile_var = None; in_warp_group = false }

(* The pipeline's mbarrier phase state survives across iterations of an outer
   tile loop, so every stage/parity expression must be derived from the global
   k index. Deriving them from the (per-tile-reset) loop variable desyncs the
   waits whenever k_tiles is not a multiple of the pipeline depth, wedging the
   producer and consumer on each other's barriers. *)
let global_k (loop_ctx : loop_ctx) (kv : var) : int32 Tirix.expr =
  match loop_ctx.tile_var, loop_ctx.k_stop with
  | Some tv, Some ks ->
    Tirix.Arith (Tirix.Arith.Add,
      Tirix.Arith (Tirix.Arith.Mul, Tirix.Var tv,
        Tirix.Const (Tirix.S32, Int32.of_int_exn ks)),
      Tirix.Var kv)
  | _ -> Tirix.Var kv

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

(* A shared tile is described by a shape whose row extent is m for A-like
   operands and n for B-like operands (the other one is 0). *)
let smem_rows shape =
  if shape.Kernel_ast.m > 0 then shape.Kernel_ast.m else shape.Kernel_ast.n

let smem_layout shape =
  let rows = smem_rows shape in
  Layout.make
    (Modes.Tuple [Modes.Int rows; Modes.Int shape.Kernel_ast.k])
    (Modes.Tuple [Modes.Int 1; Modes.Int rows])

let make_tensor name (elem : Kernel_ast.elem) (space : Kernel_ast.space)
    layout swizzle : Tirix.packed_tensor =
  let Elem e = lower_elem elem in
  let Space m = lower_space space in
  Tirix.Tensor {
    tensor_name = name
  ; tensor_id = Tirix.Type_id.create ()
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
    let layout = smem_layout shape in
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
     | Kernel_ast.SM100a ->
       (* tcgen05.mma cta_group::2 shards B along N across the CTA pair:
          each CTA loads its own N-half into its own smem, so B is never
          multicast. The rank offset is applied in the coord lowering
          below (see coord_n). *)
       Tirix.TmaLoad)

  | "shared", "global" -> Tirix.SmemToGlobal

  | "shared", "register" -> Tirix.SmemToReg

  | "register", "shared" -> Tirix.RegToSmem

  | "register", _  -> Tirix.RegToSmem

  | _ -> Tirix.CpAsync

let infer_mma_kind (ctx : ctx) : Tirix.mma_kind =
  match ctx.arch with
  | Kernel_ast.SM80 -> Tirix.Sm80Mma
  | Kernel_ast.SM90a -> Tirix.Sm90Wgmma
  | Kernel_ast.SM100a -> Tirix.Sm100Tcgen05

let lower_mask (m : Kernel_ast.mask) : bool Tirix.expr =
  let v = { Tirix.var_name = m.Kernel_ast.coord_var
           ; var_id = 0; var_type = Tirix.Scalar Tirix.S32; var_mutable = false } in
  List.foldi m.Kernel_ast.bounds ~init:(Tirix.Const (Tirix.Bool, true))
    ~f:(fun _i acc bound ->
      Logic (And, acc,
        Cmp (Lt, Tirix.Var v, Tirix.Const (S32, Int32.of_int_exn bound))))

let lower_barrier = function
  | Kernel_ast.MbarInit (mbar, count) ->
      let v = { Tirix.var_name = mbar; var_id = 0; var_type = Tirix.Scalar Tirix.U64; var_mutable = false } in
      Tirix.MbarInit { mbar = v; count }

  | Kernel_ast.MbarWaitParity (mbar, phase) ->
      let v = { Tirix.var_name = mbar; var_id = 0; var_type = Tirix.Scalar Tirix.U64; var_mutable = false } in
      Tirix.MbarWaitParity { mbar = v; phase = Tirix.Const (Tirix.S32, Int32.of_int_exn phase) }
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

  | Kernel_ast.Mbarrier _ -> Tirix.Const (Tirix.Bool, true)

let rec is_tma_source name = function
  | Kernel_ast.Load (Arg (n, _, Global), Smem _, _) when String.equal n name -> true

  | Kernel_ast.For (_, _, _, body) -> List.exists body ~f:(is_tma_source name)

  | Kernel_ast.Pipeline (_, body) -> List.exists body ~f:(is_tma_source name)

  | Kernel_ast.Seq ss -> List.exists ss ~f:(is_tma_source name)

  | Kernel_ast.If (_, thn, els) ->
    List.exists thn ~f:(is_tma_source name) || List.exists els ~f:(is_tma_source name)

  | _ -> false

(* A global tensor that is the DESTINATION of a store (e.g. the GEMM's C
   output) gets a TMA descriptor parameter too, so the epilogue can store
   through cp.async.bulk.tensor instead of raw pointer writes. *)
let rec is_tma_store_dest (name : string) (stmt : Kernel_ast.stmt) : bool =
  match stmt with
  | Kernel_ast.Store (_, Kernel_ast.Arg (n, _, Kernel_ast.Global), _) ->
    String.equal n name
  | Kernel_ast.For (_, _, _, body) -> List.exists body ~f:(is_tma_store_dest name)
  | Kernel_ast.Pipeline (_, stmts) -> List.exists stmts ~f:(is_tma_store_dest name)
  | Kernel_ast.If (_, thn, els) ->
    List.exists thn ~f:(is_tma_store_dest name)
    || List.exists els ~f:(is_tma_store_dest name)
  | Kernel_ast.Seq ss -> List.exists ss ~f:(is_tma_store_dest name)
  | _ -> false

let lower_params (ctx : ctx) (k : Kernel_ast.kernel) : Tirix.param list =
  let flat = Layout.make (Modes.Int 1) (Modes.Int 1) in
  let sw   = Swizzle.make 0 0 0 in
  List.map k.Kernel_ast.args ~f:(fun (name, elem, space, _dir) ->
    let is_tma = match ctx.arch, space with
       | Kernel_ast.SM90a, Global -> is_tma_source name k.body
       | Kernel_ast.SM100a, Global ->
         is_tma_source name k.body || is_tma_store_dest name k.body
       | _ -> false
     in
    { Tirix.param_name = name
    ; param_tensor = make_tensor name elem space flat sw
    ; param_is_tma = is_tma
    })

let lower_family = function
  | Kernel_ast.SM80 -> Kernel_desc.Ampere
  | Kernel_ast.SM90a -> Kernel_desc.Hopper
  | Kernel_ast.SM100a -> Kernel_desc.Blackwell

let rec collect_tensors (stmt : Kernel_ast.stmt)
    (acc : (string * Tirix.packed_tensor) list)
    : (string * Tirix.packed_tensor) list =
  let flat = Layout.make (Modes.Int 1) (Modes.Int 1) in
  let sw   = Swizzle.make 0 0 0 in
  let add_unique name tensor tensors =
    if List.exists tensors ~f:(fun (n, _) -> String.equal n name) then tensors
    else (name, tensor) :: tensors
  in
  match stmt with
  | Kernel_ast.Load (src, dst, _) | Kernel_ast.Store (src, dst, _) ->
    let tensors = acc in
    let tensors = match src with
      | Kernel_ast.Arg (name, elem, space) ->
        add_unique name (make_tensor name elem space flat sw) tensors

      | Kernel_ast.Smem (name, elem, shape) ->
        add_unique name
          (make_tensor name elem Kernel_ast.Shared (smem_layout shape) sw) tensors

      | _ -> tensors
    in
    let tensors = match dst with
      | Kernel_ast.Arg (name, elem, space) ->
        add_unique name (make_tensor name elem space flat sw) tensors

      | Kernel_ast.Smem (name, elem, shape) ->
        add_unique name
          (make_tensor name elem Kernel_ast.Shared (smem_layout shape) sw) tensors

      | _ -> tensors
    in
    tensors

  | Kernel_ast.Mma (a, b, c) ->
    let collect_one expr acc = match expr with
      | Kernel_ast.Arg (name, elem, space)
        when not (List.exists acc ~f:(fun (n,_) -> String.equal n name)) ->
        (name, make_tensor name elem space flat sw) :: acc

      | Kernel_ast.Smem (name, elem, shape)
        when not (List.exists acc ~f:(fun (n,_) -> String.equal n name)) ->
        (name, make_tensor name elem Kernel_ast.Shared (smem_layout shape) sw) :: acc

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
  let sw = Swizzle.make 0 0 0 in
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

(* If a warp predicate selects warps that all share one role, return it. *)
let group_role_of_pred (ctx : ctx) (pred : Kernel_ast.pred_expr)
    : Cluster.warp_role option =
  let role_of n = List.Assoc.find ctx.roles n ~equal:Int.equal in
  match pred with
  | Kernel_ast.WarpIs n -> role_of n
  | Kernel_ast.WarpIn (n :: rest) ->
    (match role_of n with
     | Some r
       when List.for_all rest ~f:(fun m ->
              match role_of m with
              | Some r' -> Poly.equal r r'
              | None -> false) -> Some r
     | _ -> None)
  | _ -> None


(* ---- Persistent pair-tile decomposition (2-CTA Blackwell) ----

   The launch grid holds num_pairs = (gridDim.x/2) * gridDim.y resident
   CTA pairs. Each pair walks the flattened pair-tile index
     idx = pair_base + tile * num_pairs,
     pair_base = blockIdx.x/2 + (gridDim.x/2) * blockIdx.y
   striding so that all pair tiles are covered regardless of the grid
   shape. The index is decomposed as
     tile_m = idx / n_blocks,  tile_n = idx % n_blocks
   with n_blocks = ceil(N / pair_n) and m_blocks = ceil(M / pair_m), where
   pair_m/pair_n are the CTA-pair tile extents. Requires scalar M and N
   kernel params; returns None otherwise (callers fall back to the static
   per-CTA coordinate mapping). *)
type pair_decomp = {
  pd_idx : int32 Tirix.expr;      (* flattened pair-tile index *)
  pd_tile_m : int32 Tirix.expr;
  pd_tile_n : int32 Tirix.expr;
  pd_total : int32 Tirix.expr;    (* m_blocks * n_blocks *)
  pd_bound : int32 Tirix.expr;    (* ceil(total / num_pairs) *)
}

let pa_c (n : int) : int32 Tirix.expr =
  Tirix.Const (Tirix.S32, Int32.of_int_exn n)
let pa_add a b = Tirix.Arith (Tirix.Arith.Add, a, b)
let pa_sub a b = Tirix.Arith (Tirix.Arith.Sub, a, b)
let pa_mul a b = Tirix.Arith (Tirix.Arith.Mul, a, b)
let pa_div a b = Tirix.Arith (Tirix.Arith.Div, a, b)
let pa_mod a b = Tirix.Arith (Tirix.Arith.Mod, a, b)

let pair_decomp_of (ctx : ctx) (tile_var : Tirix.var option)
  : pair_decomp option =
  if Poly.(ctx.arch <> Kernel_ast.SM100a) || ctx.cta_count < 2
     || not ctx.has_m || not ctx.has_n then None
  else
    let pair_m = 2 * ctx.bm and pair_n = ctx.bn in
    let half_gx = pa_div (Tirix.Builtin (Tirix.GridDim Tirix.X)) (pa_c 2) in
    let num_pairs =
      pa_mul half_gx (Tirix.Builtin (Tirix.GridDim Tirix.Y)) in
    let pair_base =
      pa_add (pa_div (Tirix.Builtin (Tirix.BlockIdx Tirix.X)) (pa_c 2))
        (pa_mul half_gx (Tirix.Builtin (Tirix.BlockIdx Tirix.Y))) in
    let idx =
      match tile_var with
      | Some tv -> pa_add pair_base (pa_mul (Tirix.Var tv) num_pairs)
      | None -> pair_base
    in
    let n_blocks =
      pa_div (pa_add (Tirix.Builtin (Tirix.KParam "N")) (pa_c (pair_n - 1)))
        (pa_c pair_n) in
    let m_blocks =
      pa_div (pa_add (Tirix.Builtin (Tirix.KParam "M")) (pa_c (pair_m - 1)))
        (pa_c pair_m) in
    let total = pa_mul m_blocks n_blocks in
    Some { pd_idx = idx
         ; pd_tile_m = pa_div idx n_blocks
         ; pd_tile_n = pa_mod idx n_blocks
         ; pd_total = total
         ; pd_bound = pa_div (pa_add total (pa_sub num_pairs (pa_c 1))) num_pairs }

(* A/B/epilogue coordinates for a persistent pair tile. The CTA's half of
   the pair tile is selected by its rank in the cluster. *)
let pair_coord ~(tile_dim : int32 Tirix.expr) ~(extent : int)
    : int32 Tirix.expr =
  pa_add (pa_mul tile_dim (pa_c extent))
    (pa_mul (Tirix.Builtin Tirix.ClusterCtaId) (pa_c (extent / 2)))

let rec convert_stmt (ctx : ctx) (loop_ctx : loop_ctx) ?(skip_empty_wait=false)
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
        let bytes = match dst_expr with
          | Kernel_ast.Smem (_, _, shape) ->
            smem_rows shape * shape.Kernel_ast.k * ctx.elem_bytes

          | _ -> ctx.bm * ctx.bk * ctx.elem_bytes
        in
        [ Tirix.SOp (Tirix.Barrier (Tirix.MbarArriveExpect {
            mbar
          ; bytes = Tirix.Const (Tirix.S32, Int32.of_int_exn bytes)
          })) ]   (* TMA load already performs an implicit arrive *)
    in
    let tma_coord_scaled var_opt tensor_expr dim_f =
      match var_opt, tensor_expr with
      | Some v, Kernel_ast.Smem (_, _, shape) ->
        let strip = dim_f shape in
        if strip > 0 then
          Some (Tirix.Arith (Tirix.Arith.Mul, Tirix.Var v,
            Tirix.Const (Tirix.S32, Int32.of_int_exn strip)))
        else
          Some (Tirix.Var v)
      | Some v, _ -> Some (Tirix.Var v)
      | None, _ -> None
    in

    (* Each CTA owns one output tile: with no explicit m/n loop, the tile row
       of A comes from blockIdx.x and the tile row of B (output column) from
       blockIdx.y. With an outer tile loop, tile t of CTA bx owns row
       (bx + t * gridDim.x): gridDim.x is the per-cluster CTA row count. *)
    let block_coord builtin strip =
      let base =
        match loop_ctx.tile_var, builtin with
        | Some tv, Tirix.BlockIdx Tirix.X ->
          Tirix.Arith (Tirix.Arith.Add, Tirix.Builtin builtin,
            Tirix.Arith (Tirix.Arith.Mul, Tirix.Var tv,
              Tirix.Builtin (Tirix.GridDim Tirix.X)))
        | _ -> Tirix.Builtin builtin
      in
      Some (Tirix.Arith (Tirix.Arith.Mul, base,
              Tirix.Const (Tirix.S32, Int32.of_int_exn strip)))
    in

    (* TMA k coordinate stays LOCAL: every output tile re-reduces over the
       full K range, so the coordinate restarts with the k loop. Only the
       stage rotation and the wait parities (below) use the global k index,
       because those track mbarrier phase state that persists across tiles. *)
    let tma_coord_k =
      tma_coord_scaled loop_ctx.k_var dst_expr (fun s -> s.Kernel_ast.k)
    in
    let coord_m =
      match tma_coord_scaled loop_ctx.m_var dst_expr (fun s -> s.Kernel_ast.m) with
      | Some _ as c -> c
      | None ->
        (match dst_expr with
        | Kernel_ast.Smem (_, _, shape) when is_tma && shape.Kernel_ast.m > 0 ->
          if Poly.(ctx.arch = Kernel_ast.SM100a) && ctx.cta_count > 1 then
            (* A is sharded along M across the pair; the row block comes
               from the persistent pair-tile index. *)
            (match pair_decomp_of ctx loop_ctx.tile_var with
             | Some pd ->
               Some (pair_coord ~tile_dim:pd.pd_tile_m
                       ~extent:(2 * shape.Kernel_ast.m))
             | None -> block_coord (Tirix.BlockIdx Tirix.X) shape.Kernel_ast.m)
          else block_coord (Tirix.BlockIdx Tirix.X) shape.Kernel_ast.m
        | _ -> None)
    in
    let coord_n =
      match tma_coord_scaled loop_ctx.n_var dst_expr (fun s -> s.Kernel_ast.n) with
      | Some _ as c -> c
      | None ->
        (match dst_expr with
        | Kernel_ast.Smem (_, _, shape) when is_tma && shape.Kernel_ast.n > 0 ->
          if Poly.(ctx.arch = Kernel_ast.SM100a) && ctx.cta_count > 1
             && shape.Kernel_ast.m = 0 then
            (* 2-CTA pair tile: B is sharded along N. tile_n comes from the
               persistent pair-tile index; this CTA loads its rank's half. *)
            (match pair_decomp_of ctx loop_ctx.tile_var with
             | Some pd ->
               Some (pair_coord ~tile_dim:pd.pd_tile_n
                       ~extent:(2 * shape.Kernel_ast.n))
             | None ->
               Some (Tirix.Arith (Tirix.Arith.Add,
                 Tirix.Arith (Tirix.Arith.Mul, Tirix.Builtin (Tirix.BlockIdx Tirix.Y),
                   Tirix.Const (Tirix.S32,
                     Int32.of_int_exn (2 * shape.Kernel_ast.n))),
                 Tirix.Arith (Tirix.Arith.Mul, Tirix.Builtin Tirix.ClusterCtaId,
                   Tirix.Const (Tirix.S32, Int32.of_int_exn shape.Kernel_ast.n)))))
          else block_coord (Tirix.BlockIdx Tirix.Y) shape.Kernel_ast.n
        | _ -> None)
    in

    let copy_stmt = Tirix.SOp (Tirix.Copy {
        copy_kind = kind
      ; src_tensor = src
      ; dst_tensor = dst
      ; pred_expr = pred
      ; mbar_var
      ; tma_coord_k = tma_coord_k
      ; tma_coord_m = coord_m
      ; tma_coord_n = coord_n
      ; stage_var = loop_ctx.stage_var
      }) in

    let phase_expr =
      match loop_ctx.k_var with
      | Some kv ->
          Tirix.Arith (Tirix.Arith.Mod, Tirix.Var kv,
            Tirix.Const (Tirix.S32, 2l))
      | None -> Tirix.Const (Tirix.S32, 0l)

    in

    (* Inside a warp group the producer must NOT wait for its own TMA or
       __syncthreads(): the other warps never reach it. The consumer waits on
       the full barrier before its MMA (see the Mma case). *)
    let wait_stmts = match mbar_var with
      | Some mbar when not loop_ctx.in_warp_group ->
        [ Tirix.SOp (Tirix.Barrier (Tirix.MbarWaitParity {
            mbar; phase = phase_expr }))
        ; Tirix.SOp (Tirix.Barrier Tirix.CtaSync) ]
      | _ -> []
    in

    (* Warp-specialised producer: before refilling stage k % depth, wait until
       the consumers released it. Round r = g / depth (g = global k, so this
       stays valid across tile loops) needs the (r-1)th release, i.e. parity
       (r + 1) % 2. Round 0 passes immediately on a fresh barrier. *)
    let empty_wait = match ctx.empty_mbar, loop_ctx.k_var with
      | Some empty, Some kv when is_tma && loop_ctx.in_warp_group && not skip_empty_wait ->
        let phase =
          Tirix.Arith (Tirix.Arith.Mod,
            Tirix.Arith (Tirix.Arith.Add,
              Tirix.Arith (Tirix.Arith.Div, global_k loop_ctx kv,
                Tirix.Const (Tirix.S32, Int32.of_int_exn ctx.pipeline_depth)),
              Tirix.Const (Tirix.S32, 1l)),
            Tirix.Const (Tirix.S32, 2l))
        in
        [ Tirix.SOp (Tirix.Barrier (Tirix.MbarWaitParity { mbar = empty; phase })) ]
      | _ -> []
    in

    (match empty_wait @ arrive_stmts @ [copy_stmt] @ wait_stmts with
     | [s] -> s
     | many -> Tirix.SSeq many)

  | Kernel_ast.Store (src_expr, dst_expr, mask_opt) ->
    let src  = lookup_tensor ctx src_expr in
    let dst  = lookup_tensor ctx dst_expr in
    let kind = infer_copy_kind ctx src dst in
    let pred = Option.map mask_opt ~f:lower_mask in
    let tma_coord_scaled var_opt tensor_expr dim_f =
      match var_opt, tensor_expr with
      | Some v, Kernel_ast.Smem (_, _, shape) ->
        let strip = dim_f shape in
        if strip > 0 then
          Some (Tirix.Arith (Tirix.Arith.Mul, Tirix.Var v,
            Tirix.Const (Tirix.S32, Int32.of_int_exn strip)))
        else
          Some (Tirix.Var v)
      | Some v, _ -> Some (Tirix.Var v)
      | None, _ -> None
    in

    let copy_stmt = Tirix.SOp (Tirix.Copy {
        copy_kind = kind
      ; src_tensor = src
      ; dst_tensor = dst
      ; pred_expr = pred
      ; mbar_var = None
      ; tma_coord_k = tma_coord_scaled loop_ctx.k_var src_expr (fun s -> s.Kernel_ast.k)
      ; tma_coord_m = (match tma_coord_scaled loop_ctx.m_var src_expr (fun s -> s.Kernel_ast.m) with
          | Some _ as c -> c
          | None ->
            (* Epilogue acc stores have no m loop: the row is the CTA's half
               of the pair tile selected by the persistent pair-tile index
               (static fallback: the CTA's tile row, offset by tile *
               gridDim.x like the A loads above). *)
            (match pair_decomp_of ctx loop_ctx.tile_var with
             | Some pd when Poly.(ctx.arch = Kernel_ast.SM100a) && ctx.cta_count > 1 ->
               Some (pair_coord ~tile_dim:pd.pd_tile_m ~extent:(2 * ctx.bm))
             | _ ->
               (match loop_ctx.tile_var with
                | Some tv ->
                  Some (Tirix.Arith (Tirix.Arith.Mul,
                    Tirix.Arith (Tirix.Arith.Add,
                      Tirix.Builtin (Tirix.BlockIdx Tirix.X),
                      Tirix.Arith (Tirix.Arith.Mul, Tirix.Var tv,
                        Tirix.Builtin (Tirix.GridDim Tirix.X))),
                    Tirix.Const (Tirix.S32, Int32.of_int_exn ctx.bm)))
                | None -> None)))
      ; tma_coord_n = (match pair_decomp_of ctx loop_ctx.tile_var with
          | Some pd when Poly.(ctx.arch = Kernel_ast.SM100a) && ctx.cta_count > 1 ->
            (* both CTAs write the pair's full N block *)
            Some (pa_mul pd.pd_tile_n (pa_c ctx.bn))
          | _ -> tma_coord_scaled loop_ctx.n_var src_expr (fun s -> s.Kernel_ast.n))
      ; stage_var = loop_ctx.stage_var
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
    let mma_stmt = Tirix.SOp (Tirix.Mma {
      mma_kind    = kind
    ; mma_atom    = Tirix.default_atom_for_tile ~bn:ctx.bn kind
    ; tensor_a    = lookup_tensor ctx a_expr
    ; tensor_b    = lookup_tensor ctx b_expr
    ; tensor_c    = lookup_tensor ctx c_expr
    ; smem_desc_a = None
    ; smem_desc_b = None
    ; accum_flag  = true
    }) in
    (* Warp-specialised consumer: wait until the producer's TMA for this stage
       has landed. Phase flips every pipeline_depth iterations of the global k
       index (tile loops included), matching the producer. The emitter
       addresses full_mbar[stage]. *)
    (match ctx.full_mbar, loop_ctx.k_var, loop_ctx.stage_var with
     | Some mbar, Some kv, Some _ when loop_ctx.in_warp_group ->
       let phase =
         Tirix.Arith (Tirix.Arith.Mod,
           Tirix.Arith (Tirix.Arith.Div, global_k loop_ctx kv,
             Tirix.Const (Tirix.S32, Int32.of_int_exn ctx.pipeline_depth)),
           Tirix.Const (Tirix.S32, 2l))
       in
        (* tcgen05.commit arrives when all prior MMAs of this thread completed,
           so the stage is only released after the hardware is done reading it.
           For cta_group::2 only the leader CTA issues the MMAs and commits, so
           the release must be multicast: each CTA's producer waits on its own
           empty_mbar and the commit delivers one arrival to each. *)
        let release = match ctx.empty_mbar with
          | Some empty ->
            let pair_release =
              Poly.(ctx.arch = Kernel_ast.SM100a) && ctx.cta_count > 1 in
            let mask =
              if pair_release || (ctx.multicast && ctx.cta_count > 1)
              then Some (Int.shift_left 1 ctx.cta_count - 1)
              else None
            in
            [ Tirix.SOp (Tirix.TmemCommit { mbar_var = empty; cta_mask = mask }) ]
          | None -> []
       in
       Tirix.SSeq
         ([ Tirix.SOp (Tirix.Barrier (Tirix.MbarWaitParity { mbar; phase }))
          ; mma_stmt ] @ release)
     | _ -> mma_stmt)

  | Kernel_ast.For (var_name, start_val, stop_val, body_stmts) ->
    let loop_var = {
      Tirix.var_name  = var_name
    ; var_id          = fresh_id ctx
    ; var_type        = Tirix.Scalar Tirix.S32
    ; var_mutable     = true
    } in
    let is_k_loop = match var_name with
      | "k" | "k_loop" | "k_tile" -> true
      | _ -> false
    in
    (* Pipeline stage counter for K loops of warp-specialised TMA kernels:
         int32_t stage = 0;  for (k ...) { stage = k % depth; ... } *)
    let stage_opt =
      if is_k_loop && loop_ctx.in_warp_group
         && Option.is_some ctx.full_mbar && ctx.pipeline_depth > 1
      then
        Some { Tirix.var_name = "stage"
             ; var_id = fresh_id ctx
             ; var_type = Tirix.Scalar Tirix.S32
             ; var_mutable = true }
      else None
    in
    let new_loop_ctx = match var_name with
      | "k" | "k_loop" | "k_tile"  ->
        { loop_ctx with
          k_var = Some loop_var
        ; k_stop = Some stop_val
        ; stage_var = (match stage_opt with
                       | Some _ -> stage_opt
                       | None -> loop_ctx.stage_var) }
      | "m" | "row"   | "block_m"  -> { loop_ctx with m_var = Some loop_var }
      | "n" | "col"   | "block_n"  -> { loop_ctx with n_var = Some loop_var }
      | "stage" | "s"              -> { loop_ctx with stage_var = Some loop_var }
      | "tile"                      -> { loop_ctx with tile_var = Some loop_var }
      | _                          -> loop_ctx
    in
    let stage_assign = match stage_opt with
      | None -> []
      | Some sv ->
        (* Stage rotation follows the global k index so it stays continuous
           across tile loops; otherwise the stage/parity bookkeeping desyncs
           at tile boundaries when k_tiles is not a multiple of the depth. *)
        [ Tirix.SAssign (sv,
            Tirix.Expr (Tirix.Arith (Tirix.Arith.Mod, global_k new_loop_ctx loop_var,
              Tirix.Const (Tirix.S32, Int32.of_int_exn ctx.pipeline_depth)))) ]
    in
    let lowered_body = convert_stmts_with_skip ctx new_loop_ctx body_stmts in
    (* Persistent tile schedule (2-CTA Blackwell): the tile loop runs
       ceil(total_pair_tiles / num_pairs) iterations over the flattened
       pair-tile index, and iterations beyond the total are skipped. All
       warp groups build the same schedule (the epilogue's loop reuses it
       via find_tile_schedule in the emitter). *)
    let persistent =
      if String.equal var_name "tile"
         && Poly.(ctx.arch = Kernel_ast.SM100a) && ctx.cta_count > 1
      then pair_decomp_of ctx (Some loop_var) else None in
    let stop =
      match persistent with
      | Some pd -> pd.pd_bound
      | None -> Tirix.Const (Tirix.S32, Int32.of_int_exn stop_val) in
    let body_stmts' =
      match persistent with
      | Some pd ->
        [ Tirix.SIf (Tirix.Cmp (Tirix.Cmp.Lt, pd.pd_idx, pd.pd_total),
                     stage_assign @ lowered_body, []) ]
      | None -> stage_assign @ lowered_body
    in
    let loop = Tirix.SFor {
      var = loop_var
    ; start = Tirix.Const (Tirix.S32, Int32.of_int_exn start_val)
    ; stop
    ; step = Tirix.Const (Tirix.S32, 1l)
    ; dir = Tirix.Upto
    ; unroll = false
    ; body = body_stmts'
    } in
    (match stage_opt with
     | None -> loop
     | Some sv ->
       Tirix.SSeq
         [ Tirix.SLetMut (sv, Tirix.Expr (Tirix.Const (Tirix.S32, 0l)))
         ; loop ])

  | Kernel_ast.Pipeline (desc, stmts) ->
    Tirix.SPipeline {
      stages = desc.Kernel_ast.stages
    ; prologue = []
    ; mainloop = convert_stmts_with_skip ctx loop_ctx stmts
    ; epilogue = []
    }

  | Kernel_ast.Barrier b ->
    Tirix.SOp (Tirix.Barrier (lower_barrier b))

  | Kernel_ast.If (pred, thn, els) ->
    (match group_role_of_pred ctx pred with
     | Some role ->
       (* warp_dispatch branch: becomes a warp group of that role *)
       Tirix.SSeq
         (Tirix.SWarpGroup
            (role, convert_stmts_with_skip ctx { loop_ctx with in_warp_group = true } thn)
          :: List.map els ~f:(convert_stmt ctx loop_ctx))
     | None ->
       Tirix.SIf (
         lower_pred pred,
         convert_stmts_with_skip ctx loop_ctx thn,
         convert_stmts_with_skip ctx loop_ctx els))

  | Kernel_ast.Seq stmts ->
    Tirix.SSeq (convert_stmts_with_skip ctx loop_ctx stmts)

and convert_stmts_with_skip (ctx : ctx) (loop_ctx : loop_ctx) (stmts : Kernel_ast.stmt list) : Tirix.stmt list =
  let rec loop prev_was_tma = function
    | [] -> []
    | s :: rest ->
      let is_tma = match s with
        | Kernel_ast.Load (Kernel_ast.Arg (_, _, Kernel_ast.Global), Kernel_ast.Smem _, _) -> true
        | _ -> false
      in
      let lowered = convert_stmt ctx loop_ctx ~skip_empty_wait:prev_was_tma s in
      lowered :: loop is_tma rest
  in
  loop false stmts


let lower (k : Kernel_ast.kernel) : Tirix.tirix =
  let tensors = collect_tensors k.Kernel_ast.body [] in
  let needs_tma = has_tma_load k.Kernel_ast.arch k.Kernel_ast.body in
  let pipeline_depth = k.Kernel_ast.stages in

  let (full_mbar_var, empty_mbar_var, mbar_tensors) =
    if needs_tma then
      let need_empty = pipeline_depth > 1 in
      ( Some (make_mbar_var "full_mbar")
      , (if need_empty then Some (make_mbar_var "empty_mbar") else None)
      , if need_empty then make_mbar_tensors pipeline_depth
        else List.filter (make_mbar_tensors pipeline_depth)
               ~f:(fun (n, _) -> String.equal n "full_mbar") )
    else
      (None, None, [])
  in

  (* Warp roles / cluster are needed while lowering the body (warp groups). *)
  let cluster = match k.Kernel_ast.arch with
    | Kernel_ast.SM80 ->
      Cluster.make { Cluster.x=1; y=1; z=1 } 4
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Epilogue); (3, Cluster.Epilogue) ]
    | Kernel_ast.SM90a ->
      Cluster.make { Cluster.x=1; y=1; z=1 } 8
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Consumer); (3, Cluster.Consumer)
        ; (4, Cluster.Consumer); (5, Cluster.Epilogue)
        ; (6, Cluster.Epilogue); (7, Cluster.Epilogue) ]
    | Kernel_ast.SM100a ->
      (* warps 2..5: warp_id % 4 covers all four TMEM lane quadrants *)
      Cluster.make { Cluster.x=2; y=1; z=1 } 6
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Epilogue); (3, Cluster.Epilogue)
        ; (4, Cluster.Epilogue); (5, Cluster.Epilogue) ]
  in

  (* tcgen05.mma cta_group::2 covers the CTA-PAIR tile: M and N are each
     split across the two CTAs (A along M, B along N). tile.n is the
     per-CTA B shard, so the pair N — used for the MMA atom, the
     instruction descriptor, the TMEM allocation and the epilogue column
     base — is twice tile.n. bm stays per-CTA because each CTA owns its
     A rows directly (blockIdx.x addresses CTA granularity). *)
  let pair_n = match k.Kernel_ast.arch with
    | Kernel_ast.SM100a -> 2 * k.Kernel_ast.tile.Kernel_ast.n
    | _ -> k.Kernel_ast.tile.Kernel_ast.n in

  let ctx = {
    arch = k.Kernel_ast.arch
  ; tensors = tensors @ mbar_tensors
  ; var_counter = ref 0
  ; full_mbar  = full_mbar_var
  ; empty_mbar = empty_mbar_var
  ; pipeline_depth
  ; bm = k.Kernel_ast.tile.Kernel_ast.m
  ; bn = pair_n
  ; bk = k.Kernel_ast.tile.Kernel_ast.k
  ; elem_bytes = elem_byte_width k.Kernel_ast.elem
  ; roles = cluster.Cluster.warp_roles
  ; cta_count =
      cluster.Cluster.dims.Cluster.x * cluster.Cluster.dims.Cluster.y
      * cluster.Cluster.dims.Cluster.z
  ;   multicast = k.Kernel_ast.multicast
  ; has_m = List.exists k.Kernel_ast.args
        ~f:(fun (n, _, _, _) -> String.equal n "M")
  ; has_n = List.exists k.Kernel_ast.args
        ~f:(fun (n, _, _, _) -> String.equal n "N")
  } in

  (* __syncthreads() after mbarrier init loop (warp 0 lane 0) so that
     all warps see initialized mbar state before issuing cp.async.bulk. *)
  let sync_after_init =
    if needs_tma then [ Tirix.SOp (Tirix.Barrier Tirix.CtaSync) ]
    else []
  in

  let body_stmt = convert_stmt ctx empty_loop_ctx k.Kernel_ast.body in

  { Tirix.name           = k.Kernel_ast.name
  ; family               = lower_family k.Kernel_ast.arch
  ; params               = lower_params ctx k
  ; tensors              = ctx.tensors
  ; smem_bytes           = 0
  ; pipeline_depth
  ; bm                   = k.Kernel_ast.tile.Kernel_ast.m
  ; bn                   = pair_n
  ; bk                   = k.Kernel_ast.tile.Kernel_ast.k
  ; cluster
  ; body                 = sync_after_init @ [ body_stmt ]
  ; helpers              = []
  }
