open Base
open Tesserae_kernel
open Tesserae_pipeline
open Tesserae_core


type ctx = {
  arch : Kernel_ast.arch;
  tensors     : (string * Tirix.packed_tensor) list;
  var_counter : int ref;
}

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



let make_tensor name (elem : Kernel_ast.elem) (space : Kernel_ast.space)
    layout swizzle : Tirix.packed_tensor =
  let Elem e = lower_elem elem in
  let Space m = lower_space space in

  Tirix.Tensor {
    tensor_name = name
  ; tensor_id = Tirix.Type_id.create ()
  ; tensor_elem_type = e
  ; tensor_memspace = m
  ; tensor_layout = layout
  ; tensor_swizzle = swizzle
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
    make_tensor name elem Kernel_ast.Shared layout sw
  | Kernel_ast.Tile (inner, _)      -> lookup_tensor ctx inner
  | Kernel_ast.LocalTile (inner, _) -> lookup_tensor ctx inner



let infer_copy_kind (ctx : ctx)
    (src : Tirix.packed_tensor) (dst : Tirix.packed_tensor) : Tirix.copy_kind =
  let (Tirix.Tensor s) = src in
  let (Tirix.Tensor d) = dst in
  match Memspace.name s.tensor_memspace,
        Memspace.name d.tensor_memspace with
  | "global", "shared" ->
    (match ctx.arch with
     | Kernel_ast.SM80  -> Tirix.CpAsync
     | Kernel_ast.SM90  -> Tirix.TmaLoad
     | Kernel_ast.SM100 -> Tirix.TmaMulticast)
  | "shared", "register" -> Tirix.SmemToReg
  | "register", "shared" -> Tirix.RegToSmem
  | "register", "global" -> Tirix.RegToSmem
  | _ -> Tirix.CpAsync


let infer_mma_kind (ctx : ctx) : Tirix.mma_kind =
  match ctx.arch with
  | Kernel_ast.SM80 -> Tirix.Sm80Mma
  | Kernel_ast.SM90 -> Tirix.Sm90Wgmma
  | Kernel_ast.SM100 -> Tirix.Sm100Tcgen05



let lower_mask (m : Kernel_ast.mask) : bool Tirix.expr =
  let v = {
    Tirix.var_name = m.Kernel_ast.coord_var
  ; var_id = 0
  ; var_type = Tirix.Scalar Tirix.S32
  ; var_mutable = false
  } in
  List.foldi m.Kernel_ast.bounds ~init:(Tirix.Const (Tirix.Bool, true))
    ~f:(fun _i acc bound ->
      Logic (And, acc,
        Cmp (Lt,
          Tirix.Var v,
          Tirix.Const (S32, Int32.of_int_exn bound))))


let lower_barrier = function
  | Kernel_ast.ThreadSync -> Tirix.CtaSync
  | Kernel_ast.ClusterSync -> Tirix.ClusterArrive
  | Kernel_ast.MbarFull  var ->
    let v = { Tirix.var_name=var; var_id=0
            ; var_type=Tirix.Scalar Tirix.U64; var_mutable=false } in
    Tirix.MbarWaitParity { mbar=v; phase=Tirix.Const (Tirix.S32, 0l) }
  | Kernel_ast.MbarEmpty var ->
    let v = { Tirix.var_name=var; var_id=0
            ; var_type=Tirix.Scalar Tirix.U64; var_mutable=false } in
    Tirix.MbarArrive { mbar=v }


let lower_pred (p : Kernel_ast.pred_expr) : bool Tirix.expr =
  match p with
  | Kernel_ast.WarpIs n ->
    Cmp (Eq,
      Tirix.Builtin Tirix.WarpId,
      Const (Tirix.S32, Int32.of_int_exn n))
  | Kernel_ast.WarpIn ns ->
    List.fold ns ~init:(Tirix.Const (Tirix.Bool, false))
      ~f:(fun acc n ->
        Logic (Or, acc,
          Cmp (Eq,
            Tirix.Builtin Tirix.WarpId,
            Tirix.Const (Tirix.S32, Int32.of_int_exn n))))
  | Kernel_ast.InBounds (var, bounds) ->
    lower_mask { Kernel_ast.coord_var = var; bounds }


let lower_params (ctx : ctx) (k : Kernel_ast.kernel) : Tirix.param list =
  let flat = Layout.make (Modes.Int 1) (Modes.Int 1) in
  let sw   = Swizzle.make 0 0 0 in
  List.map k.Kernel_ast.args ~f:(fun (name, elem, space) ->
    let is_tma = match ctx.arch, space with
      | (Kernel_ast.SM90 | Kernel_ast.SM100), Kernel_ast.Global -> true
      | _ -> false
    in
    { Tirix.param_name = name
    ; param_tensor = make_tensor name elem space flat sw
    ; param_is_tma = is_tma
    })

let lower_family = function
  | Kernel_ast.SM80 -> Kernel_desc.Ampere
  | Kernel_ast.SM90 -> Kernel_desc.Hopper
  | Kernel_ast.SM100 -> Kernel_desc.Blackwell


let rec convert_stmt (ctx : ctx) (stmt : Kernel_ast.stmt) : Tirix.stmt =
  match stmt with
  | Kernel_ast.Load (src_expr, dst_expr, mask_opt) ->
    let src  = lookup_tensor ctx src_expr in
    let dst  = lookup_tensor ctx dst_expr in
    let kind = infer_copy_kind ctx src dst in
    let pred = Option.map mask_opt ~f:lower_mask in
    Tirix.SOp (Tirix.Copy {
      copy_kind  = kind
    ; src_tensor = src
    ; dst_tensor = dst
    ; pred_expr = pred
    ; mbar_var = None
    })
  | Kernel_ast.Store (src_expr, dst_expr, mask_opt) ->
    let src = lookup_tensor ctx src_expr in
    let dst = lookup_tensor ctx dst_expr in
    let kind = infer_copy_kind ctx src dst in
    let pred = Option.map mask_opt ~f:lower_mask in
    Tirix.SOp (Tirix.Copy {
      copy_kind = kind
    ; src_tensor = src
    ; dst_tensor = dst
    ; pred_expr = pred
    ; mbar_var = None
    })
  | Kernel_ast.Mma (a_expr, b_expr, c_expr) ->
    Tirix.SOp (Tirix.Mma {
      mma_kind = infer_mma_kind ctx
    ; tensor_a = lookup_tensor ctx a_expr
    ; tensor_b = lookup_tensor ctx b_expr
    ; tensor_c=  lookup_tensor ctx c_expr
    ; smem_desc_a = None
    ; smem_desc_b = None
    ; accum_flag  = true
    })
  | Kernel_ast.For (var_name, start_val, stop_val, body_stmts) ->
    let loop_var = {
      Tirix.var_name = var_name
    ; var_id = fresh_id ctx
    ; var_type = Tirix.Scalar Tirix.S32
    ; var_mutable = true
    } in
    Tirix.SFor {
      var = loop_var
    ; start = Tirix.Const (Tirix.S32, Int32.of_int_exn start_val)
    ; stop = Tirix.Const (Tirix.S32, Int32.of_int_exn stop_val)
    ; step = Tirix.Const (Tirix.S32, 1l)
    ; dir = Tirix.Upto
    ; unroll = false
    ; body = List.map body_stmts ~f:(convert_stmt ctx)
    }
  | Kernel_ast.Pipeline (desc, stmts) ->
    Tirix.SPipeline {
      stages = desc.Kernel_ast.stages
    ; prologue = []
    ; mainloop = List.map stmts ~f:(convert_stmt ctx)
    ; epilogue = []
    }
  | Kernel_ast.Barrier b ->
    Tirix.SOp (Tirix.Barrier (lower_barrier b))
  | Kernel_ast.If (pred, thn, els) ->
    Tirix.SIf (
      lower_pred pred,
      List.map thn ~f:(convert_stmt ctx),
      List.map els ~f:(convert_stmt ctx))
  | Kernel_ast.Seq stmts ->
    Tirix.SSeq (List.map stmts ~f:(convert_stmt ctx))


let lower (k : Kernel_ast.kernel) : Tirix.tirix =
  let ctx = {
    arch = k.Kernel_ast.arch
  ; tensors = []
  ; var_counter = ref 0
  } in
  let body = convert_stmt ctx k.Kernel_ast.body in
  let cluster = match k.Kernel_ast.arch with
    | Kernel_ast.SM80 ->
      Cluster.make { Cluster.x=1; y=1; z=1 } 4
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Epilogue); (3, Cluster.Epilogue) ]
    | Kernel_ast.SM90 ->
      Cluster.make { Cluster.x=2; y=1; z=1 } 8
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Consumer); (3, Cluster.Consumer)
        ; (4, Cluster.Consumer); (5, Cluster.Epilogue)
        ; (6, Cluster.Epilogue); (7, Cluster.Epilogue) ]
    | Kernel_ast.SM100 ->
      Cluster.make { Cluster.x=2; y=1; z=1 } 6
        [ (0, Cluster.Producer); (1, Cluster.Consumer)
        ; (2, Cluster.Epilogue); (3, Cluster.Epilogue)
        ; (4, Cluster.Epilogue); (5, Cluster.Scheduler) ]
  in
  { Tirix.name = k.Kernel_ast.name
  ; family = lower_family k.Kernel_ast.arch
  ; params = lower_params ctx k
  ; tensors = []
  ; smem_bytes =  0
  ; pipeline_depth = k.Kernel_ast.stages
  ; bm = k.Kernel_ast.tile.Kernel_ast.m
  ; bn = k.Kernel_ast.tile.Kernel_ast.n
  ; bk = k.Kernel_ast.tile.Kernel_ast.k
  ; cluster
  ; body = [ body ]
  ; helpers = []
  }
