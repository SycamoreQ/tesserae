open Base
open Tesserae_kernel
open Tesserae_pipeline


type ctx = {
  arch : Kernel_ast.arch;
  tensors     : (string * Tirix.packed_tensor) list;
  var_counter : int ref;
}

let fresh_id (ctx : ctx) : int =
  let id = !(ctx.var_counter) in
  ctx.var_counter := id + 1;
  id

let lookup_tensor (ctx : ctx) (expr : Kernel_ast.tensor_expr)
  : Tirix.packed_tensor =
  match expr with
  | Kernel_ast.Arg (name, elem, space) ->
    let elem_type = Lower.elem_to_elemtype_t elem in
    let memspace  = lower_space space in
    (match List.find ctx.tensors ~f:(fun (n,_) -> String.equal n name) with
     | Some (_, t) -> t
     | None ->
       Tir.Tensor {
         tensor_name = name
       ; tensor_id = Tir.Type_id.create ()
       ; tensor_elem_type = elem_type
       ; tensor_memspace = memspace
       ; tensor_layout = Layout.make (Modes.Int 1) (Modes.Int 1)
       ; tensor_swizzle = Swizzle.make 0 0 0
       })
  | Kernel_ast.Smem (name, elem, shape) ->
    let elem_type = Lower.elem_to_elemtype_t elem in
    let layout = Layout.make
      (Modes.Tuple [Modes.Int shape.Kernel_ast.m; Modes.Int shape.Kernel_ast.k])
      (Modes.Tuple [Modes.Int 1; Modes.Int shape.Kernel_ast.m])
    in
    Tir.Tensor {
      tensor_name = name
    ; tensor_id = Tir.Type_id.create ()
    ; tensor_elem_type = elem_type
    ; tensor_memspace = Memspace.Shared
    ; tensor_layout = layout
    ; tensor_swizzle = Swizzle.make 0 0 0
    }
  | Kernel_ast.Tile (inner, _) -> lookup_tensor ctx inner
  | Kernel_ast.LocalTile (inner, _) -> lookup_tensor ctx inner


let infer_copy_kind (ctx : ctx)
    (src : Tirix.packed_tensor) (dst : Tirix.packed_tensor) : Tirix.copy_kind =
  let (Tirix.Tensor s) = src in
  let (Tirix.Tensor d) = dst in
  match Memspace.name s.tensor_memspace,
        Memspace.name d.tensor_memspace with
  | "global", "shared" ->
    (match ctx.arch with
     | Kernel_ast.SM80  -> Tir.CpAsync
     | Kernel_ast.SM90  -> Tir.TmaLoad
     | Kernel_ast.SM100 -> Tir.TmaMulticast)
  | "shared", "register" -> Tir.SmemToReg
  | "register", "shared" -> Tir.RegToSmem
  | "register", "global" -> Tir.RegToSmem
  | _ -> Tir.CpAsync

let infer_mma_kind (ctx : ctx) : Tirix.mma_kind =
  match ctx.arch with
  | Kernel_ast.SM80  -> Tirix.Sm80Mma
  | Kernel_ast.SM90  -> Tirix.Sm90Wgmma
  | Kernel_ast.SM100 -> Tirix.Sm100Tcgen05


let lower_mask (m : Kernel_ast.mask) : bool Tirix.expr =
  let v = {
    Tirix.var_name = m.Kernel_ast.coord_var
  ; var_id = 0
  ; var_type = Tir.Scalar Tir.S32
  ; var_mutable = false
  } in
  List.foldi m.Kernel_ast.bounds ~init:(Tirix.Const (Tir.Bool, true))
    ~f:(fun i acc bound ->
      Tirix Logical (Tirix.And, acc,
        Tirix Cmp (Tirix.Lt,
          Tirix.Var v,
          Tirix.Const (Tir.S32, Int32.of_int_exn bound))))

let lower_barrier = function
  | Kernel_ast.ThreadSync -> Tirix.CtaSync
  | Kernel_ast.ClusterSync -> Tirix.ClusterArrive
  | Kernel_ast.MbarFull  var ->
    let v = { Tirix.var_name=var; var_id=0
            ; var_type=Tir.Scalar Tir.U64; var_mutable=false } in
    Tirix.MbarWaitParity { mbar=v; phase=Tirix.Const (Tirix.S32, 0l) }
  | Kernel_ast.MbarEmpty var ->
    let v = { Tir.var_name=var; var_id=0
            ; var_type=Tirix.Scalar Tirix.U64; var_mutable=false } in
    Tirix.MbarArrive { mbar=v }

let lower_pipeline = function
  | Pipeline (pipeline_desc, body_stmts) ->
        let stages = pipeline_desc.stages in
        let prologue_tirix =
          List.map ~f:convert_stmt []
        in

        let mainloop_tirix =
          List.map body_stmts ~f:convert_stmt
        in

        let epilogue_tirix =
          List.map ~f:convert_stmt []
        in

        SPipeline {
          stages = stages;
          prologue = prologue_tirix;
          mainloop = mainloop_tirix;
          epilogue = epilogue_tirix;
        }

let lower_pred (p : Kernel_ast.pred_expr) : bool Tirix.expr =
  match p with
  | Kernel_ast.WarpIs n ->
    Tirix.Binop (Tirix.Eq,
      Tirix.Builtin Tirix.WarpId,
      Tirix.Const (Tirix.S32, Int32.of_int_exn n))
  | Kernel_ast.WarpIn ns ->
    List.fold ns ~init:(Tirix.Const (Tirix.Bool, false))
      ~f:(fun acc n ->
        Tirix.Binop (Tirix.Or, acc,
          Tirix.Binop (Tirix.Eq,
            Tirix.Builtin Tirix.WarpId,
            Tirix.Const (Tirix.S32, Int32.of_int_exn n))))
  | Kernel_ast.InBounds (var, bounds) ->
    lower_mask { Kernel_ast.coord_var = var; bounds }


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
  { Tir.name = k.Kernel_ast.name
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
