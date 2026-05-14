open Base

type arch =
  | SM80
  | SM90
  | SM100

type elem =
  | F16
  | BF16
  | F32
  | S8
  | S32

type space =
  | Global
  | Shared
  | Register
  | TensorMem

type tensor_expr =
  | Arg       of string * elem * space
  | Tile      of tensor_expr * tile_shape
  | LocalTile of tensor_expr * tile_shape
  | Smem      of string * elem * tile_shape

and tile_shape = {
  m : int;
  n : int;
  k : int;
}

type stmt =
  | Load     of tensor_expr * tensor_expr * mask option
  | Store    of tensor_expr * tensor_expr * mask option
  | Mma      of tensor_expr * tensor_expr * tensor_expr
  | Pipeline of pipeline_desc * stmt list
  | Barrier  of barrier_kind
  | For      of string * int * int * stmt list
  | If       of pred_expr * stmt list * stmt list
  | Seq      of stmt list

and pipeline_desc = {
  stages  : int;
  k_iters : string;
}

and mask = {
  coord_var : string;
  bounds    : int list;
}

and barrier_kind =
  | MbarFull   of string
  | MbarEmpty  of string
  | ClusterSync
  | ThreadSync

and pred_expr =
  | WarpIs   of int
  | WarpIn   of int list
  | InBounds of string * int list

type kernel = {
  name   : string;
  arch   : arch;
  elem   : elem;
  tile   : tile_shape;
  stages : int;
  args   : (string * elem * space) list;
  body   : stmt;
}

let make ~name ~arch ~elem ~tile ~stages ~args ~body =
  { name; arch; elem; tile; stages; args; body }

let arg name e s = Arg (name, e, s)

let smem name e m k = Smem (name, e, { m; n = 0; k })

let load ~src ~dst ?mask () = Load (src, dst, mask)

let store ~src ~dst ?mask () = Store (src, dst, mask)

let mma a b c = Mma (a, b, c)

let pipeline ~stages ~k body = Pipeline ({ stages; k_iters = k }, body)

let syncthreads () = Barrier ThreadSync

let warp_dispatch cases =
  Seq (List.map cases ~f:(fun (pred, body) -> If (pred, body, [])))

let arch_str = function
  | SM80  -> "SM80"
  | SM90  -> "SM90"
  | SM100 -> "SM100"

let elem_str = function
  | F16  -> "f16"
  | BF16 -> "bf16"
  | F32  -> "f32"
  | S8   -> "s8"
  | S32  -> "s32"

let rec pp_tensor fmt = function
  | Arg (name, e, _) ->
    Stdlib.Format.fprintf fmt "%s:%s" name (elem_str e)
  | Tile (t, sh) ->
    Stdlib.Format.fprintf fmt "tile(%a,%dx%dx%d)" pp_tensor t sh.m sh.n sh.k
  | LocalTile (t, sh) ->
    Stdlib.Format.fprintf fmt "local_tile(%a,%dx%dx%d)" pp_tensor t sh.m sh.n sh.k
  | Smem (name, e, _) ->
    Stdlib.Format.fprintf fmt "smem(%s:%s)" name (elem_str e)

let rec pp_stmt fmt = function
  | Load (src, dst, _) ->
    Stdlib.Format.fprintf fmt "load %a -> %a" pp_tensor src pp_tensor dst
  | Store (src, dst, _) ->
    Stdlib.Format.fprintf fmt "store %a -> %a" pp_tensor src pp_tensor dst
  | Mma (a, b, c) ->
    Stdlib.Format.fprintf fmt "mma(%a, %a) -> %a"
      pp_tensor a pp_tensor b pp_tensor c
  | Pipeline (pd, body) ->
    Stdlib.Format.fprintf fmt "pipeline(stages=%d, k=%s) {\n"
      pd.stages pd.k_iters;
    List.iter body ~f:(fun s ->
      Stdlib.Format.fprintf fmt "  %a\n" pp_stmt s);
    Stdlib.Format.fprintf fmt "}"
  | Barrier ThreadSync    -> Stdlib.Format.fprintf fmt "syncthreads()"
  | Barrier ClusterSync   -> Stdlib.Format.fprintf fmt "cluster_sync()"
  | Barrier (MbarFull v)  -> Stdlib.Format.fprintf fmt "mbar_wait_full(%s)" v
  | Barrier (MbarEmpty v) -> Stdlib.Format.fprintf fmt "mbar_wait_empty(%s)" v
  | For (v, lo, hi, body) ->
    Stdlib.Format.fprintf fmt "for %s = %d to %d { %d stmts }"
      v lo hi (List.length body)
  | If (WarpIs n, t, _) ->
    Stdlib.Format.fprintf fmt "if warp_id==%d { %d stmts }" n (List.length t)
  | If (WarpIn ns, t, _) ->
    Stdlib.Format.fprintf fmt "if warp_id in [%s] { %d stmts }"
      (String.concat ~sep:"," (List.map ns ~f:Int.to_string))
      (List.length t)
  | If (InBounds (v, _), t, _) ->
    Stdlib.Format.fprintf fmt "if in_bounds(%s) { %d stmts }" v (List.length t)
  | Seq stmts ->
    List.iter stmts ~f:(fun s ->
      pp_stmt fmt s;
      Stdlib.Format.fprintf fmt "\n")

let pp fmt k =
  Stdlib.Format.fprintf fmt "kernel %s arch=%s tile=%dx%dx%d stages=%d\n"
    k.name (arch_str k.arch) k.tile.m k.tile.n k.tile.k k.stages;
  pp_stmt fmt k.body
