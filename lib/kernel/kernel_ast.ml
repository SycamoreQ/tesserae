open Base

type arch =
  | SM80
  | SM90a
  | SM100a

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

type arg_dir = In | Out

type tensor_arg = string * elem * space * arg_dir

type tensor_expr =
  | Arg of string * elem * space
  | Tile of tensor_expr * tile_shape
  | LocalTile of tensor_expr * tile_shape
  | Smem of string * elem * tile_shape

and tile_shape = {
  m : int;
  n : int;
  k : int;
}

type pipeline_desc = {
  stages  : int;
  k_iters : string;
}

type mask = {
  coord_var : string;
  bounds    : int list;
}

type barrier_kind =
  | MbarInit of string * int      (* mbarrier name, transaction count *)
  | MbarWaitParity of string * int (* mbarrier name, parity (0 or 1) *)
  | MbarFull of string
  | MbarEmpty of string
  | ClusterSync
  | ThreadSync

type pred_expr =
  | WarpIs of int
  | WarpIn of int list
  | InBounds of string * int list
  | Mbarrier of string

type stmt =
  | Load     of tensor_expr * tensor_expr * mask option
  | Store    of tensor_expr * tensor_expr * mask option
  | Mma      of tensor_expr * tensor_expr * tensor_expr
  | Pipeline of pipeline_desc * stmt list
  | Barrier  of barrier_kind
  | For      of string * int * int * stmt list
  | If       of pred_expr * stmt list * stmt list
  | Seq      of stmt list

type kernel = {
  name : string;
  arch : arch;
  elem : elem;
  tile : tile_shape;
  stages : int;
  args : tensor_arg list;
  body: stmt;
}

let make ~name ~arch ~elem ~tile ~stages ~args ~body =
  { name; arch; elem; tile; stages; args; body }

let tensor_arg name e sp = Arg (name, e, sp)
let arg name e sp = Arg (name, e, sp)

let in_arg  name e = (name, e, Global, In)
let out_arg name e = (name, e, Global, Out)

let smem name e m k = Smem (name, e, { m; n = 0; k })

let load ~src ~dst ?mask () = Load (src, dst, mask)
let store ~src ~dst ?mask () = Store (src, dst, mask)
let mma a b c = Mma (a, b, c)

let pipeline ~stages ~k body = Pipeline ({ stages; k_iters = k }, body)
let syncthreads () = Barrier ThreadSync

let warp_dispatch cases =
  Seq (List.map cases ~f:(fun (pred, body) -> If (pred, body, [])))

let arch_str = function
  | SM80   -> "SM80"
  | SM90a  -> "SM90a"
  | SM100a -> "SM100a"

let elem_str = function
  | F16  -> "f16"
  | BF16 -> "bf16"
  | F32  -> "f32"
  | S8   -> "s8"
  | S32  -> "s32"

let rec pp_tensor fmt = function
  | Arg (name, e, _ ) ->
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
  | Barrier MbarFull v ->
    Stdlib.Format.fprintf fmt "mbar_full(%s)" v
  | Barrier MbarEmpty v ->
    Stdlib.Format.fprintf fmt "mbar_empty(%s)" v
  | Barrier (MbarInit (v, cnt)) ->
    Stdlib.Format.fprintf fmt "mbar_init(%s, %d)" v cnt
  | Barrier (MbarWaitParity (v, phase)) ->
    Stdlib.Format.fprintf fmt "mbar_wait_parity(%s, %d)" v phase
  | Barrier ThreadSync  -> Stdlib.Format.fprintf fmt "syncthreads()"
  | Barrier ClusterSync   -> Stdlib.Format.fprintf fmt "cluster_sync()"
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
  | If (Mbarrier name, t, _) ->
      Stdlib.Format.fprintf fmt "if mbarrier(%s) { %d stmts }" name (List.length t)

let pp fmt k =
  Stdlib.Format.fprintf fmt "kernel %s arch=%s tile=%dx%dx%d stages=%d\n"
    k.name (arch_str k.arch) k.tile.m k.tile.n k.tile.k k.stages;
  pp_stmt fmt k.body
