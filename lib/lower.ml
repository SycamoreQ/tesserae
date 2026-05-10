open Base

module type Elem_witness = sig
  type t
  val witness : t Elemtype.t
end

type error =
  | UnsupportedArch  of string
  | UnsupportedElem  of string
  | IncompatibleTile of string
  | MissingArg of string
  | InvalidPipeline  of string

let pp_error fmt = function
  | UnsupportedArch  s -> Stdlib.Format.fprintf fmt "UnsupportedArch: %s"  s
  | UnsupportedElem  s -> Stdlib.Format.fprintf fmt "UnsupportedElem: %s"  s
  | IncompatibleTile s -> Stdlib.Format.fprintf fmt "IncompatibleTile: %s" s
  | MissingArg s -> Stdlib.Format.fprintf fmt "MissingArg: %s" s
  | InvalidPipeline  s -> Stdlib.Format.fprintf fmt "InvalidPipeline: %s"  s

let arch_to_strategy : Kernel_ast.arch -> Tile_io.strategy = function
  | Kernel_ast.SM80  -> Tile_io.CpAsync
  | Kernel_ast.SM90 -> Tile_io.TmaLoad
  | Kernel_ast.SM100 -> Tile_io.TmaMulticast

let arch_to_accum : Kernel_ast.arch -> Tile_op.accum_loc = function
  | Kernel_ast.SM80 -> Tile_op.Registers
  | Kernel_ast.SM90 -> Tile_op.Registers
  | Kernel_ast.SM100 -> Tile_op.TensorMem

(** Extract problem dims — default to tile*32 if not inferable *)
let infer_m (k : Kernel_ast.kernel) : int =
  ignore k; k.Kernel_ast.tile.Kernel_ast.m * 32

let infer_n (k : Kernel_ast.kernel) : int =
  ignore k; k.Kernel_ast.tile.Kernel_ast.n * 32

let infer_k (k : Kernel_ast.kernel) : int =
  ignore k; k.Kernel_ast.tile.Kernel_ast.k * 32

(** elem_to_elemtype — returns a first-class module carrying the witness *)
let elem_to_elemtype (e : Kernel_ast.elem): (module Elem_witness) =
  match e with
  | Kernel_ast.F16  -> (module struct type t = Elemtype.float16
                                       let witness = Elemtype.Float16 end)
  | Kernel_ast.BF16 -> (module struct type t = Elemtype.bfloat16
                                       let witness = Elemtype.Bfloat16 end)
  | Kernel_ast.F32  -> (module struct type t = Elemtype.float32
                                       let witness = Elemtype.Float32 end)
  | Kernel_ast.S8   -> (module struct type t = Elemtype.int8
                                       let witness = Elemtype.Int8 end)
  | Kernel_ast.S32  -> (module struct type t = Elemtype.int32
                                       let witness = Elemtype.Int32 end)

type packed = Pack : (_, _, _, _, _, _) Kernel_desc.t -> packed

let lower (k : Kernel_ast.kernel) : (packed, error) Result.t =
  let bm = k.Kernel_ast.tile.Kernel_ast.m in
  let bn = k.Kernel_ast.tile.Kernel_ast.n in
  let bk = k.Kernel_ast.tile.Kernel_ast.k in
  let name = k.Kernel_ast.name in
  let m = infer_m k in
  let n = infer_n k in
  let kk = infer_k k in
  match k.Kernel_ast.arch, k.Kernel_ast.elem with
  | Kernel_ast.SM80, Kernel_ast.F16 ->
    Ok (Pack (Kernel_desc.make_ampere ~name ~bm ~bn ~bk ~elem:Elemtype.Float16 ~m ~n ~k:kk))
  | Kernel_ast.SM80, Kernel_ast.BF16 ->
    Ok (Pack (Kernel_desc.make_ampere ~name ~bm ~bn ~bk ~elem:Elemtype.Bfloat16 ~m ~n ~k:kk))
  | Kernel_ast.SM80, Kernel_ast.S8 ->
    Ok (Pack (Kernel_desc.make_ampere ~name ~bm ~bn ~bk ~elem:Elemtype.Int8 ~m ~n ~k:kk))
  | Kernel_ast.SM90, Kernel_ast.F16 ->
    Ok (Pack (Kernel_desc.make_hopper ~name ~bm ~bn ~bk ~elem:Elemtype.Float16 ~m ~n ~k:kk))
  | Kernel_ast.SM90, Kernel_ast.BF16 ->
    Ok (Pack (Kernel_desc.make_hopper ~name ~bm ~bn ~bk ~elem:Elemtype.Bfloat16 ~m ~n ~k:kk))
  | Kernel_ast.SM100, Kernel_ast.F16 ->
    Ok (Pack (Kernel_desc.make_blackwell ~name ~bm ~bn ~bk ~elem:Elemtype.Float16 ~m ~n ~k:kk))
  | Kernel_ast.SM100, Kernel_ast.BF16 ->
    Ok (Pack (Kernel_desc.make_blackwell ~name ~bm ~bn ~bk ~elem:Elemtype.Bfloat16 ~m ~n ~k:kk))
  | Kernel_ast.SM100, Kernel_ast.S8 ->
    Ok (Pack (Kernel_desc.make_blackwell ~name ~bm ~bn ~bk ~elem:Elemtype.Int8 ~m ~n ~k:kk))
  | arch, elem ->
    let arch_s = match arch with
      | Kernel_ast.SM80  -> "SM80"
      | Kernel_ast.SM90  -> "SM90"
      | Kernel_ast.SM100 -> "SM100"
    in
    let elem_s = match elem with
      | Kernel_ast.F16  -> "F16"  | Kernel_ast.BF16 -> "BF16"
      | Kernel_ast.F32  -> "F32"  | Kernel_ast.S8   -> "S8"
      | Kernel_ast.S32  -> "S32"
    in
    Error (UnsupportedElem (Printf.sprintf "arch=%s elem=%s" arch_s elem_s))

let lower_exn (k : Kernel_ast.kernel) : packed =
  match lower k with
  | Ok p -> p
  | Error e ->
      let msg = Stdlib.Format.asprintf "%a" pp_error e in
      failwith msg
