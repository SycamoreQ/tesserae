open Base

type t = {
  b : int;
  m : int;
  s : int;
}

let make (b : int) (m : int) (s : int) : t =
  if b < 0 || m < 0 || s < 0 then
    invalid_arg "swizzle parameters must be non-negative"
else
  {b;m;s}

let apply (sw : t) (offset : int) : int =
  let yyy_msk = ((1 lsl sw.b) - 1) lsl (sw.m + (max 0 sw.s)) in
  offset lxor ((offset land yyy_msk) lsr sw.s)

let is_identity (sw : t) : bool =
  match sw.b with
  | 0 -> true
  | _ -> false

let inverse (sw : t) : t =
  sw

let mask_bits (sw : t) : int =
  1 lsl sw.b

let compose (sw1 : t) (sw2 : t) : t =
  if sw1.m <> sw2.m || sw1.s <> sw2.s then
    invalid_arg "incompatible swizzle shifts for composition"
  else
    { b = sw1.b + sw2.b; m = sw1.m; s = sw1.s }

let apply_to_layout (sw : t) (l : Layout.t) : Layout.t =
  let rec swizzle_strides (sw : t) (m : Modes.t) : Modes.t =
    match m with
    | Modes.Int d -> Modes.Int (apply sw d)
    | Modes.Tuple elts -> Modes.Tuple (List.map elts ~f:(swizzle_strides sw))
  in
  { Layout.shape  = l.Layout.shape
  ; Layout.stride = swizzle_strides sw l.Layout.stride }


let smem_selector (elem : _ Elemtype.t) (_tile_m : int) (tile_k : int) : t =
  let byte_width = Elemtype.byte_width elem in
  let contiguous_bytes = tile_k * byte_width in
  let b =
    let rec log2_floor n acc =
      if n <= 1 then acc
      else log2_floor (n / 2) (acc + 1)
    in
    Int.min 3 (Int.max 0 (log2_floor (contiguous_bytes / 16) 0))
  in
  make b 4 3

let pp (fmt : Stdlib.Format.formatter) (sw : t) : unit =
  Stdlib.Format.fprintf fmt "Swizzle<%d,%d,%d>" sw.b sw.m sw.s

let emit_cpp (sw : t) : string =
  Printf.sprintf "Swizzle<%d,%d,%d>" sw.b sw.m sw.s
