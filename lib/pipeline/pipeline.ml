open Base

type t = {
  depth : int;
  tile_bytes : int;
  smem_bytes : int;
}

let make (depth : int) (tile_bytes : int) : t =
  if depth < 1 then
    invalid_arg  "pipeline depth must be >= 1"
  else if tile_bytes <= 0 then
    invalid_arg "tile_bytes must be > 0"
  else
    {depth ; tile_bytes ; smem_bytes = depth * tile_bytes }

let stage_of (iter : int) (depth : int) : int =
  iter % depth

let phase_of (iter : int) (depth : int) : int =
  (iter / depth ) % 2

let smem_offset_of (t : t) (stage : int) : int =
  stage * t.tile_bytes

let a_smem_offset_of (t : t) (stage : int) (_bm : int) (_bk : int) (_elem_bytes : int) : int =
  stage * t.tile_bytes

let b_smem_offset_of (t : t) (stage : int) (bm : int) (bk : int) (elem_bytes : int) : int =
  let a_tile_bytes = bm * bk * elem_bytes in
  stage * t.tile_bytes + a_tile_bytes

let emit_full_mbar (var_name : string) (t : t) : string =
  Printf.sprintf "__shared__ __align__(8) uint64_t %s[%d];" var_name t.depth

let emit_empty_mbar (var_name : string) (t : t) : string =
  Printf.sprintf "__shared__ __align__(8) uint64_t %s[%d];" var_name t.depth

let emit_smem_buf (var_name : string) (_t : t) : string =
  Printf.sprintf "extern __shared__ char %s[];" var_name

let emit_advance_stage (var_name : string) (t : t) : string =
  Printf.sprintf "%s = (%s + 1) %% %d;" var_name var_name t.depth

let emit_phase_toggle (phase_var : string) (stage_var : string) (_t : t) : string =
  Printf.sprintf "if (%s == 0) %s ^= 1;" stage_var phase_var

let pp (fmt : Stdlib.Format.formatter) (t : t) : unit =
  let open Stdlib.Format in
  fprintf fmt "@[<hov 2>Pipeline Descriptor:@ ";
  fprintf fmt "depth: %d,@ " t.depth;
  fprintf fmt "tile_bytes: %d,@ " t.tile_bytes;
  fprintf fmt "smem_bytes: %d@]" t.smem_bytes
