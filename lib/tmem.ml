open Base

type cta_group =
  | CTA1
  | CTA2

type layout_variant =
  | LayoutD

type fragment =
  | Frag32_32b
  | Frag8_16b

type t = {
  cta_group     : cta_group;
  layout        : layout_variant;
  fragment      : fragment;
  num_cols      : int;
  num_rows      : int;
  elem_type     : Elemtype.float32 Elemtype.t;
}

let make ~(cta_group : cta_group) ~(num_cols : int) ~(num_rows : int) : t =
  if num_rows > 128 then
    invalid_arg "num_rows must be <= 128"
  else if num_cols > 512 then
    invalid_arg "num_cols must be <= 512"
  else if (match cta_group with CTA1 -> num_cols > 256 | CTA2 -> false) then
    invalid_arg "CTA1 num_cols must be <= 256"
  else
    { cta_group
    ; layout   = LayoutD
    ; fragment = Frag32_32b
    ; num_cols
    ; num_rows
    ; elem_type = Elemtype.Float32
    }


let address ~(row : int) ~(col : int) : int =
  let ans1 = row lsl 16 in
  let address = ans1 lor col in

  address

let warp_row_offset (_t : t) (warp_id : int) : int =
  warp_id * 32

let elems_per_thread_per_load (t : t) : int =
  match t.fragment with
    | Frag32_32b -> 8
    | Frag8_16b -> 1

let num_loads_per_warp (t : t) : int =
  match t.fragment with
  | Frag32_32b -> t.num_cols / 8
  | Frag8_16b -> t.num_cols

let total_elems (t : t) : int =
  t.num_rows * t.num_cols

let bytes (t : t) : int =
  let total_elements = total_elems t in
  total_elements * 4

let cta_group_str (t : t) : string =
  match t.cta_group with
  | CTA1 -> "1"
  | CTA2 -> "2"

let alloc_ptx (t : t) (smem_var : string) : string =
  Printf.sprintf
    "tcgen05.alloc.cta_group::%s.sync.aligned.shared::cta.b32 [%s], %d;"
    (cta_group_str t) smem_var t.num_cols

let dealloc_ptx (t : t) (taddr_var : string) : string =
  Printf.sprintf
    "tcgen05.dealloc.cta_group::%s.sync.aligned.b32 %s, %d;"
    (cta_group_str t) taddr_var t.num_cols

let ld_ptx (t : t) (taddr_var : string) (dst_vars : string list) (n_col : int) : string =
  let regs = String.concat ~sep:", " dst_vars in
  let addr = Printf.sprintf "%s + (%d << 16) + %d"
    taddr_var (warp_row_offset t 0) n_col in
  Printf.sprintf
    "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%s}, [%s];"
    regs addr

let commit_ptx (t : t) (mbar_var : string) : string =
  Printf.sprintf
    "tcgen05.commit.cta_group::%s.mbarrier::arrive::one.shared::cluster.b64 [%s];"
    (cta_group_str t) mbar_var

let emit_shared_storage (_t : t) (smem_var : string) (mbar_var : string) : string =
  Printf.sprintf
    "__shared__ uint32_t %s[1];\n__shared__ uint64_t %s[1];"
    smem_var mbar_var

let pp (fmt : Stdlib.Format.formatter) (t : t) : unit =
  Stdlib.Format.fprintf fmt
    "Tmem(cta_group=%s rows=%d cols=%d elems=%d bytes=%d)"
    (cta_group_str t) t.num_rows t.num_cols
    (total_elems t) (bytes t)
