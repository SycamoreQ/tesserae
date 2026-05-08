open Base

type accum_loc =
  | Registers
  | TensorMem

type ('arch, 'a, 'b, 'c, 'd) t = {
  tiled_mma   : ('arch, 'a, 'b, 'c, 'd) Tiled_mma.t;
  accum_loc   : accum_loc;
  tmem        : Tmem.t option;
  smem_desc_a : Smem_desc.t option;
  smem_desc_b : Smem_desc.t option;
  double_buf  : bool;
}

let make
    (tiled_mma  : ('arch, 'a, 'b, 'c, 'd) Tiled_mma.t)
    (accum_loc : accum_loc)
    ?(tmem : Tmem.t option)
    ?(double_buf : bool = false)
    () : ('arch, 'a, 'b, 'c, 'd) t =
  (match accum_loc with
  | TensorMem ->
    if Option.is_none tmem then
      invalid_arg "TensorMem requires a tmem descriptor"
  | Registers -> ());
  { tiled_mma
  ; accum_loc
  ; tmem
  ; smem_desc_a = None
  ; smem_desc_b = None
  ; double_buf
  }

let is_tmem (t : (_, _, _, _, _) t) : bool =
  match t.accum_loc with
  | Registers -> false
  | TensorMem -> true

let is_wgmma (t : (_, _, _, _, _) t) : bool =
  Mma_atom.is_wgmma t.tiled_mma.atom

let accum_elems_per_thread (t : (_, _, _, _, _) t) : int =
  let layout = Tiled_mma.partition_c t.tiled_mma in
  Layout.size layout

let emit_mma
    (t : (_, _, _, _, _) t)
    (a_desc : string)
    (b_desc : string)
    (enable_accum : bool) : string =
  let enable = if enable_accum then 1 else 0 in
  match t.accum_loc with
  | Registers ->
    Printf.sprintf "mma.sync.aligned.%s %s, %s, %s, %s;"
      (Mma_atom.emit_cpp t.tiled_mma.Tiled_mma.atom)
      "acc" a_desc b_desc "acc"
  | TensorMem ->
    let cg = match t.tmem with
      | Some tm -> Tmem.cta_group_str tm
      | None    -> "1"
    in
    Printf.sprintf
      "tcgen05.mma.cta_group::%s.kind::mxf16 [tmem_addr], %s, %s, [i_desc], %d;"
      cg a_desc b_desc enable

let emit_commit
    (t : (_, _, _, _, _) t)
    (mbar_var : string)
    (cta_mask : int) : string =
  match t.accum_loc with
  | Registers -> ""
  | TensorMem ->
    match t.tmem with
    | None    -> ""
    | Some tm ->
      if cta_mask > 1 then
        Tmem.commit_multicast_ptx tm mbar_var cta_mask
      else
        Tmem.commit_ptx tm mbar_var

let emit_accum_decl
    (t : (_, _, _, _, _) t)
    (var_name : string) : string =
  match t.accum_loc with
  | TensorMem -> ""
  | Registers ->
    let n = accum_elems_per_thread t in
    let zeros = String.concat ~sep:", "
      (List.init n ~f:(fun _ -> "0.0f")) in
    Printf.sprintf "float %s[%d] = {%s};" var_name n zeros

let emit_tmem_alloc
    (t : (_, _, _, _, _) t)
    (smem_var : string) : string =
  match t.tmem with
  | None -> ""
  | Some tm -> Tmem.alloc_ptx tm smem_var

let emit_tmem_dealloc
    (t : (_, _, _, _, _) t)
    (taddr_var : string) : string =
  match t.tmem with
  | None    -> ""
  | Some tm -> Tmem.dealloc_ptx tm taddr_var

let pp (fmt : Stdlib.Format.formatter) (t : (_, _, _, _, _) t) : unit =
  let loc = match t.accum_loc with
    | Registers -> "registers"
    | TensorMem -> "tmem"
  in
  Stdlib.Format.fprintf fmt
    "TileOp(atom=%s accum=%s wgmma=%b double_buf=%b)"
    (Mma_atom.emit_cpp t.tiled_mma.Tiled_mma.atom)
    loc
    (is_wgmma t)
    t.double_buf
