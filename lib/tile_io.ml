open Base

type strategy =
  | CpAsync
  | TmaLoad
  | TmaMulticast

type 'elem t = {
  strategy : strategy;
  tiled_copy_a : (Memspace.global, Memspace.shared, 'elem) Tiled_copy.t;
  tiled_copy_b : (Memspace.global, Memspace.shared, 'elem) Tiled_copy.t;
  cluster : Cluster.t;
  pipeline : Pipeline.t;
  swizzle : Swizzle.t;
}

let make
    (strategy : strategy)
    (elem : 'elem Elemtype.t)
    (cluster : Cluster.t)
    (pipeline : Pipeline.t)
    (swizzle : Swizzle.t)
    (bm : int) (bn : int) (_bk : int)
  : 'elem t =
  (* validate strategy vs cluster *)
  (match strategy with
  | TmaMulticast ->
    if not (Cluster.is_2sm cluster) then
      invalid_arg "TmaMulticast requires a 2SM cluster"
  | _ -> ());
  (* select copy atom based on strategy *)
  let atom_a, atom_b = match strategy with
    | CpAsync ->
      Copy_atom.sm80_cp_async_global elem,
      Copy_atom.sm80_cp_async_global elem
    | TmaLoad ->
      Copy_atom.sm90_tma_load elem,
      Copy_atom.sm90_tma_load elem
    | TmaMulticast ->
      Copy_atom.sm100_tma_load_multicast elem,
      Copy_atom.sm100_tma_load_multicast elem
  in
  (* thread layout: TMA uses 1 thread to issue, cp.async uses 32 *)
  let thread_layout = match strategy with
    | CpAsync ->
      let vec  = Elemtype.vec_width elem in
      let threads = (bm * bn) / vec in
      Layout.make (Modes.Int threads) (Modes.Int 1)
    | TmaLoad | TmaMulticast ->
      Layout.make (Modes.Int 128) (Modes.Int 1)
  in
  (* val layout = vec_width elements per thread *)
  let val_layout = match strategy with
    | CpAsync ->
      Layout.make (Modes.Int (Elemtype.vec_width elem)) (Modes.Int 1)
    | TmaLoad | TmaMulticast ->
      let bulk = 128 / Elemtype.byte_width elem in
      Layout.make (Modes.Int bulk) (Modes.Int 1)
  in
  { strategy
  ; tiled_copy_a = Tiled_copy.make atom_a thread_layout val_layout
  ; tiled_copy_b = Tiled_copy.make atom_b thread_layout val_layout
  ; cluster
  ; pipeline
  ; swizzle
  }

let is_tma (t : _ t) : bool =
  Tiled_copy.is_tma t.tiled_copy_a

let requires_mbar (t : _ t) : bool =
  Tiled_copy.requires_mbar t.tiled_copy_a

let bytes_per_load_a (t : _ t) (bm : int) (bk : int) : int =
  let bw = Elemtype.byte_width t.tiled_copy_a.Tiled_copy.atom.Copy_atom.elem_type in
  bm * bk * bw

let bytes_per_load_b (t : _ t) (bn : int) (bk : int) (cta_group : Tmem.cta_group) : int =
  let bw = Elemtype.byte_width t.tiled_copy_b.Tiled_copy.atom.Copy_atom.elem_type in
  let effective_bn = match cta_group with
    | Tmem.CTA2 -> bn / 2
    | Tmem.CTA1 -> bn
  in
  effective_bn * bk * bw

let emit_tma_load_a (t : _ t) (var_a : string) (tmap_a : string)
    (mbar : string) (row : string) (col : string) (_k : string) : string =
  match t.strategy with
  | TmaMulticast ->
    Printf.sprintf
      "tma_2d_gmem2smem_multicast(%s, &%s, %s, %s, %s, 0b11);"
      var_a tmap_a col row mbar
  | TmaLoad | CpAsync ->
    Printf.sprintf
      "tma_2d_gmem2smem(%s, &%s, %s, %s, %s);"
      var_a tmap_a col row mbar

let emit_tma_load_b (t : _ t) (var_b : string) (tmap_b : string)
    (mbar : string) (row : string) (col : string) (_k : string) : string =
  match t.strategy with
  | TmaMulticast ->
    Printf.sprintf
      "tma_2d_gmem2smem_multicast(%s, &%s, %s, %s + cta_rank * (BN / 2), %s, 0b11);"
      var_b tmap_b col row mbar
  | TmaLoad | CpAsync ->
    Printf.sprintf
      "tma_2d_gmem2smem(%s, &%s, %s, %s, %s);"
      var_b tmap_b col row mbar

let emit_cp_async_load (_t : _ t) (var : string) (src : string)
    (offset : string) (_mbar : string) : string =
  Printf.sprintf
    "asm volatile(\"cp.async.ca.shared.global [%%0], [%%1], 16;\" \
     :: \"r\"(%s + %s), \"l\"(%s) : \"memory\");"
    var offset src

let emit_mbar_expect (t : _ t) (mbar_var : string)
    (bm : int) (bn : int) (bk : int) : string =
  let bw = Elemtype.byte_width t.tiled_copy_a.Tiled_copy.atom.Copy_atom.elem_type in
  let b_cols = match t.strategy with
    | TmaMulticast -> bn / 2
    | _ -> bn
  in
  let total = (bm + b_cols) * bk * bw in
  Cluster.mbarrier_arrive_expect_cluster_ptx mbar_var total

let pp (fmt : Stdlib.Format.formatter) (t : _ t) : unit =
  let strat = match t.strategy with
    | CpAsync -> "CpAsync" | TmaLoad -> "TmaLoad" | TmaMulticast -> "TmaMulticast"
  in
  Stdlib.Format.fprintf fmt "TileIO(strategy=%s pipeline_depth=%d)"
    strat t.pipeline.Pipeline.depth
