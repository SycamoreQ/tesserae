open Base
open Tesserae_core


type ('arch, 'a, 'b, 'c, 'd) t = {
  atom  : ('arch, 'a, 'b, 'c, 'd) Mma_atom.t;
  thread_layout : Layout.t;
  warp_layout : Layout.t;
  tiler_mn : Modes.t;
}

let make atom thread_layout warp_layout tiler_mn =
  if Layout.size thread_layout <> Mma_atom.thread_count atom then
    invalid_arg "thread_layout size does not match atom thread count";
  { atom; thread_layout; warp_layout; tiler_mn }

let thread_count (t : (_, _, _, _, _) t) : int =
  Layout.size t.thread_layout * Layout.size t.warp_layout

let warp_count (t : (_, _, _, _, _) t) : int =
  Layout.size t.warp_layout

let tile_shape_mnk (t : (_, _, _, _, _) t) : int * int * int =
  match Modes.flatten t.tiler_mn with
  | [m; n] ->
      let (_, _, atom_k) = Mma_atom.shape t.atom in
      (m, n, atom_k)
  | _ -> failwith "tiler_mn must represent exactly two dimensions (M and N)"


let partition_c (t : (_, _, _, _, _) t) : Layout.t =
  let (a_m, a_n, _) = Mma_atom.shape t.atom in
  let total_c = a_m * a_n in
  let per_thread = total_c / Layout.size t.thread_layout in
  Layout.make (Modes.Int per_thread) (Modes.Int 1)


let partition_a (t : (_, _, _, _, _) t) : Layout.t =
  let (a_m, _, a_k) = Mma_atom.shape t.atom in
  let total_a = a_m * a_k in
  let per_thread = total_a / Layout.size t.thread_layout in
  Layout.make
    (Modes.Int per_thread)
    (Modes.Int 1)


let partition_b (t : (_, _, _, _, _) t) : Layout.t =
  let (_, a_n, a_k) = Mma_atom.shape t.atom in
  let total_b = a_n * a_k in
  let per_thread = total_b / Layout.size t.thread_layout in
  Layout.make
    (Modes.Int per_thread)
    (Modes.Int 1)


let emit_cpp (t : (_, _, _, _, _) t) : string =
  Printf.sprintf "TiledMMA<\n  MMA_Atom<%s>,\n  %s,\n  %s>"
    (Mma_atom.emit_cpp t.atom)
    (Codegen.emit_layout t.thread_layout)
    (Codegen.emit_layout t.warp_layout)

let pp (fmt : Stdlib.Format.formatter) (t : (_, _, _, _, _) t) : unit =
  let (m, n, k) = tile_shape_mnk t in
  Stdlib.Format.fprintf fmt "TiledMMA(%dx%dx%d) atom=%s threads=%d warps=%d"
    m n k
    (Mma_atom.emit_cpp t.atom)
    (thread_count t)
    (warp_count t)
