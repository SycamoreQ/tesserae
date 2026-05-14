open Base
open Tesserae_core

type ('src, 'dst, 'elem) t = {
  atom : ('src, 'dst, 'elem) Copy_atom.t;
  thread_layout : Layout.t;
  val_layout : Layout.t;
}

let make
    (atom : ('src, 'dst, 'elem) Copy_atom.t)
    (thread_layout : Layout.t)
    (val_layout : Layout.t)
  : ('src, 'dst, 'elem) t =
  if Layout.size val_layout <> atom.Copy_atom.vec_width then
    invalid_arg "val_layout size does not match atom vec_width";
  if Layout.size thread_layout <= 0 then
    invalid_arg "thread_layout size must be positive";
  { atom; thread_layout; val_layout }


let thread_count (t : (_, _, _) t) : int =
  Layout.size t.thread_layout

let elements_per_thread (t : (_, _, _) t) : int =
  Layout.size t.val_layout

let tile_size (t : (_, _, _) t) : int =
  thread_count t * elements_per_thread t

let is_tma (t : (_, _, _) t) : bool =
  Copy_atom.is_tma t.atom


let requires_mbar (t : (_, _, _) t) : bool =
  Copy_atom.requires_mbar t.atom

let partition_src (t : (_, _, _) t) (layout : Layout.t) : Layout.t =
  let divided = Algebra.flat_divide layout (thread_count t) in
  match divided.Layout.shape, divided.Layout.stride with
  | Modes.Tuple [_; across_s], Modes.Tuple [_; across_d] ->
    Layout.make across_s across_d
  | _ -> divided

let partition_dst (t : (_, _, _) t) (layout : Layout.t) : Layout.t =
  let divided = Algebra.flat_divide layout (thread_count t) in
  match divided.Layout.shape, divided.Layout.stride with
  | Modes.Tuple [_; across_s], Modes.Tuple [_; across_d] ->
    Layout.make across_s across_d
  | _ -> divided

let emit_cpp (t : (_, _, _) t) : string =
  Printf.sprintf "TiledCopy<\n  Copy_Atom<%s, %s>,\n  %s,\n  %s>"
    (Copy_atom.emit_cpp t.atom)
    (Elemtype.cpp_name t.atom.Copy_atom.elem_type)
    (Codegen.emit_layout t.thread_layout)
    (Codegen.emit_layout t.val_layout)

let pp (fmt : Stdlib.Format.formatter) (t : (_, _, _) t) : unit =
  Stdlib.Format.fprintf fmt "TiledCopy atom=%s threads=%d elems_per_thread=%d"
    (Copy_atom.emit_cpp t.atom)
    (thread_count t)
    (elements_per_thread t)
