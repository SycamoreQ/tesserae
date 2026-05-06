open Base

let rec emit_shape (m : Modes.t) : string =
  match m with
  | Modes.Int n -> Printf.sprintf "_%d" n
  | Modes.Tuple ms ->
    let inner = String.concat ~sep:"," (List.map ms ~f:emit_shape) in
    Printf.sprintf "Shape<%s>" inner

let rec emit_stride (m : Modes.t) : string =
  match m with
  | Modes.Int n -> Printf.sprintf "_%d" n
  | Modes.Tuple ms ->
    let inner = String.concat ~sep:"," (List.map ms ~f:emit_stride) in
    Printf.sprintf "Stride<%s>" inner

let emit_int_mode (m : Modes.t) : string = emit_shape m

let emit_layout (l : Layout.t) : string =
  Printf.sprintf "Layout<%s,%s>"
    (emit_shape l.Layout.shape)
    (emit_stride l.Layout.stride)

let emit_tensor_type
    (elem  : _ Elemtype.t)
    (_space : _ Memspace.space)
    (layout : Layout.t) : string =
  Printf.sprintf "Tensor<%s*, %s>"
    (Elemtype.cpp_name elem)
    (emit_layout layout)

let emit_make_tensor
    (ptr_name : string)
    (elem     : _ Elemtype.t)
    (_space   : _ Memspace.space)
    (layout   : Layout.t) : string =
  Printf.sprintf "make_tensor<%s>(%s, %s{})"
    (Elemtype.cpp_name elem)
    ptr_name
    (emit_layout layout)

let emit_smem_decl
    (var_name : string)
    (elem     : _ Elemtype.t)
    (layout   : Layout.t) : string =
  Printf.sprintf "__shared__ %s %s[%d];"
    (Elemtype.cpp_name elem)
    var_name
    (Layout.size layout)

let emit_include_guard () : string = "#pragma once"

let emit_cute_includes () : string =
  String.concat ~sep:"\n" [
    "#include <cute/cute.hpp>";
    "#include <cute/tensor.hpp>";
    "#include <cute/atom/mma_atom.hpp>";
    "#include <cute/atom/copy_atom.hpp>";
    "using namespace cute;";
  ]
