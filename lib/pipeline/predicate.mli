open Tesserae_core

(** Predicate tensors for boundary handling in tiled GEMM kernels.

    When a tile doesn't fit evenly into the problem dimensions (M, N, K),
    threads need to mask off out-of-bounds accesses. This module provides
    the machinery to compute per-thread predicates.

    In CuTe this corresponds to:
    - make_identity_tensor  — a tensor whose value at index i is i itself
    - coordinate tensors    — tensors that hold (row, col) coordinates
    - predicate evaluation  — compare coordinates against problem bounds *)

(** A predicate tensor — a layout paired with problem bounds.
    Each "element" is a boolean indicating whether that index is in-bounds. *)
type t = {
  layout : Layout.t;
  bounds : int list;
    (** One bound per mode — the problem dimension size for each mode.
        e.g. [M; K] for an A tile, [K; N] for a B tile. *)
}

(** [make layout bounds] constructs a predicate tensor.
    Raises [Invalid_argument] if length of bounds != rank of layout. *)
val make : Layout.t -> int list -> t

(** [is_in_bounds p coord] returns true iff all coordinates are
    within their respective bounds.
    [coord] is a flat int list with one entry per mode leaf.
    Raises [Invalid_argument] if length of coord != rank of layout. *)
val is_in_bounds : t -> int list -> bool

(** [count_valid p] returns the number of in-bounds elements
    by iterating over all logical indices of the layout. *)
val count_valid : t -> int

(** [needs_predication p] returns true iff any bound does not
    evenly divide the corresponding layout dimension.
    i.e. true when the tile doesn't fit perfectly and masking is needed. *)
val needs_predication : t -> bool

(** [residue p] returns for each mode the number of valid elements
    in the last (potentially partial) tile.
    residue_i = bounds_i mod shape_leaf_i
    If residue_i = 0 the tile is full in that dimension. *)
val residue : t -> int list

(** [emit_predicate_check p coord_var] emits a C++ boolean expression
    for in-bounds checking.
    e.g. with bounds [M; N] and coord_var "coord":
    "get<0>(coord) < M && get<1>(coord) < N"
    where M and N are the actual bound values. *)
val emit_predicate_check : t -> string -> string

(** [emit_identity_tensor var_name layout] emits a CuTe
    make_identity_tensor call.
    "auto {var_name} = make_identity_tensor(Layout<...>{});" *)
val emit_identity_tensor : string -> Layout.t -> string

(** [pp fmt p] pretty-prints the predicate. *)
val pp : Stdlib.Format.formatter -> t -> unit
