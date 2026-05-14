(** A [Layout.t] pairs a shape [Mode.t] with a stride [Mode.t].
    The two modes must be structurally compatible. *)
type t = {
  shape  : Modes.t;
  stride : Modes.t;
}

(** [make shape stride] constructs a layout.
    Raises [Invalid_argument] if shape and stride are not compatible. *)
val make : Modes.t -> Modes.t -> t

(** [size l] is the total number of elements — the product of all shape leaves. *)
val size : t -> int

(** [rank l] is the top-level rank of the shape. *)
val rank : t -> int

(** [cosize l] is the codomain size — the index one past the last element
    this layout can produce. It is the sum over each mode of
    [(shape_leaf - 1) * stride_leaf], plus 1.
    Think of it as the minimum allocation needed to hold this layout. *)
val cosize : t -> int

(** [idx l i] computes the flat memory offset for logical index [i].
    [i] must be a [Modes.t] with the same structure as [l.shape].
    Raises [Invalid_argument] if structures differ.

    Hint: each leaf of [i] is a coordinate in that mode dimension.
    Each leaf contributes [coord * stride_leaf] to the total offset. *)
val idx : t -> Modes.t -> int

(** [pp fmt l] pretty-prints as [shape:stride]. *)
val pp : Stdlib.Format.formatter -> t -> unit
