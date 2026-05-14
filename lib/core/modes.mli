(** A [Mode.t] is a node in CuTe's integer mode tree.
    It is either a leaf integer or a hierarchical tuple of sub-modes.
    Both Shape and Stride are represented as [Mode.t]. *)
type t =
  | Int of int
  | Tuple of t list

(** [size m] is the total number of elements a shape [m] represents.
    For a leaf, this is the integer itself.
    For a tuple, this is the product of the sizes of its children. *)
val size : t -> int

(** [depth m] is the nesting depth of the mode.
    A leaf has depth 0. A tuple of leaves has depth 1. *)
val depth : t -> int

(** [rank m] is the number of top-level children.
    A leaf has rank 1. A tuple has rank equal to its list length. *)
val rank : t -> int

(** [flatten m] collapses a nested mode into a flat [Int list],
    visiting leaves left-to-right in tree order. *)
val flatten : t -> int list

(** [compatible shape stride] returns true iff the shape and stride trees
    have identical structure — same nesting at every level. *)
val compatible : t -> t -> bool

val pp : Format.formatter -> t -> unit
