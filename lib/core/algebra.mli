(** Layout algebra: sort, coalesce, complement.
    All operations preserve the layout function unless stated otherwise. *)

(** [sort l] returns a layout with modes reordered so that strides are
    non-decreasing. When two strides are equal, the mode with smaller
    shape comes first.
    Sorting changes the layout function — it is NOT semantics-preserving.
    It is a prerequisite for complementation and canonical forms. *)
val sort : Layout.t -> Layout.t

(** [coalesce l] simplifies a layout by merging adjacent modes where
    possible, without changing the layout function.

    Two adjacent flat modes [(N0,N1):(d0,d1)] can be merged into
    [(N0*N1):d0] exactly when [d1 = N0 * d0].

    Modes of size 1 are always dropped.
    The result has the minimum number of modes needed to represent
    the same mapping. *)
val coalesce : Layout.t -> Layout.t

(** [is_admissible l m] returns true iff the pair [{l, m}] is admissible
    for complementation, meaning:
    - [l] is sorted
    - for all adjacent modes i, [N_i * d_i] divides [d_{i+1}]
    - [N_last * d_last] divides [m] *)
val is_admissible : Layout.t -> int -> bool

(** [complement l m] computes the complement layout B of [{l, m}] such
    that the concatenation [(l, B)] forms a bijection on [0, m).

    Requires [{l, m}] to be admissible for complementation.
    Raises [Invalid_argument] if not admissible or [m <= 0].

    The complement has shape and stride:
      shape  = (d0, d1/(N0*d0), d2/(N1*d1), ..., m/(N_last*d_last))
      stride = (1,  N0*d0,      N1*d1,      ..., N_last*d_last) *)
val complement : Layout.t -> int -> Layout.t

(** [cosize l] is the codomain size — same as [Layout.cosize] but
    exposed here for use in admissibility checks. *)
val cosize : Layout.t -> int

(** [flat_divide layout divisor] divides a flat (single-mode) layout
    by [divisor], producing a 2-mode layout where:
    - mode 0 indexes within a division (size = divisor)
    - mode 1 indexes across divisions (size = layout_size / divisor)

    Requires: [divisor] divides [Layout.size layout] exactly.
    Requires: [layout] must be a single flat mode after coalescing.
    Raises [Invalid_argument] if divisor does not divide evenly. *)
val flat_divide : Layout.t -> int -> Layout.t

(** [logical_divide layout tile] divides [layout] by [tile] layout,
    producing a result where:
    - The first mode has the shape of [tile] and indexes within a tile
    - The second mode indexes across tiles

    This is the operation used in CuTe's [local_partition] to distribute
    a CTA-level layout across threads.

    Requires: each leaf of [tile.shape] divides the corresponding
    leaf of [layout.shape].
    Raises [Invalid_argument] if shapes are incompatible or
    tile does not divide layout. *)
val logical_divide : Layout.t -> Layout.t -> Layout.t

(** [zipped_divide layout tile] is like [logical_divide] but the result
    is "zipped" — the within-tile and across-tile modes are interleaved
    rather than grouped.

    For a 2D layout divided by a 2D tile, the result has shape:
    ((within_row, within_col), (across_row, across_col))
    instead of
    (within_row, across_row, within_col, across_col).

    This matches CuTe's [zipped_divide] used for MMA partitioning. *)
val zipped_divide : Layout.t -> Layout.t -> Layout.t
