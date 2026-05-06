(** [compose outer inner] produces a layout C such that
    C(i) = outer(inner(i)).
    The shape of the result is the shape of [inner].
    The stride of the result encodes the combined mapping.

    Requires: [inner]'s cosize <= [outer]'s size.
    Raises [Invalid_argument] if this does not hold. *)
val compose : Layout.t -> Layout.t -> Layout.t

(** [tile layout tile_shape] partitions [layout] by [tile_shape].
    The result has shape [(tile_shape, tile_counts)] where
    [tile_counts] is derived from dividing [layout]'s shape by [tile_shape]
    at each leaf.

    The first mode indexes within a tile, the second indexes across tiles.

    Requires: each leaf of [tile_shape] divides the corresponding leaf
    of [layout.shape] exactly.
    Raises [Invalid_argument] if this does not hold. *)
val tile : Layout.t -> Modes.t -> Layout.t
