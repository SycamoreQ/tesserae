(** A tensor is a layout paired with a memory space witness.
    The phantom type ['space] tracks where this tensor lives.
    ['elem] tracks the element type (float32, float16, etc). *)
type ('elem, 'space) t

(** [make layout space] creates a tensor with the given layout
    and memory space witness. No allocation occurs —
    this is a descriptor, not a buffer. *)
val make : Layout.t -> 'space Memspace.space -> ('elem, 'space) t

(** [layout t] returns the underlying layout. *)
val layout : ('elem, 'space) t -> Layout.t

(** [space t] returns the memory space witness. *)
val space : ('elem, 'space) t -> 'space Memspace.space

(** [size t] is the total number of elements. *)
val size : ('elem, 'space) t -> int

(** [rank t] is the top-level rank of the shape. *)
val rank : ('elem, 'space) t -> int

(** [local_tile t tile_shape] partitions [t] into a tiled tensor
    using [Compose.tile]. The memory space is preserved. *)
val local_tile : ('elem, 'space) t -> Modes.t -> ('elem, 'space) t

(** [transfer ~src ~dst_space] produces a new tensor descriptor
    with the same layout as [src] but in [dst_space].
    Raises [Invalid_argument] if the transfer is not valid
    per [Memspace.can_transfer]. *)
val transfer :
  src:('elem, 'src_space) t ->
  dst_space:'dst_space Memspace.space ->
  ('elem, 'dst_space) t

(** [pp fmt t] pretty-prints as [space:layout]. *)
val pp : Stdlib.Format.formatter -> ('elem, 'space) t -> unit
