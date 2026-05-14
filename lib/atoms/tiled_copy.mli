open Tesserae_core

(** TiledCopy — combines a Copy_atom with a thread layout to produce
    a partitioned copy operation across threads.

    In CuTe, a TiledCopy answers:
    "Given a CTA-level tile, how does each thread copy its slice?" *)

type ('src, 'dst, 'elem) t = {
  atom          : ('src, 'dst, 'elem) Copy_atom.t;
  thread_layout : Layout.t;
    (** How threads are arranged to cover the tile.
        size must equal the number of threads doing the copy. *)
  val_layout    : Layout.t;
    (** How values (elements) are arranged per thread per copy instruction.
        size = atom.vec_width *)
}

(** [make atom thread_layout val_layout] constructs a TiledCopy.
    Raises [Invalid_argument] if:
    - val_layout size does not match atom.vec_width
    - thread_layout size is zero *)
val make :
  ('src, 'dst, 'elem) Copy_atom.t ->
  Layout.t ->
  Layout.t ->
  ('src, 'dst, 'elem) t

(** [thread_count t] returns the number of threads in the copy. *)
val thread_count : (_, _, _) t -> int

(** [elements_per_thread t] returns how many elements each thread
    copies per instruction = val_layout size. *)
val elements_per_thread : (_, _, _) t -> int

(** [tile_size t] returns the total elements copied per instruction
    across all threads = thread_count * elements_per_thread. *)
val tile_size : (_, _, _) t -> int

(** [is_tma t] returns true iff the underlying atom is TMA. *)
val is_tma : (_, _, _) t -> bool

(** [requires_mbar t] returns true iff the copy needs an mbarrier. *)
val requires_mbar : (_, _, _) t -> bool

(** [partition_src t layout] partitions a source layout for one thread.
    Uses [Algebra.logical_divide] to split [layout] by the thread tile,
    then returns the per-thread slice.
    The result layout has size = [elements_per_thread t]. *)
val partition_src : (_, _, _) t -> Layout.t -> Layout.t

(** [partition_dst t layout] same as [partition_src] but for destination. *)
val partition_dst : (_, _, _) t -> Layout.t -> Layout.t

(** [emit_cpp t] emits the CuTe TiledCopy type string.
    e.g.
    "TiledCopy
       Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float>,
       Layout<Shape<_32,_1>>,
       Layout<Shape<_1,_4>>
     >" *)
val emit_cpp : (_, _, _) t -> string

(** [pp fmt t] pretty-prints the tiled copy. *)
val pp : Stdlib.Format.formatter -> (_, _, _) t -> unit
