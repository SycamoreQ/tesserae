open Tesserae_core

(** TiledMMA — combines an MMA atom with a thread layout to produce
    a partitioned MMA operation across a warpgroup or CTA.

    In CuTe, a TiledMMA answers the question:
    "Given a CTA-level tile, how does each thread get its fragment?" *)

(** A tiled MMA descriptor. *)
type ('arch, 'a, 'b, 'c, 'd) t = {
  atom          : ('arch, 'a, 'b, 'c, 'd) Mma_atom.t;
  thread_layout : Layout.t;
    (** Layout of threads within the MMA tile.
        shape encodes how threads are arranged spatially.
        e.g. (32,):(1,) for a single warp,
             (128,):(1,) for a warpgroup *)
  warp_layout   : Layout.t;
    (** Layout of warps within the CTA.
        e.g. (4,1):(1,0) for 4 warps in M, 1 in N *)
  tiler_mn      : Modes.t;
    (** The CTA-level tile shape in (M, N).
        Must be divisible by atom.(m, n) * warp_layout.shape *)
}

(** [make atom thread_layout warp_layout tiler_mn] constructs a TiledMMA.
    Raises [Invalid_argument] if:
    - thread_layout size does not match atom thread count
    - tiler_mn is not divisible by atom shape * warp count *)
val make :
  ('arch, 'a, 'b, 'c, 'd) Mma_atom.t ->
  Layout.t ->
  Layout.t ->
  Modes.t ->
  ('arch, 'a, 'b, 'c, 'd) t

(** [thread_count t] returns total threads participating. *)
val thread_count : (_, _, _, _, _) t -> int

(** [warp_count t] returns total warps in the tiled MMA. *)
val warp_count : (_, _, _, _, _) t -> int

(** [tile_shape_mnk t] returns the full CTA tile shape (M, N, K)
    where K comes from the atom. *)
val tile_shape_mnk : (_, _, _, _, _) t -> int * int * int

(** [partition_c t] returns the layout of the C fragment
    owned by a single thread.
    Shape is (atom_m * warp_m / threads, atom_n * warp_n / threads).
    This is what each thread accumulates into. *)
val partition_c : (_, _, _, _, _) t -> Layout.t

(** [partition_a t] returns the layout of the A fragment
    owned by a single thread.
    Shape is (atom_m / threads_m, atom_k). *)
val partition_a : (_, _, _, _, _) t -> Layout.t

(** [partition_b t] returns the layout of the B fragment
    owned by a single thread.
    Shape is (atom_n / threads_n, atom_k). *)
val partition_b : (_, _, _, _, _) t -> Layout.t

(** [emit_cpp t] emits the CuTe TiledMMA type string.
    e.g.
    "TiledMMA
       MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
       Layout<Shape<_32,_1,_1>>,
       Layout<Shape<_1,_2,_1>>
     >" *)
val emit_cpp : (_, _, _, _, _) t -> string

(** [pp fmt t] pretty-prints the tiled MMA. *)
val pp : Stdlib.Format.formatter -> (_, _, _, _, _) t -> unit
