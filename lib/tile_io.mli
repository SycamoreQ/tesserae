(** TileIO — owns data movement for a kernel tile.

    Abstracts over:
    - SM80: cp.async gmem → smem
    - SM90: TMA load gmem → smem
    - SM100: TMA multicast gmem → smem (2SM)

    A single [load] call hides all copy atom / cluster differences
    behind one typed interface. *)

(** The copy strategy — which hardware path to use. *)
type strategy =
  | CpAsync      (** SM80 cp.async, 16 bytes per thread            *)
  | TmaLoad      (** SM90/SM100 TMA, bulk 128 bytes                *)
  | TmaMulticast (** SM100 2SM TMA, each CTA loads half of B       *)

(** A TileIO descriptor. *)
type ('elem) t = {
  strategy     : strategy;
  tiled_copy_a : (Memspace.global, Memspace.shared, 'elem) Tiled_copy.t;
  tiled_copy_b : (Memspace.global, Memspace.shared, 'elem) Tiled_copy.t;
  cluster      : Cluster.t;
  pipeline     : Pipeline.t;
  swizzle      : Swizzle.t;
}

(** [make strategy elem cluster pipeline swizzle bm bn bk] constructs a TileIO.
    Selects the appropriate copy atoms based on strategy.
    Raises [Invalid_argument] if the strategy is incompatible with the cluster. *)
val make :
  strategy ->
  'elem Elemtype.t ->
  Cluster.t ->
  Pipeline.t ->
  Swizzle.t ->
  int -> int -> int ->
  'elem t

(** [is_tma t] returns true iff TMA is used. *)
val is_tma : _ t -> bool

(** [requires_mbar t] returns true iff mbarriers are needed. *)
val requires_mbar : _ t -> bool

(** [bytes_per_load_a t bm bk] returns bytes transferred per A tile load. *)
val bytes_per_load_a : _ t -> int -> int -> int

(** [bytes_per_load_b t bn bk cta_group] returns bytes per B tile load.
    For TmaMulticast, bn is halved since each CTA loads half. *)
val bytes_per_load_b : _ t -> int -> int -> Tmem.cta_group -> int

(** [emit_tma_load_a t var_a tmap_a mbar row col k] emits a TMA load for A.
    Handles both 2D (SM90) and 3D (SM100 swizzled) variants. *)
val emit_tma_load_a :
  _ t -> string -> string -> string -> string -> string -> string -> string

(** [emit_tma_load_b t var_b tmap_b mbar row col k] emits a TMA load for B. *)
val emit_tma_load_b :
  _ t -> string -> string -> string -> string -> string -> string -> string

(** [emit_cp_async_load t var src offset mbar] emits a cp.async load. *)
val emit_cp_async_load : _ t -> string -> string -> string -> string -> string

(** [emit_mbar_expect t mbar_var bm bn bk] emits the mbarrier expect_tx call
    with the correct byte count for the current strategy. *)
val emit_mbar_expect : _ t -> string -> int -> int -> int -> string

(** [pp fmt t] pretty-prints the TileIO descriptor. *)
val pp : Stdlib.Format.formatter -> _ t -> unit
