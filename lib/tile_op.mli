(** TileOp — owns MMA computation for a kernel tile.

    Abstracts over:
    - SM80: mma.sync (single warp, register accumulators)
    - SM90: wgmma.mma_async (warpgroup, register accumulators)
    - SM100: tcgen05.mma (warpgroup, TMEM accumulators)

    A single [mma] call hides atom/TMEM differences behind
    one typed interface. *)

(** Accumulator location — where results land. *)
type accum_loc =
  | Registers  (** SM80/SM90 — accumulators in thread registers *)
  | TensorMem  (** SM100     — accumulators in TMEM             *)

(** A TileOp descriptor. *)
type ('arch, 'a, 'b, 'c, 'd) t = {
  tiled_mma  : ('arch, 'a, 'b, 'c, 'd) Tiled_mma.t;
  accum_loc  : accum_loc;
  tmem       : Tmem.t option;
    (** Some if accum_loc = TensorMem, None otherwise *)
  smem_desc_a : Smem_desc.t option;
  smem_desc_b : Smem_desc.t option;
  double_buf : bool;
    (** true iff TMEM is double-buffered (Blackwell persistent kernels) *)
}

(** [make tiled_mma accum_loc ?tmem ?double_buf ()] constructs a TileOp.
    Raises [Invalid_argument] if accum_loc=TensorMem but tmem=None. *)
val make :
  ('arch, 'a, 'b, 'c, 'd) Tiled_mma.t ->
  accum_loc ->
  ?tmem:Tmem.t ->
  ?double_buf:bool ->
  unit ->
  ('arch, 'a, 'b, 'c, 'd) t

(** [is_tmem t] returns true iff accumulators live in TMEM. *)
val is_tmem : (_, _, _, _, _) t -> bool

(** [is_wgmma t] returns true iff this uses warpgroup MMA. *)
val is_wgmma : (_, _, _, _, _) t -> bool

(** [accum_elems_per_thread t] returns how many accumulator elements
    each thread owns. *)
val accum_elems_per_thread : (_, _, _, _, _) t -> int

(** [emit_mma t a_desc b_desc enable_accum] emits the MMA instruction.
    SM80:  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ..."
    SM90:  "wgmma.mma_async.sync.aligned ..."
    SM100: "tcgen05.mma.cta_group::1.kind::mxf16 ..." *)
val emit_mma :
  (_, _, _, _, _) t -> string -> string -> bool -> string

(** [emit_commit t mbar_var cta_mask] emits the commit instruction.
    SM80/SM90: empty string (synchronous or implicit)
    SM100: tcgen05.commit PTX *)
val emit_commit :
  (_, _, _, _, _) t -> string -> int -> string

(** [emit_accum_decl t var_name] emits the accumulator declaration.
    Registers: "float {var_name}[{n}] = {0.0f, ...};"
    TMEM: empty — TMEM is allocated separately *)
val emit_accum_decl : (_, _, _, _, _) t -> string -> string

(** [emit_tmem_alloc t smem_var] emits TMEM allocation if needed. *)
val emit_tmem_alloc : (_, _, _, _, _) t -> string -> string

(** [emit_tmem_dealloc t taddr_var] emits TMEM deallocation if needed. *)
val emit_tmem_dealloc : (_, _, _, _, _) t -> string -> string

(** [pp fmt t] pretty-prints the TileOp descriptor. *)
val pp : Stdlib.Format.formatter -> (_, _, _, _, _) t -> unit
