(** Tensor Memory (TMEM) descriptor for Blackwell SM100.

    TMEM is a dedicated on-chip memory for MMA accumulators.
    Physical capacity: 128 rows × 512 columns, each element 32-bit.
    Allocation granularity: columns (all 128 rows always allocated).

    Address encoding: (row << 16) | col
    where row ∈ [0,127] and col ∈ [0,511].

    Lifecycle:
      tcgen05.alloc  → one full warp allocates N columns
      tcgen05.mma    → writes accumulator results into TMEM
      tcgen05.ld     → reads TMEM into registers (epilogue)
      tcgen05.dealloc → must be called before kernel exit *)

(** CTA group for MMA — 1SM or 2SM paired MMA. *)
type cta_group =
  | CTA1  (** Single SM MMA — standard, MMA_M up to 128 *)
  | CTA2  (** 2SM pair MMA  — doubles MMA_M up to 256    *)

(** Layout of accumulators in TMEM.
    Layout D is the standard for tcgen05.mma with MMA_M=128. *)
type layout_variant =
  | LayoutD   (** Standard layout D: 4 warps × 32 rows     *)

(** Fragment shape for tcgen05.ld. *)
type fragment =
  | Frag32_32b  (** .32x32b — 8 FP32 per thread per load    *)
  | Frag8_16b (** 8 times 16 -  1 FP32 per thread per load *)

(** A TMEM descriptor — describes how a tile is stored in TMEM. *)
type t = {
  cta_group     : cta_group;
  layout        : layout_variant;
  fragment      : fragment;
  num_cols      : int;    (** number of columns allocated = tile N *)
  num_rows      : int;    (** number of rows used = tile M ≤ 128   *)
  elem_type     : Elemtype.float32 Elemtype.t;
    (** TMEM always holds float32 or int32 accumulators.
        We use float32 as the canonical type here. *)
}

(** [make ~cta_group ~num_cols ~num_rows] constructs a TMEM descriptor.
    Raises [Invalid_argument] if:
    - num_rows > 128
    - num_cols > 512
    - num_cols > 256 for CTA1 (single SM limit)
    - num_rows > 256 for CTA2 (2SM limit, but num_rows split across 2 CTAs so each ≤ 128) *)
val make : cta_group:cta_group -> num_cols:int -> num_rows:int -> t

(** [address ~row ~col] computes the TMEM address for a given row and column.
    address = (row lsl 16) lor col *)
val address : row:int -> col:int -> int

(** [warp_row_offset t warp_id] returns the starting row for a given warp.
    In Layout D with MMA_M=128 and 4 warps:
    warp 0 → rows 0..31
    warp 1 → rows 32..63
    warp 2 → rows 64..95
    warp 3 → rows 96..127 *)
val warp_row_offset : t -> int -> int

(** [elems_per_thread_per_load t] returns how many FP32 elements
    a single thread loads per tcgen05.ld instruction.
    For Frag32x32b this is always 8. *)
val elems_per_thread_per_load : t -> int

(** [num_loads_per_warp t] returns how many tcgen05.ld calls
    a single warp needs to read its entire portion of TMEM.
    = num_cols / 8  (since each .32x32b load covers 8 columns) *)
val num_loads_per_warp : t -> int

(** [total_elems t] returns total accumulator elements in the tile.
    = num_rows * num_cols *)
val total_elems : t -> int

(** [bytes t] returns total bytes used by the tile.
    = total_elems * 4  (each element is 32-bit) *)
val bytes : t -> int

(** [alloc_ptx t smem_var] emits the PTX inline assembly string
    for tcgen05.alloc.
    e.g. "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_var], num_cols;" *)
val alloc_ptx : t -> string -> string

(** [dealloc_ptx t taddr_var] emits the PTX for tcgen05.dealloc.
    e.g. "tcgen05.dealloc.cta_group::1.sync.aligned.b32 taddr_var, num_cols;" *)
val dealloc_ptx : t -> string -> string

(** [ld_ptx t taddr_var dst_vars n_col] emits the PTX for tcgen05.ld.
    Uses .32x32b.x8 to load 8 elements.
    e.g. "tcgen05.ld.sync.aligned.32x32b.x8.b32 {r0,...,r7}, [taddr];" *)
val ld_ptx : t -> string -> string list -> int -> string

(** [commit_ptx t mbar_var] emits the PTX for tcgen05.commit
    which signals MMA completion to an mbarrier.
    e.g. "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar_var];" *)
val commit_ptx : t -> string -> string

(** [emit_shared_storage t smem_var mbar_var] emits the shared memory
    declarations needed for TMEM management:
    - uint32_t smem_var[1] for the TMEM address
    - uint64_t mbar_var[n] for mbarrier(s) *)
val emit_shared_storage : t -> string -> string -> string

(** [pp fmt t] pretty-prints the TMEM descriptor. *)
val pp : Stdlib.Format.formatter -> t -> unit

(** Double-buffered TMEM — two 128×256 tiles in 512 columns.
    buf_id ∈ {0,1}, col offset = buf_id * 256. *)
val buf_col_offset : int -> int

(** [double_buf_make cta_group num_rows] creates a TMEM descriptor
    sized for double buffering — always num_cols=512. *)
val double_buf_make : cta_group:cta_group -> num_rows:int -> t

(** [commit_multicast_ptx t mbar_var cta_mask] emits tcgen05.commit
    with multicast for 2SM kernels.
    cta_mask: 0b11 selects both CTAs in the cluster.
    e.g. "tcgen05.commit.cta_group::2.mbarrier::arrive::one.multicast::cluster.shared::cluster.b64 [mbar], cta_mask;" *)
val commit_multicast_ptx : t -> string -> int -> string

(** [fence_after_thread_sync_ptx ()] emits:
    "tcgen05.fence::after_thread_sync;" *)
val fence_after_thread_sync_ptx : unit -> string

(** [before_thread_sync_ptx ()] emits:
    "tcgen05.fence::before_thread_sync;" *)
val before_thread_sync_ptx : unit -> string

(** [wait_ld_ptx ()] emits:
    "tcgen05.wait::ld.sync.aligned;" *)
val wait_ld_ptx : unit -> string

(** [ld_batched_ptx t taddr_var dst_var_groups n_cols] emits
    batched tcgen05.ld calls without intervening waits.
    Each group in dst_var_groups is a list of 8 register names. *)
val ld_batched_ptx : t -> string -> string list list -> int list -> string

val cta_group_str: t -> string
