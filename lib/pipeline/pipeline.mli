(** Software pipeline descriptor for warp-specialized GEMM kernels.

    Manages the circular smem buffer used by producer/consumer warps.
    Each stage holds one A+B tile pair. The phase bit toggles every
    full revolution to distinguish old from new arrivals on mbarriers. *)

type t = {
  depth      : int;    (** number of pipeline stages = smem slots  *)
  tile_bytes : int;    (** bytes per stage = (BM + BN/cta) * BK * elem_bytes *)
  smem_bytes : int;    (** total smem = depth * tile_bytes          *)
}

(** [make depth tile_bytes] constructs a pipeline descriptor.
    Raises [Invalid_argument] if depth < 1 or tile_bytes <= 0. *)
val make : int -> int -> t

(** [stage_of iter depth] returns the smem stage index for iteration [iter].
    = iter mod depth *)
val stage_of : int -> int -> int

(** [phase_of iter depth] returns the mbarrier phase bit.
    Toggles every [depth] iterations.
    = (iter / depth) mod 2 *)
val phase_of : int -> int -> int

(** [smem_offset_of t stage] returns byte offset into the smem buffer
    for the given stage. = stage * tile_bytes *)
val smem_offset_of : t -> int -> int

(** [a_smem_offset_of t stage bm bk elem_bytes] returns the byte offset
    for the A tile in stage [stage].
    A tile is always at the start of each stage slot. *)
val a_smem_offset_of : t -> int -> int -> int -> int -> int

(** [b_smem_offset_of t stage bm bk elem_bytes] returns the byte offset
    for the B tile in stage [stage].
    B tile follows the A tile: offset = stage_offset + BM * BK * elem_bytes *)
val b_smem_offset_of : t -> int -> int -> int -> int -> int

(** [emit_full_mbar var_name t] emits the full_mbar declaration.
    "__shared__ __align__(8) uint64_t {var_name}[{depth}];" *)
val emit_full_mbar : string -> t -> string

(** [emit_empty_mbar var_name t] emits the empty_mbar declaration. *)
val emit_empty_mbar : string -> t -> string

(** [emit_smem_buf var_name t] emits the dynamic smem buffer declaration.
    "extern __shared__ char {var_name}[];" *)
val emit_smem_buf : string -> t -> string

(** [emit_advance_stage var_name t] emits the stage advance logic.
    "{var_name} = ({var_name} + 1) % {depth};" *)
val emit_advance_stage : string -> t -> string

(** [emit_phase_toggle var_name t depth_var] emits the phase toggle.
    "if ({stage_var} == 0) {phase_var} ^= 1;" *)
val emit_phase_toggle : string -> string -> t -> string

(** [pp fmt t] pretty-prints the pipeline descriptor. *)
val pp : Stdlib.Format.formatter -> t -> unit
