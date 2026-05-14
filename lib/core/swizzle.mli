(** CuTe Swizzle<B,M,S> — a bit-level bijection on linear offsets
    used to eliminate shared memory bank conflicts.

    The canonical patterns are always Swizzle<B,4,3> where:
      B=0 → identity (no swizzle)
      B=1 → 32B swizzle
      B=2 → 64B swizzle
      B=3 → 128B swizzle

    The swizzle function on a linear offset [x] is:
      yyy_msk = ((1 << B) - 1) << (M + max(0, S))
      apply(x) = x XOR ((x AND yyy_msk) >> S)

    Swizzle is its own inverse: applying it twice returns the original. *)

type t = {
  b : int;
  m : int;
  s : int;
}

(** [make b m s] constructs a swizzle descriptor.
    Raises [Invalid_argument] if any parameter is negative. *)
val make : int -> int -> int -> t

(** [apply sw offset] applies the swizzle to a linear offset.
    yyy_msk = ((1 lsl b) - 1) lsl (m + max(0, s))
    result  = offset lxor ((offset land yyy_msk) lsr s) *)
val apply : t -> int -> int

(** [is_identity sw] returns true iff [sw.b = 0]. *)
val is_identity : t -> bool

(** [inverse sw] returns the inverse. Since XOR is self-inverse, returns [sw]. *)
val inverse : t -> t

(** [mask_bits sw] returns [1 lsl sw.b] — number of elements in the XOR group. *)
val mask_bits : t -> int

(** [compose sw1 sw2] composes two swizzles.
    Requires sw1.m = sw2.m and sw1.s = sw2.s.
    Result has b = sw1.b + sw2.b.
    Raises [Invalid_argument] if shifts are incompatible. *)
val compose : t -> t -> t

(** [apply_to_layout sw l] produces a new layout where each stride leaf
    is swizzle-adjusted via [apply sw]. *)
val apply_to_layout : t -> Layout.t -> Layout.t

(** [smem_selector elem tile_m tile_k] selects the canonical
    Swizzle<B,4,3> for a shared memory layout.

    M is always 4, S is always 3.
    B = min(3, floor(log2(tile_k * byte_width / 16)))
    capped at 3 (128B max). *)
val smem_selector : _ Elemtype.t -> int -> int -> t

(** [pp fmt sw] pretty-prints as "Swizzle<B,M,S>". *)
val pp : Stdlib.Format.formatter -> t -> unit

(** [emit_cpp sw] emits the CuTe C++ type string "Swizzle<B,M,S>". *)
val emit_cpp : t -> string
