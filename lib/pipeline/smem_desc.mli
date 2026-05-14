open Tesserae_core

(** Shared memory descriptor for tcgen05.mma operands.

    tcgen05.mma takes A and B as 64-bit smem descriptors, not raw pointers.
    The descriptor encodes:
    - base address (low 32 bits of smem pointer, shifted right by 4)
    - leading dimension byte offset (stride between rows)
    - stride dimension byte offset
    - swizzle mode (matches the Swizzle<B,M,S> used for smem layout)

    Format (from PTX ISA):
    bits [13:0]  — base address / 16
    bits [29:16] — leading dim offset / 16
    bits [45:32] — stride dim offset / 16
    bits [62:61] — swizzle mode (0=none, 1=32B, 2=64B, 3=128B) *)

(** Swizzle mode encoded in the descriptor. *)
type swizzle_mode =
  | NoSwizzle    (** 0 — no swizzle                  *)
  | Swizzle32B   (** 1 — 32B  swizzle (B=1,M=4,S=3) *)
  | Swizzle64B   (** 2 — 64B  swizzle (B=2,M=4,S=3) *)
  | Swizzle128B  (** 3 — 128B swizzle (B=3,M=4,S=3) *)

(** A smem descriptor value. *)
type t = {
  base_addr    : int;  (** smem base address >> 4        *)
  leading_off  : int;  (** leading dim byte offset >> 4  *)
  stride_off   : int;  (** stride dim byte offset >> 4   *)
  swizzle_mode : swizzle_mode;
}

(** [swizzle_mode_of sw] converts a [Swizzle.t] to the descriptor mode.
    Swizzle b=0 → NoSwizzle
    Swizzle b=1 → Swizzle32B
    Swizzle b=2 → Swizzle64B
    Swizzle b=3 → Swizzle128B
    b>3         → Swizzle128B (clamped) *)
val swizzle_mode_of : Swizzle.t -> swizzle_mode

(** [make ~base_addr ~leading_off ~stride_off ~swizzle] constructs a descriptor. *)
val make :
  base_addr:int ->
  leading_off:int ->
  stride_off:int ->
  swizzle_mode:swizzle_mode ->
  t

(** [encode d] encodes the descriptor into a 64-bit integer.
    bits[13:0]  = base_addr
    bits[29:16] = leading_off
    bits[45:32] = stride_off
    bits[62:61] = swizzle_mode as int *)
val encode : t -> int

(** [swizzle_mode_bits m] returns the 2-bit encoding of the swizzle mode. *)
val swizzle_mode_bits : swizzle_mode -> int

(** [emit_make_smem_desc ptr_var leading stride sw] emits the C++ call
    to construct a smem descriptor at runtime.
    "uint64_t desc = make_smem_desc({ptr_var}, {leading}, {stride}, {sw_bits});" *)
val emit_make_smem_desc : string -> int -> int -> swizzle_mode -> string

(** [emit_cpp_helper ()] emits the make_smem_desc helper function
    that should appear at the top of the kernel file. *)
val emit_cpp_helper : unit -> string

(** [pp fmt d] pretty-prints the descriptor. *)
val pp : Stdlib.Format.formatter -> t -> unit
