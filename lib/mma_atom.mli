(** MMA instruction descriptors for Ampere, Hopper, and Blackwell.

    Each atom encodes:
    - The target architecture (sm80, sm90, sm100)
    - The instruction shape (M, N, K)
    - Element types for A, B, C, D operands
    - Memory major order (row/col major for each operand) *)

(** Architecture tags — phantom types for arch-level dispatch. *)
type sm80
type sm90
type sm100

type _ arch =
  | SM80  : sm80  arch   (** Ampere  — mma.sync          *)
  | SM90  : sm90  arch   (** Hopper  — wgmma.mma_async   *)
  | SM100 : sm100 arch   (** Blackwell — tcgen05.mma/umma *)

(** Major order of a matrix operand. *)
type major =
  | RowMajor
  | ColMajor

(** A single MMA atom descriptor.
    ['arch] is the architecture phantom.
    ['a] ['b] ['c] ['d] are the element type phantoms for each operand. *)
type ('arch, 'a, 'b, 'c, 'd) t = {
  arch    : 'arch arch;
  m       : int;          (** instruction M dimension *)
  n       : int;          (** instruction N dimension *)
  k       : int;          (** instruction K dimension *)
  a_type  : 'a Elemtype.t;
  b_type  : 'b Elemtype.t;
  c_type  : 'c Elemtype.t;
  d_type  : 'd Elemtype.t;
  a_major : major;
  b_major : major;
}

(** Canonical Ampere atoms (mma.sync.aligned) *)

(** SM80 16x8x8 F16 input, F32 accumulate — the base Ampere tensor core atom *)
val sm80_16x8x8_f32f16f16f32 : major -> major ->
  (sm80,
   Elemtype.float16, Elemtype.float16,
   Elemtype.float32, Elemtype.float32) t

(** SM80 16x8x16 F16 input, F32 accumulate *)
val sm80_16x8x16_f32f16f16f32 : major -> major ->
  (sm80,
   Elemtype.float16, Elemtype.float16,
   Elemtype.float32, Elemtype.float32) t

(** SM80 16x8x16 BF16 input, F32 accumulate *)
val sm80_16x8x16_f32bf16bf16f32 : major -> major ->
  (sm80,
   Elemtype.bfloat16, Elemtype.bfloat16,
   Elemtype.float32,  Elemtype.float32) t

(** SM80 16x8x32 INT8 input, INT32 accumulate *)
val sm80_16x8x32_s32s8s8s32 : major -> major ->
  (sm80,
   Elemtype.int8,  Elemtype.int8,
   Elemtype.int32, Elemtype.int32) t

(** Canonical Hopper atoms (wgmma.mma_async) *)

(** SM90 64x64x16 F16 wgmma — base Hopper warpgroup MMA *)
val sm90_64x64x16_f32f16f16f32 : major -> major ->
  (sm90,
   Elemtype.float16, Elemtype.float16,
   Elemtype.float32, Elemtype.float32) t

(** SM90 64x128x16 F16 wgmma *)
val sm90_64x128x16_f32f16f16f32 : major -> major ->
  (sm90,
   Elemtype.float16, Elemtype.float16,
   Elemtype.float32, Elemtype.float32) t

(** SM90 64x64x16 BF16 wgmma *)
val sm90_64x64x16_f32bf16bf16f32 : major -> major ->
  (sm90,
   Elemtype.bfloat16, Elemtype.bfloat16,
   Elemtype.float32,  Elemtype.float32) t

(** Canonical Blackwell atoms (tcgen05.mma / umma) *)

(** SM100 64x64x16 F16 umma — base Blackwell atom *)
val sm100_64x64x16_f32f16f16f32 : major -> major ->
  (sm100,
   Elemtype.float16, Elemtype.float16,
   Elemtype.float32, Elemtype.float32) t

(** SM100 128x128x16 F16 umma — 2SM pair atom *)
val sm100_128x128x16_f32f16f16f32 : major -> major ->
  (sm100,
   Elemtype.float16, Elemtype.float16,
   Elemtype.float32, Elemtype.float32) t

(** SM100 64x64x32 INT8 umma *)
val sm100_64x64x32_s32s8s8s32 : major -> major ->
  (sm100,
   Elemtype.int8,  Elemtype.int8,
   Elemtype.int32, Elemtype.int32) t

(** [shape a] returns (M, N, K) of the atom. *)
val shape : (_, _, _, _, _) t -> int * int * int

(** [thread_count a] returns the number of threads that participate
    in this atom.
    SM80:  32  (one warp)
    SM90:  128 (one warpgroup = 4 warps)
    SM100: 128 (one warpgroup) *)
val thread_count : (_, _, _, _, _) t -> int

(** [is_wgmma a] returns true iff this is a warpgroup MMA (SM90/SM100). *)
val is_wgmma : (_, _, _, _, _) t -> bool

(** [emit_cpp a] emits the CuTe C++ atom type string.
    e.g. "SM80_16x8x16_F32F16F16F32_TN"
         "SM90_64x64x16_F32F16F16F32_TN"
         "SM100_64x64x16_F32F16F16F32_TN" *)
val emit_cpp : (_, _, _, _, _) t -> string

(** [pp fmt a] pretty-prints the atom. *)
val pp : Stdlib.Format.formatter -> (_, _, _, _, _) t -> unit
