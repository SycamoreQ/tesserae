open Tesserae_core
open Tesserae_atoms
open Tesserae_pipeline


(** Kernel descriptor — the complete typed description of a GEMM kernel.

    Composes Tile_io + Pipeline + Tile_op + Predicate + problem dims
    into a single object. The kernel emitter takes this as input and
    produces a complete .cuh file.

    Supports three kernel families:
    - Ampere  (SM80): cp.async, mma.sync, register accumulators
    - Hopper  (SM90): TMA, wgmma, register accumulators
    - Blackwell (SM100): TMA multicast, tcgen05.mma, TMEM accumulators *)

(** The kernel family — determines which hardware features are used. *)
type family =
  | Ampere    (** SM80: cp.async + mma.sync                    *)
  | Hopper    (** SM90: TMA + wgmma.mma_async                  *)
  | Blackwell (** SM100: TMA multicast + tcgen05.mma + TMEM    *)

(** A complete GEMM kernel descriptor. *)
type ('arch, 'a, 'b, 'c, 'd, 'elem) t = {
  family     : family;
  name       : string;          (** kernel function name              *)
  bm         : int;             (** CTA tile M dimension              *)
  bn         : int;             (** CTA tile N dimension              *)
  bk         : int;             (** CTA tile K dimension              *)
  tile_io    : 'elem Tile_io.t;
  pipeline   : Pipeline.t;
  tile_op    : ('arch, 'a, 'b, 'c, 'd) Tile_op.t;
  pred_a     : Predicate.t;     (** predicate for A tile              *)
  pred_b     : Predicate.t;     (** predicate for B tile              *)
  pred_c     : Predicate.t;     (** predicate for C tile              *)
  cluster    : Cluster.t;
  sm_count   : int;             (** number of SMs on target GPU       *)
}

(** [make_ampere ~name ~bm ~bn ~bk ~elem ~m ~n ~k] constructs an Ampere
    kernel descriptor with sensible defaults:
    - cp.async copy atoms
    - mma.sync with SM80_16x8x16 atom
    - register accumulators
    - single SM cluster
    - pipeline depth = 4
    - Swizzle<3,4,3> *)
val make_ampere :
  name:string ->
  bm:int -> bn:int -> bk:int ->
  elem:'elem Elemtype.t ->
  m:int -> n:int -> k:int ->
  (Mma_atom.sm80,
    Elemtype.float16, Elemtype.float16,
    Elemtype.float32, Elemtype.float32,
    'elem) t

(** [make_hopper ~name ~bm ~bn ~bk ~elem ~m ~n ~k] constructs a Hopper
    kernel descriptor:
    - TMA load
    - wgmma SM90_64x64x16 atom
    - register accumulators
    - single SM cluster
    - pipeline depth = 4
    - Swizzle<3,4,3> *)
val make_hopper :
  name:string ->
  bm:int -> bn:int -> bk:int ->
  elem:'elem Elemtype.t ->
  m:int -> n:int -> k:int ->
  (Mma_atom.sm90,
    Elemtype.bfloat16, Elemtype.bfloat16,
    Elemtype.float32, Elemtype.float32,
    'elem) t

(** [make_blackwell ~name ~bm ~bn ~bk ~elem ~m ~n ~k] constructs a
    Blackwell kernel descriptor:
    - TMA multicast
    - tcgen05.mma SM100_128x128x16 atom
    - TMEM double-buffered accumulators
    - 2SM cluster
    - pipeline depth = 4
    - Swizzle<3,4,3>
    - sm_count = 148 (B200) *)
val make_blackwell :
  name:string ->
  bm:int -> bn:int -> bk:int ->
  elem:'elem Elemtype.t ->
  m:int -> n:int -> k:int ->
  (Mma_atom.sm100,
    Elemtype.float16, Elemtype.float16,
    Elemtype.float32, Elemtype.float32,
    'elem) t

(** [validate t] checks that the descriptor is internally consistent.
    Returns [Ok ()] or [Error msg]. Checks:
    - bm, bn, bk are all positive and power-of-2
    - bm divisible by atom M, bn by atom N, bk by atom K
    - pipeline smem fits within sm_count * 227KB
    - Blackwell requires 2SM cluster *)
val validate :
  (_, _, _, _, _, _) t -> (unit, string) Result.t

(** [arithmetic_intensity t] computes the arithmetic intensity
    of one CTA tile = 2*BM*BN*BK / ((BM+BN)*BK*byte_width). *)
val arithmetic_intensity : (_, _, _, _, _, _) t -> float

(** [smem_bytes t] returns total shared memory bytes needed per CTA. *)
val smem_bytes : (_, _, _, _, _, _) t -> int

(** [num_warps t] returns total warps per CTA. *)
val num_warps : (_, _, _, _, _, _) t -> int

(** [emit_kernel_params t] emits the C++ kernel parameter list.
    e.g. "const __half* A, const __half* B, float* C,
          int M, int N, int K,
          const __grid_constant__ CUtensorMap A_tmap,
          const __grid_constant__ CUtensorMap B_tmap" *)
val emit_kernel_params : (_, _, _, _, _, _) t -> string

(** [emit_launch_config t m n] emits the host-side launch config.
    grid dims, block dims, smem size. *)
val emit_launch_config : (_, _, _, _, _, _) t -> int -> int -> string

(** [pp fmt t] pretty-prints the kernel descriptor. *)
val pp : Stdlib.Format.formatter -> (_, _, _, _, _, _) t -> unit
