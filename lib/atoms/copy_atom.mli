open Tesserae_core

(** Copy operation descriptors for Ampere, Hopper, and Blackwell.

    Each atom encodes:
    - The copy operation kind (async, TMA, ldmatrix, etc.)
    - Source and destination memory spaces
    - The element type being copied
    - The vector width (how many elements per copy instruction) *)

(** Copy operation kinds. *)
type kind =
  | AsyncCopyGlobal      (** SM80 cp.async gmem → smem, cache global *)
  | AsyncCopyCached      (** SM80 cp.async gmem → smem, cache all   *)
  | TmaLoad              (** SM90 TMA load  gmem → smem              *)
  | TmaStore             (** SM90 TMA store smem → gmem              *)
  | TmaLoadMulticast     (** SM100 TMA multicast gmem → smem (2SM)   *)
  | Ldmatrix             (** SM80 ldmatrix  smem → register          *)
  | LdmatrixTrans        (** SM80 ldmatrix.trans smem → register     *)
  | UniversalCopy        (** Generic element-wise copy               *)

(** A copy atom descriptor.
    ['src] is the source memory space phantom.
    ['dst] is the destination memory space phantom.
    ['elem] is the element type phantom. *)
type ('src, 'dst, 'elem) t = {
  kind       : kind;
  src_space  : 'src Memspace.space;
  dst_space  : 'dst Memspace.space;
  elem_type  : 'elem Elemtype.t;
  vec_width  : int;   (** elements per copy instruction *)
  bulk_bytes : int;   (** bytes transferred per instruction *)
}

(** --- Ampere async copy atoms --- *)

(** [sm80_cp_async_global elem] — cp.async with cache-global hint.
    Transfers [vec_width elem] elements per instruction.
    Source: global, Dest: shared. *)
val sm80_cp_async_global :
  'elem Elemtype.t ->
  (Memspace.global, Memspace.shared, 'elem) t

(** [sm80_cp_async_cached elem] — cp.async with cache-all hint. *)
val sm80_cp_async_cached :
  'elem Elemtype.t ->
  (Memspace.global, Memspace.shared, 'elem) t

(** [sm80_ldmatrix elem] — ldmatrix.x4 loads 4x8 matrix fragments
    from shared memory into registers.
    Always transfers 8 elements of the given type. *)
val sm80_ldmatrix :
  'elem Elemtype.t ->
  (Memspace.shared, Memspace.register, 'elem) t

(** [sm80_ldmatrix_trans elem] — ldmatrix.x4.trans, transposed variant. *)
val sm80_ldmatrix_trans :
  'elem Elemtype.t ->
  (Memspace.shared, Memspace.register, 'elem) t

(** --- Hopper TMA atoms --- *)

(** [sm90_tma_load elem] — TMA load from global to shared.
    Bulk transfer; vec_width = 128 / byte_width. *)
val sm90_tma_load :
  'elem Elemtype.t ->
  (Memspace.global, Memspace.shared, 'elem) t

(** [sm90_tma_store elem] — TMA store from shared to global. *)
val sm90_tma_store :
  'elem Elemtype.t ->
  (Memspace.shared, Memspace.global, 'elem) t

(** --- Blackwell TMA multicast atoms --- *)

(** [sm100_tma_load_multicast elem] — TMA multicast load.
    Loads from global and broadcasts to multiple CTAs in a cluster. *)
val sm100_tma_load_multicast :
  'elem Elemtype.t ->
  (Memspace.global, Memspace.shared, 'elem) t

(** --- Universal fallback --- *)

(** [universal elem] — element-wise copy, any space.
    Used as a fallback when no hardware copy atom is available. *)
val universal :
  'src Memspace.space ->
  'dst Memspace.space ->
  'elem Elemtype.t ->
  ('src, 'dst, 'elem) t

(** --- Queries --- *)

(** [is_async a] returns true iff this is an async copy
    (cp.async or TMA — i.e. does not block the issuing thread). *)
val is_async : (_, _, _) t -> bool

(** [is_tma a] returns true iff this is a TMA operation. *)
val is_tma : (_, _, _) t -> bool

(** [requires_mbar a] returns true iff this copy requires
    an mbarrier for completion signaling (TMA ops do). *)
val requires_mbar : (_, _, _) t -> bool

(** [bulk_bytes a] returns the number of bytes transferred
    per copy instruction. *)
val bulk_bytes_of : (_, _, _) t -> int

(** --- Codegen --- *)

(** [emit_cpp a] emits the CuTe C++ copy atom type string.
    e.g. "SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>"
         "SM90_TMA_LOAD"
         "SM100_TMA_LOAD_MULTICAST"
         "UniversalCopy<float>" *)
val emit_cpp : (_, _, _) t -> string

(** [pp fmt a] pretty-prints the atom. *)
val pp : Stdlib.Format.formatter -> (_, _, _) t -> unit
