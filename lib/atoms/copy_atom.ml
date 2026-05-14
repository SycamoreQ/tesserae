open Base
open Tesserae_core

type kind =
  | AsyncCopyGlobal
  | AsyncCopyCached
  | TmaLoad
  | TmaStore
  | TmaLoadMulticast
  | Ldmatrix
  | LdmatrixTrans
  | UniversalCopy

type ('src, 'dst, 'elem) t = {
  kind : kind;
  src_space  : 'src Memspace.space;
  dst_space : 'dst Memspace.space;
  elem_type : 'elem Elemtype.t;
  vec_width : int;
  bulk_bytes : int;
}

let sm80_cp_async_global (elem : 'elem Elemtype.t)
  : (Memspace.global, Memspace.shared, 'elem) t =
  let bulk = 16 in
  { kind = AsyncCopyGlobal
  ; src_space  = Memspace.Global
  ; dst_space = Memspace.Shared
  ; elem_type  = elem
  ; bulk_bytes = bulk
  ; vec_width = bulk / Elemtype.byte_width elem }

let sm80_cp_async_cached (elem : 'elem Elemtype.t)
  : (Memspace.global, Memspace.shared, 'elem) t =
  let bulk = 16 in
  { kind = AsyncCopyCached
  ; src_space = Memspace.Global
  ; dst_space = Memspace.Shared
  ; elem_type = elem
  ; bulk_bytes = bulk
  ; vec_width = bulk / Elemtype.byte_width elem }

let sm80_ldmatrix (elem : 'elem Elemtype.t)
  : (Memspace.shared, Memspace.register, 'elem) t =
  { kind = Ldmatrix
  ; src_space = Memspace.Shared
  ; dst_space = Memspace.Register
  ; elem_type  = elem
  ; bulk_bytes = 32
  ; vec_width = 8 }

let sm80_ldmatrix_trans (elem : 'elem Elemtype.t)
  : (Memspace.shared, Memspace.register, 'elem) t =
  { kind  = LdmatrixTrans
  ; src_space  = Memspace.Shared
  ; dst_space = Memspace.Register
  ; elem_type  = elem
  ; bulk_bytes = 32
  ; vec_width  = 8 }

let sm90_tma_load (elem : 'elem Elemtype.t)
  : (Memspace.global, Memspace.shared, 'elem) t =
  let bulk = 128 in
  { kind = TmaLoad
  ; src_space  = Memspace.Global
  ; dst_space = Memspace.Shared
  ; elem_type = elem
  ; bulk_bytes = bulk
  ; vec_width = bulk / Elemtype.byte_width elem }

let sm90_tma_store (elem : 'elem Elemtype.t)
  : (Memspace.shared, Memspace.global, 'elem) t =
  let bulk = 128 in
  { kind  = TmaStore
  ; src_space  = Memspace.Shared
  ; dst_space  = Memspace.Global
  ; elem_type  = elem
  ; bulk_bytes = bulk
  ; vec_width  = bulk / Elemtype.byte_width elem }

let sm100_tma_load_multicast (elem : 'elem Elemtype.t)
  : (Memspace.global, Memspace.shared, 'elem) t =
  let bulk = 128 in
  { kind = TmaLoadMulticast
  ; src_space  = Memspace.Global
  ; dst_space  = Memspace.Shared
  ; elem_type  = elem
  ; bulk_bytes = bulk
  ; vec_width  = bulk / Elemtype.byte_width elem }

let universal
    (src_space : 'src Memspace.space)
    (dst_space : 'dst Memspace.space)
    (elem : 'elem Elemtype.t)
  : ('src, 'dst, 'elem) t =
  let bw = Elemtype.byte_width elem in
  { kind       = UniversalCopy
  ; src_space
  ; dst_space
  ; elem_type  = elem
  ; bulk_bytes = bw
  ; vec_width  = 1 }

let is_async (a : (_, _, _) t) : bool =
  match a.kind with
  | AsyncCopyGlobal | AsyncCopyCached
  | TmaLoad | TmaStore | TmaLoadMulticast -> true
  | Ldmatrix | LdmatrixTrans | UniversalCopy -> false

let is_tma (a : (_, _, _) t) : bool =
  match a.kind with
  | TmaLoad | TmaStore | TmaLoadMulticast -> true
  | _ -> false

let requires_mbar (a : (_, _, _) t) : bool =
  is_tma a

let bulk_bytes_of (a : (_, _, _) t) : int =
  a.bulk_bytes

let emit_cpp (a : (_, _, _) t) : string =
  match a.kind with
  | AsyncCopyGlobal -> "SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>"
  | AsyncCopyCached -> "SM80_CP_ASYNC_CACHEALL<cute::uint128_t>"
  | TmaLoad -> "SM90_TMA_LOAD"
  | TmaStore -> "SM90_TMA_STORE"
  | TmaLoadMulticast -> "SM100_TMA_LOAD_MULTICAST"
  | Ldmatrix -> "SM80_U32x4_LDSM_N"
  | LdmatrixTrans -> "SM80_U16x8_LDSM_T"
  | UniversalCopy ->
    Printf.sprintf "UniversalCopy<%s>" (Elemtype.cpp_name a.elem_type)

let pp (fmt : Stdlib.Format.formatter) (a : (_, _, _) t) : unit =
  Stdlib.Format.fprintf fmt "%s" (emit_cpp a)
