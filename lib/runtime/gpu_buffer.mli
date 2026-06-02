open Base

type gpu_ptr_block
type memory_kind = Device | Host

type t = {
  handle : gpu_ptr_block;
  n_elems : int;
  elem_bytes : int;
  kind : memory_kind;
}

val alloc : int -> t
val free : t -> unit
val size : t -> int
val byte_size : t -> int
val ptr : t -> nativeint

val float_to_bytes : float array -> bytes
val of_host : float array -> t
val to_host : t -> float array
val copy_from_host : t -> float array -> unit

val alloc_host : int -> t
val copy_buffer : dst:t -> src:t -> bytes:int -> unit

val alloc_f16 : int -> t
val copy_from_host_f16 : t -> float array -> unit
val to_host_f16 : t -> float array

val float_to_f16_bits : float -> int
val f16_bits_to_float : int -> float
