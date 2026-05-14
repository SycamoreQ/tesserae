(** Typed device memory buffer for float32 elements. *)
type t

(** [alloc n] allocates n float32 elements on device, zeroed. *)
val alloc : int -> t

(** [free t] frees device memory. Safe to call multiple times. *)
val free : t -> unit

(** [size t] returns number of elements. *)
val size : t -> int

(** [byte_size t] returns size in bytes. *)
val byte_size : t -> int

(** [ptr t] returns raw device pointer as nativeint. *)
val ptr : t -> nativeint

(** [of_host arr] copies host float array to device. *)
val of_host : float array -> t

(** [to_host t] copies device buffer to host float array. *)
val to_host : t -> float array

(** [copy_from_host t arr] copies host array into existing buffer. *)
val copy_from_host : t -> float array -> unit
