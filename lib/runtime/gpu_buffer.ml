open Base
type gpu_ptr_block

type t = {
  handle : gpu_ptr_block;
  n_elems : int;
  elem_bytes : int;
}

external caml_gpu_alloc : int -> gpu_ptr_block
  = "caml_gpu_alloc"
external caml_gpu_free : gpu_ptr_block -> unit
  = "caml_gpu_free"
external caml_gpu_copy_to_device : gpu_ptr_block -> bytes -> int -> unit
  = "caml_gpu_copy_to_device"
external caml_gpu_copy_to_host : bytes -> gpu_ptr_block -> int -> unit
  = "caml_gpu_copy_to_host"
external caml_gpu_memset_zero : gpu_ptr_block -> int -> unit
  = "caml_gpu_memset_zero"
external caml_gpu_ptr : gpu_ptr_block -> nativeint
  = "caml_gpu_ptr"

let elem_size = 4

let alloc (n : int) : t =
  let p = caml_gpu_alloc (n * elem_size) in
  caml_gpu_memset_zero p (n * elem_size);
  { handle = p; n_elems = n; elem_bytes = elem_size }

let free (t : t) : unit =
  caml_gpu_free t.handle

let size      (t : t) : int = t.n_elems
let byte_size (t : t) : int = t.n_elems * t.elem_bytes

let ptr       (t : t) : nativeint = caml_gpu_ptr t.handle

let float_to_bytes (arr : float array) : bytes =
  let b = Bytes.create (Array.length arr * elem_size) in
  Array.iteri arr ~f:(fun i f ->
    Stdlib.Bytes.set_int32_le b (i * 4) (Int32.bits_of_float f));
  b

let of_host (arr : float array) : t =
  let n   = Array.length arr in
  let buf = alloc n in
  let b   = float_to_bytes arr in
  caml_gpu_copy_to_device buf.handle b (n * elem_size);
  buf

let to_host (t : t) : float array =
  let b = Bytes.create (byte_size t) in
  caml_gpu_copy_to_host b t.handle (byte_size t);
  Array.init t.n_elems ~f:(fun i ->
    Int32.float_of_bits (Stdlib.Bytes.get_int32_le b (i * 4)))

let copy_from_host (t : t) (arr : float array) : unit =
  let b = float_to_bytes arr in
  caml_gpu_copy_to_device t.handle b (byte_size t)
