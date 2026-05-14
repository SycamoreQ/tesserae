open Base

type t ={
  handle: bytes;
  n_elems: int ;
  elem_bytes: int;
}

external caml_gpu_alloc: int -> bytes = "ocaml_gpu_alloc"
external caml_gpu_free: bytes -> unit = " caml_gpu_free"
external caml_gpu_copy_to_device: bytes -> bytes -> int -> unit = "caml_gpu_copy_to_device"
external caml_gpu_copy_to_host: bytes -> bytes -> int -> unit = "caml_gpu_copy_to_host"
external caml_gpu_ptr: bytes -> nativeint = "caml_gpu_ptr"


let elem_size = 4

let alloc (n : int) : t =
  { handle    = caml_gpu_alloc (n * elem_size)
  ; n_elems   = n
  ; elem_bytes = elem_size }

let free (t : t) : unit =
  caml_gpu_free t.handle

let size (t : t) : int = t.n_elems

let byte_size (t : t) : int = t.n_elems * t.elem_bytes

let ptr (t : t) : nativeint = caml_gpu_ptr t.handle

let of_host (arr : float array) : t =
  let n = Array.length arr in
  let buf = alloc n in
  let b = Bytes.create (n * elem_size) in
  Array.iteri (fun i f ->
    let bits = Int32.bits_of_float f in
    Bytes.set_int32_le b (i * 4) bits) arr;
  caml_gpu_copy_to_device buf.handle b (n * elem_size);
  buf

let to_host (t : t) : float array =
  let b = Bytes.create (byte_size t) in
  caml_gpu_copy_to_host b t.handle (byte_size t);
  Array.init t.n_elems (fun i ->
    Int32.float_of_bits (Bytes.get_int32_le b (i * 4)))

let copy_from_host (t : t) (arr : float array) : unit =
  let b = Bytes.create (byte_size t) in
  Array.iteri (fun i f ->
    let bits = Int32.bits_of_float f in
    Bytes.set_int32_le b (i * 4) bits) arr;
  caml_gpu_copy_to_device t.handle b (byte_size t)
