open Base
type gpu_ptr_block
type memory_kind = Device | Host

type t = {
  handle : gpu_ptr_block;
  n_elems : int;
  elem_bytes : int;
  kind : memory_kind;
}

external caml_host_alloc : int -> gpu_ptr_block
  = "caml_host_alloc"
external caml_gpu_alloc : int -> gpu_ptr_block
  = "caml_gpu_alloc"
external caml_gpu_free : gpu_ptr_block -> unit
  = "caml_gpu_free"
external caml_host_free : gpu_ptr_block -> unit
  = "caml_host_free"
external caml_gpu_copy_to_device : gpu_ptr_block -> bytes -> int -> unit
  = "caml_gpu_copy_to_device"
external caml_gpu_copy_to_host : bytes -> gpu_ptr_block -> int -> unit
  = "caml_gpu_copy_to_host"
external caml_gpu_memset_zero : gpu_ptr_block -> int -> unit
  = "caml_gpu_memset_zero"
external caml_gpu_ptr : gpu_ptr_block -> nativeint
  = "caml_gpu_ptr"
external caml_gpu_copy_ptr_to_ptr : gpu_ptr_block -> gpu_ptr_block -> int -> unit
  = "caml_gpu_copy_ptr_to_ptr"

let elem_size = 4

let alloc (n : int) : t =
  let p = caml_gpu_alloc (n * elem_size) in
  caml_gpu_memset_zero p (n * elem_size);
  { handle = p; n_elems = n; elem_bytes = elem_size; kind = Device }

let free (t : t) : unit =
  match t.kind with
  | Device -> caml_gpu_free t.handle
  | Host -> caml_host_free t.handle

let size      (t : t) : int = t.n_elems
let byte_size (t : t) : int = t.n_elems * t.elem_bytes

let ptr       (t : t) : nativeint = caml_gpu_ptr t.handle

let float_to_bytes (arr : float array) : bytes =
  let b = Bytes.create (Array.length arr * elem_size) in
  Array.iteri arr ~f:(fun i f ->
    Stdlib.Bytes.set_int32_le b (i * 4) (Int32.bits_of_float f));
  b

let of_host (arr : float array) : t =
  let n = Array.length arr in
  let buf = alloc n in
  let b = float_to_bytes arr in
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

let alloc_host (n : int) : t =
  let p = caml_host_alloc (n * elem_size) in
  { handle = p; n_elems = n; elem_bytes = elem_size; kind = Host }

let copy_buffer ~dst ~src ~bytes =
  caml_gpu_copy_ptr_to_ptr dst.handle src.handle bytes

let alloc_f16 (n : int) : t =
  let p = caml_gpu_alloc (n * 2) in
  caml_gpu_memset_zero p (n * 2);
  { handle = p; n_elems = n; elem_bytes = 2; kind = Device }


let float_to_f16_bits (f : float) : int =
  let bits = Int32.bits_of_float f in
  let sign = Int32.(to_int_exn (shift_right_logical bits 31)) in
  let exp  = Int32.(to_int_exn (shift_right_logical (shift_left bits 1) 24)) in
  let mant = Int32.(to_int_exn (bit_and bits 0x7FFFFF_l)) in
  if exp = 0 && mant = 0 then sign lsl 15
  else if exp = 0xFF then
    (sign lsl 15) lor 0x7C00 lor (if mant <> 0 then 0x200 else 0)
  else
    let new_exp = exp - 127 + 15 in
    if new_exp >= 31 then (sign lsl 15) lor 0x7C00
    else if new_exp <= 0 then
      if new_exp < -10 then sign lsl 15
      else
        let mant = mant lor 0x800000 in
        let shift = 14 - new_exp in
        let rounded = (mant + (1 lsl (shift - 1))) lsr shift in
        if rounded > 0x3FF then (sign lsl 15) lor 0x7C00
        else (sign lsl 15) lor rounded
    else
      let rounded = (mant + 0x1000) lsr 13 in
      if rounded > 0x3FF then (sign lsl 15) lor ((new_exp + 1) lsl 10)
      else (sign lsl 15) lor (new_exp lsl 10) lor rounded

let f16_bits_to_float (bits : int) : float =
  let sign = (bits lsr 15) land 1 in
  let exp  = (bits lsr 10) land 0x1F in
  let mant = bits land 0x3FF in
  let f32_bits =
    if exp = 0 && mant = 0 then Int32.of_int_exn (sign lsl 31)
    else if exp = 0x1F then Int32.of_int_exn ((sign lsl 31) lor 0x7F800000 lor (mant lsl 13))
    else Int32.of_int_exn ((sign lsl 31) lor ((exp - 15 + 127) lsl 23) lor (mant lsl 13))
  in
  Int32.float_of_bits f32_bits

let copy_from_host_f16 (t : t) (arr : float array) : unit =
  let b = Bytes.create (Array.length arr * 2) in
  Array.iteri arr ~f:(fun i f ->
    Stdlib.Bytes.set_uint16_le b (i * 2) (float_to_f16_bits f));
  caml_gpu_copy_to_device t.handle b (Array.length arr * 2)


let to_host_f16 (t : t) : float array =
  let b = Bytes.create (t.n_elems * 2) in
  caml_gpu_copy_to_host b t.handle (t.n_elems * 2);
  Array.init t.n_elems ~f:(fun i ->
    f16_bits_to_float (Stdlib.Bytes.get_uint16_le b (i * 2)))
