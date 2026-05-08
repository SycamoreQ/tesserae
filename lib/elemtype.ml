open Base

type float32
type float16
type bfloat16
type int8
type int32

type _ t =
  | Float32  : float32  t
  | Float16  : float16  t
  | Bfloat16 : bfloat16 t
  | Int8 : int8     t
  | Int32 : int32    t

let cpp_name : type a. a t -> string = fun e ->
  match e with
  | Float32 -> "float"
  | Float16 -> "__half"
  | Bfloat16 -> "__nv_bfloat16"
  | Int8 -> "int8_t"
  | Int32 -> "int32_t"

let byte_width : type a. a t -> int = fun e ->
  match e with
  | Float32 -> 4
  | Float16 -> 2
  | Bfloat16 -> 2
  | Int8 -> 1
  | Int32 -> 4

let bits : type a. a t -> int = fun e ->
  match e with
  | Float32 -> 32
  | Float16 -> 16
  | Bfloat16 -> 16
  | Int8 -> 8
  | Int32 -> 32

let is_floating : type a. a t -> bool = function
  | Float32 | Float16 | Bfloat16 -> true
  | _ -> false

let is_integer : type a. a t -> bool = function
  | Int8 | Int32 -> true
| _ -> false

let vec_width : type a. a t -> int = fun e ->
  match e with
  | Float32 -> 128/32
  | Float16 -> 128/16
  | Bfloat16 -> 128/16
  | Int8 -> 128/8
  | Int32 -> 128/32

let pp (fmt : Stdlib.Format.formatter) (e : _ t) : unit =
  Stdlib.Format.fprintf fmt "%s" (cpp_name e)
