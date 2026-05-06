(** Element types supported by CuTe kernels.
    The phantom type ['a] is the OCaml representation type.
    The GADT witness carries the C++ name and byte width. *)

type float32
type float16
type bfloat16
type int8
type int32

(** GADT witness for element types. *)
type _ t =
  | Float32  : float32  t
  | Float16  : float16  t
  | Bfloat16 : bfloat16 t
  | Int8     : int8     t
  | Int32    : int32    t

(** [cpp_name e] returns the C++ type string for this element.
    Float32  → "float"
    Float16  → "__half"
    Bfloat16 → "__nv_bfloat16"
    Int8     → "int8_t"
    Int32    → "int32_t" *)
val cpp_name : _ t -> string

(** [byte_width e] returns the size in bytes.
    Float32  → 4
    Float16  → 2
    Bfloat16 → 2
    Int8     → 1
    Int32    → 4 *)
val byte_width : _ t -> int

(** [bits e] returns the size in bits. *)
val bits : _ t -> int

(** [is_floating e] returns true iff the element type is a floating
    point type (float32, float16, bfloat16). *)
val is_floating : _ t -> bool

(** [is_integer e] returns true iff the element type is an integer
    type (int8, int32). *)
val is_integer : _ t -> bool

(** [vec_width e] returns how many elements fit in a 128-bit vector
    load/store (the width of an LDG.128 / STG.128 on Ampere).
    This is 128 / bits(e). *)
val vec_width : _ t -> int

(** [pp fmt e] pretty-prints the C++ name. *)
val pp : Stdlib.Format.formatter -> _ t -> unit
