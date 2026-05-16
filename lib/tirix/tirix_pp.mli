(** Tir_pp — pretty printer for the Tir IR.
    Produces human-readable output for debugging, not CUDA C++ emission.
    For CUDA C++ output use Tir_emit. *)

open Tirix

(** [pp_expr e] pretty-prints an expression. *)
val pp_expr : _ expr -> string

(** [pp_packed_expr e] pretty-prints a packed expression. *)
val pp_packed_expr : packed_expr -> string

(** [pp_barrier b] pretty-prints a barrier as a PTX asm string. *)
val pp_barrier : barrier -> string

(** [pp_op op] pretty-prints a primitive operation. *)
val pp_op : op -> string

(** [pp_stmt ?depth s] pretty-prints a statement with indentation. *)
val pp_stmt : ?depth:int -> stmt -> string

(** [pp_helper h] pretty-prints a helper function. *)
val pp_helper : helper_func -> string

(** [pp_tir k] pretty-prints the full kernel IR. *)
val pp_tirix : tir -> string

(** [pp_scalar_ty t] returns the C++ type string for a scalar type. *)
val pp_scalar_ty : _ scalar_ty -> string

(** [pp_packed_scalar s] returns the C++ type string for a packed scalar. *)
val pp_packed_scalar : packed_scalar -> string
