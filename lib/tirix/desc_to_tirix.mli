open Base
open Tesserae_core
open Tesserae_kernel
open Tirix

(** [lower desc] translates a high-level GPU kernel descriptor into the
    Tirix IR representations, orchestrating memory layout calculations,
    hardware synchronization barriers, and thread-group execution paths. *)
val lower : (_, _, _, _, _, _) Kernel_desc.t -> tirix

(** {2 Internal AST Component Builders}
    These builders are exposed for unit-testing, structural validation,
    and profiling hooks within pipeline verification cycles. *)

val fresh_id : unit -> int

val mk_var : string -> 'a scalar_ty -> ?mut:bool -> unit -> var

val i32 : int -> int32 expr

val elem_type_of : (_, _, _, _, _, 'f) Kernel_desc.t -> 'f Elemtype.t

val is_tma : (_, _, _, _, _, _) Kernel_desc.t -> bool

val is_blackwell : (_, _, _, _, _, _) Kernel_desc.t -> bool

val bn_smem : (_, _, _, _, _, _) Kernel_desc.t -> int

val flat_layout : unit -> Layout.t

val packed_atom_of_desc: (_, _, _, _, _, _) Kernel_desc.t -> Tirix.packed_mma_atom

val make_global_tensor : string -> 'a Elemtype.t -> packed_tensor

val construct_smem_tensors :
  (_, _, _, _, _, _) Kernel_desc.t -> (string * packed_tensor) list

val construct_vars : (_, _, _, _, _, _) Kernel_desc.t -> var list

val construct_params : (_, _, _, _, _, _) Kernel_desc.t -> param list

val construct_helpers : (_, _, _, _, _, _) Kernel_desc.t -> helper_func list

val construct_producer_body :
  (_, _, _, _, _, _) Kernel_desc.t ->
  var list ->
  (string * packed_tensor) list ->
  stmt

val construct_consumer_body :
  (_, _, _, _, _, _) Kernel_desc.t ->
  var list ->
  (string * packed_tensor) list ->
  stmt

val construct_epilogue_body :
  (_, _, _, _, _, _) Kernel_desc.t -> var list -> stmt

val construct_mbar_init :
  (_, _, _, _, _, _) Kernel_desc.t -> var list -> stmt list

val tmem_alloc_op :
  (_, _, _, _, _, _) Kernel_desc.t -> var list -> stmt option

val tmem_dealloc_op :
  (_, _, _, _, _, _) Kernel_desc.t -> var list -> stmt option
