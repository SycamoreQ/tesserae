open Tesserae_tirix
open Tirix
(**Complete replacement to backend_cute. Backend_cute only returns hardcoded
strings , but tirix_emit will iterate through the tirix IR tree and return the string**)

val tirix_is_tma : tirix -> bool
(** [tirix_is_tma k] returns [true] if any parameter inside the kernel
    requires a Tensor Memory Accelerator (TMA) descriptor layout. *)

(** {1 Type and Scalar Emission} *)

val emit_scalar_ty : 'a scalar_ty -> string
(** [emit_scalar_ty ty] maps a scalar type witness to its C++/CUDA primitive string representation. *)

val emit_packed_scalar : packed_scalar -> string
(** [emit_packed_scalar ps] unwraps and emits an existential scalar type block. *)

(** {1 Expression Emission} *)

val emit_arith_op : Arith.t -> string

val emit_cmp_op : Cmp.t -> string

val emit_logic_op : Logic.t -> string

val emit_bitwise_op : Bitwise.t -> string

val emit_unop : Unop.t -> string

val emit_expr : 'a expr -> string
(** [emit_expr e] recursively compiles a typed Tirix expression into a target C++ mathematical or hardware intrinsic expression. *)

val emit_packed_expr : packed_expr -> string
(** [emit_packed_expr pe] unwraps and emits an existential expression node. *)

(** {1 GPU Hardware Primitive Emission} *)

val emit_barrier : barrier -> string
(** [emit_barrier b] generates raw inline PTX or CUDA cooperative group synchronization strings. *)

val emit_copy : tirix -> copy -> string
(** [emit_copy c] generates data movement primitives (e.g., [cp.async], TMA descriptors, or CuTe copies). *)

val emit_mma : mma_desc -> string
(** [emit_mma m] emits hardware matrix multiply-accumulate calls (Ampere [cute::gemm], Hopper [wgmma], or Blackwell [tcgen05]). *)

val emit_op : tirix -> op -> string
(** [emit_op op] translates primitive operations (like memory allocations and descriptor initializations) to hardware-level statements. *)

val emit_stmt : tirix -> ?depth:int -> ?stage_depth:int -> stmt -> string
(** [emit_stmt ~depth s] compiles an individual structured statement, applying the specified indentation level. *)

val emit_stmts : tirix -> ?depth:int -> ?stage_depth:int -> stmt list -> string
(** [emit_stmts ~depth ss] builds a clean newline-delimited, indented code block from a list of statements. *)

(** {1 Kernel Component Generation} *)

val emit_shared_storage : tirix -> string
(** [emit_shared_storage k] constructs the overarching global [SharedStorage] structure holding tensors, asynchronous mbarriers, and tmem addresses. *)

val emit_helper : tirix -> helper_func -> string
(** [emit_helper h] generates fully typed inline device utilities ([__device__ __forceinline__]). *)

val emit_params : tirix -> string
(** [emit_params k] compiles the argument vector signatures required for launch grids (handling grid constants and raw pointers). *)

val emit_kernel_func : tirix -> string
(** [emit_kernel_func k] emits the entrypoint block mapping roles ([__global__]) with explicit warp group state dispatch. *)

val emit_host_launcher : tirix -> string
(** [emit_host_launcher k] constructs the standard host runtime configuration launcher wrapper (supporting Blackwell dynamic extended launches). *)

val emit_includes : tirix -> string
(** [emit_includes k] returns header guards and targeted matrix-architecture header libraries ([cute/tensor.hpp], [cuda_bf16.h], etc.). *)

val emit : tirix -> Backend_cute.output
(** [emit k] executes the complete backend transformation sequence on the Tirix IR kernel tree, packing separate header string components into a unified generation record. *)
