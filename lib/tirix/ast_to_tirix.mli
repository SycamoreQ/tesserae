open Base
open Tesserae_kernel

(** Compilation context for tracking current hardware target parameters,
    active tensor metadata bindings, and unique variable generation keys. *)
type ctx = {
  arch : Kernel_ast.arch;
  tensors : (string * Tirix.packed_tensor) list;
  var_counter : int ref;
}

(** [lookup_tensor ctx expr] resolves a high-level frontend tensor expression
    into a backend [Tirix.packed_tensor] description footprint by searching
    the context bindings or building an explicit structural allocation. *)
val lookup_tensor : ctx -> Kernel_ast.tensor_expr -> Tirix.packed_tensor

(** [infer_copy_kind ctx src dst] evaluates the underlying memory space boundaries
    of the source and destination tensors to infer the optimal physical hardware
    copy mechanic (e.g., [TmaLoad], [CpAsync], or [SmemToReg]). *)
val infer_copy_kind : ctx -> Tirix.packed_tensor -> Tirix.packed_tensor -> Tirix.copy_kind

(** [infer_mma_kind ctx] maps the architecture generation parameter within the
    current context to the target hardware Tensor Core instruction set variant. *)
val infer_mma_kind : ctx -> Tirix.mma_kind

(** [lower_mask mask] lowers a multi-dimensional bounding matrix coordinate constraint
    into a single composite Boolean evaluation expression loop. *)
val lower_mask : Kernel_ast.mask -> bool Tirix.expr

(** [lower_barrier barrier] transforms a high-level abstract execution block barrier
    into a targeted physical execution synchronization instruction primitive. *)
val lower_barrier : Kernel_ast.barrier_kind -> Tirix.barrier

(** [lower_pred pred] lowers high-level warp spatial queries and conditional execution
    guards into a Boolean hardware target expression. *)
val lower_pred : Kernel_ast.pred_expr -> bool Tirix.expr

(** [lower kernel] compiles a frontend [Kernel_ast.kernel] configuration description
    into a completely verified, lower-level target structural [Tirix.tirix] IR tree. *)
val lower : Kernel_ast.kernel -> Tirix.tirix
