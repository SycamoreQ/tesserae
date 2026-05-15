open Tesserae_kernel

(** Backend_cute — emits CuTe C++ from a Kernel_desc.t.

    This is the compiler backend. The user never calls this directly —
    it is driven by Compile.to_ptx via:
      Kernel_ast → Lower → Backend_cute → nvrtc → PTX *)

(** The complete emitted kernel as structured sections. *)
type output = {
  filename       : string;
  includes       : string;
  helpers        : string;  (** make_smem_desc, tma helpers etc. *)
  shared_storage : string;
  producer_body  : string;
  consumer_body  : string;
  epilogue_body  : string;
  kernel_func    : string;
  host_launcher  : string;
  full_source    : string;  (** concatenation of all sections *)
}

(** [emit desc] produces the complete output for a kernel descriptor. *)
val emit : (_, _, _, _, _, _) Kernel_desc.t -> output

(** [emit_includes desc] emits #pragma once + CuTe includes. *)
val emit_includes : (_, _, _, _, _, _) Kernel_desc.t -> string

(** [emit_helpers desc] emits helper functions.
    - make_smem_desc for TMA kernels
    - tma_2d_gmem2smem / tma_2d_gmem2smem_multicast wrappers *)
val emit_helpers : (_, _, _, _, _, _) Kernel_desc.t -> string

(** [emit_shared_storage desc] emits the SharedStorage struct. *)
val emit_shared_storage : (_, _, _, _, _, _) Kernel_desc.t -> string

(** [emit_producer_body desc] emits the producer warp body. *)
val emit_producer_body : (_, _, _, _, _, _) Kernel_desc.t -> string

(** [emit_consumer_body desc] emits the consumer warp body. *)
val emit_consumer_body : (_, _, _, _, _, _) Kernel_desc.t -> string

(** [emit_epilogue_body desc] emits the epilogue warp body. *)
val emit_epilogue_body : (_, _, _, _, _, _) Kernel_desc.t -> string

(** [emit_kernel_func desc] emits the __global__ kernel. *)
val emit_kernel_func : (_, _, _, _, _, _) Kernel_desc.t -> string

(** [emit_host_launcher desc] emits the host launch wrapper. *)
val emit_host_launcher : (_, _, _, _, _, _) Kernel_desc.t -> string

(** [write desc path] writes the full source to [path]. *)
val write : (_, _, _, _, _, _) Kernel_desc.t -> string -> unit
