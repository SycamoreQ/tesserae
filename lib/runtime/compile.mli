open Tesserae_kernel

(** Compile — the single user-facing entry point for Tesserae.

    The full pipeline:
      Kernel_ast - Lower - Backend_cute -  nvrtc - PTX → Cudajit
**)

(** Compilation result. *)
type result = {
  kernel_name : string;
  source      : string;   (** CuTe C++ source                    *)
  ptx         : string option;  (** PTX — Some after nvrtc**)
  duration_ms : float;    (** wall time of compilation            *)
}

(** Compilation error. *)
type compile_error =
  | LowerError   of Lower.error
  | NvrtcError   of string   (** nvrtc compilation failure — Phase 4 *)
  | LaunchError  of string   (** kernel launch failure    — Phase 4 *)

(** [to_source k] lowers and emits CuTe C++ for kernel [k].
    Returns [Ok result] with ptx=None, or [Error e]. *)
val to_source :
  Kernel_ast.kernel ->
  (result, compile_error) Result.t

(** [to_source_exn k] like [to_source] but raises on error. *)
val to_source_exn : Kernel_ast.kernel -> result

(** [to_ptx k] lowers, emits, and compiles via nvrtc.
    Requires ocaml-cudajit — stub in Phase 3, functional in Phase 4.
    Returns [Ok result] with ptx=Some ptx_string, or [Error e]. *)
val to_ptx :
  Kernel_ast.kernel ->
  (result, compile_error) Result.t

(** [write_source k path] compiles to source and writes to [path].
    e.g. [write_source k "gemm.cuh"] *)
val write_source : Kernel_ast.kernel -> string -> unit

(** [pp_result fmt r] pretty-prints a compilation result. *)
val pp_result : Stdlib.Format.formatter -> result -> unit

(** [pp_error fmt e] pretty-prints a compilation error. *)
val pp_error : Stdlib.Format.formatter -> compile_error -> unit
