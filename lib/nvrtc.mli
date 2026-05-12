type program

(** [create_program source name] initializes a new NVRTC program context. *)
val create_program : string -> string -> program

(** [destroy_program p] manually releases CUDA resources. Usually handled by GC. *)
val destroy_program : program -> unit

(** [compile_program p options] compiles the source. Returns Error log on failure. *)
val compile_program : program -> string list -> (unit, string) result

val get_ptx : program -> string
val get_log : program -> string
val is_valid : program -> bool
val arch_string : Kernel_ast.arch -> string

(** High-level wrapper for kernel generation. *)
val compile_source :
  string ->
  name:string ->
  arch:string ->
  ?options:string list ->
  unit -> (string, string) Result.t

val compile_source_default : string -> name:string -> (string, string) result
