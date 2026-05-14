val is_available : unit -> bool
val device_info  : unit -> string
val run :
  Kernel_ast.kernel ->
  m:int -> n:int -> k_:int ->
  a:float array -> b:float array ->
  (float array, string) Result.t
val run_exn :
  Kernel_ast.kernel ->
  m:int -> n:int -> k_:int ->
  a:float array -> b:float array ->
  float array
