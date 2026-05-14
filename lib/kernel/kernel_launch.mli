type module_
type func


val load_ptx      : string -> module_
val unload        : module_ -> unit
val is_valid      : module_ -> bool
val get_function  : module_ -> string -> func
val func_is_valid : func -> bool
val launch        : func
  -> grid:(int * int * int)
  -> block:(int * int * int)
  -> smem:int
  -> args:nativeint list
  -> unit
val synchronize   : unit -> unit
val device_info   : unit -> string
