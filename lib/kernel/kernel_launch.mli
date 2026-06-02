type module_
type func

val load_ptx : string -> module_
val unload : module_ -> unit
val is_valid : module_ -> bool

val get_function : module_ -> string -> func
val func_is_valid : func -> bool

val launch :
  func ->
  grid:(int * int * int) ->
  block:(int * int * int) ->
  smem:int ->
  args:nativeint list ->
  unit

val synchronize : unit -> unit
val device_info : unit -> string

val encode_tensor_map_2d :
  tmap_ptr:nativeint ->
  data_ptr:nativeint ->
  elem_bytes:int ->
  global_rows:int ->
  global_cols:int ->
  tile_rows:int ->
  tile_cols:int ->
  unit

val create_tma_descriptor : nativeint -> int -> int -> int -> int -> nativeint
val free_tma_descriptor   : nativeint -> unit
