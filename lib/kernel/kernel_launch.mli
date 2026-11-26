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
  ?cluster:(int * int * int) ->
  args:nativeint list ->
  unit ->
  unit

val synchronize : unit -> unit
val device_info : unit -> string

(** [max_active_clusters f ~block ~smem ~cluster] returns the maximum number of
    clusters that can be co-resident on the device for the given launch
    configuration, via [cuOccupancyMaxActiveClusters]. *)
val max_active_clusters :
  func ->
  block:(int * int * int) ->
  smem:int ->
  cluster:(int * int * int) ->
  int

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

val create_tma_descriptor_sw128 : nativeint -> int -> int -> int -> int -> nativeint

val create_tma_descriptor_f32_sw128_l2 :
  nativeint -> int -> int -> int -> int -> nativeint
