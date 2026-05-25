open Base
open Bigarray

type module_ = {
  handle : nativeint ref;
}

type func = {
  handle : nativeint ref;
  name   : string;
}

external caml_cuinit : unit -> unit
  = "caml_cuinit"
external caml_module_load_ptx : string -> nativeint
  = "caml_module_load_ptx"
external caml_module_unload : nativeint -> unit
  = "caml_module_unload"
external caml_get_function : nativeint -> string -> nativeint
  = "caml_get_function"

external caml_launch_kernel : nativeint
  -> int -> int -> int
  -> int -> int -> int
  -> int
  -> (nativeint, nativeint_elt, c_layout) Array1.t
  -> unit
  = "caml_launch_kernel_bytecode" "caml_launch_kernel"

external caml_device_synchronize  : unit -> unit
  = "caml_device_synchronize"
external caml_device_info         : unit -> string
  = "caml_device_info"

let init_called = ref false

let ensure_init () =
  if not !init_called then begin
    caml_cuinit ();
    init_called := true
  end

let load_ptx (ptx : string) : module_ =
  ensure_init ();
  let h = caml_module_load_ptx ptx in
  { handle = ref h }

let unload (m : module_) : unit =
  if Nativeint.(!(m.handle) <> zero) then begin
    caml_module_unload !(m.handle);
    m.handle := Nativeint.zero
  end

let is_valid (m : module_) : bool =
  Nativeint.(!(m.handle) <> zero)

let get_function (m : module_) (name : string) : func =
  let h = caml_get_function !(m.handle) name in
  { handle = ref h; name }

let func_is_valid (f : func) : bool =
  Nativeint.(!(f.handle) <> zero)

let launch (f : func)
    ~(grid  : int * int * int)
    ~(block : int * int * int)
    ~(smem  : int)
    ~(args  : nativeint list) : unit =
  let gx, gy, gz = grid in
  let bx, by, bz = block in

  (* Pack arguments into a safe, contiguous off-heap Bigarray *)
  let len = List.length args in
  let arr = Array1.create nativeint c_layout len in
  List.iteri args ~f:(fun i v -> Array1.set arr i v);

  caml_launch_kernel !(f.handle)
    gx gy gz bx by bz smem arr

let synchronize () : unit =
  caml_device_synchronize ()

let device_info () : string =
  caml_device_info ()
