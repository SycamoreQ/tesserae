open Tesserae_kernel

type program

let default_flags () = [
  "-I/usr/local/cuda/include"
; "-I/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl"
; "-I/usr/include"
; "-I/usr/lib/gcc/x86_64-linux-gnu/13/include"
; "-I/tmp/cutlass_latest/include"
; "--std=c++17"
; "--device-as-default-execution-space"
]

external create_program : string -> string -> program
  = "caml_nvrtc_create"

external destroy_program : program -> unit
  = "caml_nvrtc_destroy"

external is_valid : program -> bool
  = "caml_nvrtc_is_valid"

external compile_program : program -> string list -> (unit, string) Result.t
  = "caml_nvrtc_compile"

external get_ptx : program -> string
  = "caml_nvrtc_get_ptx"

external get_log : program -> string
  = "caml_nvrtc_get_log"

let arch_string = function
  | Kernel_ast.SM80  -> "sm_80"
  | Kernel_ast.SM90  -> "sm_90"
  | Kernel_ast.SM100 -> "sm_100"

let compile_source source ~name ~arch ?(options=[]) () =
  let prog = create_program source name in
  let arch_flag = Printf.sprintf "--gpu-architecture=%s" arch in
  let all_opts = arch_flag :: (default_flags () @ options) in
  match compile_program prog all_opts with
  | Error msg ->
    destroy_program prog;
    Error msg
  | Ok () ->
    let ptx = get_ptx prog in
    destroy_program prog;
    Ok ptx


let compile_source_default source ~name =
  compile_source source ~name ~arch:"sm_80" ()
