open Base
open Stdio
open Tesserae_kernel
open Tesserae_backend
open Tesserae_tirix

type result = {
  kernel_name : string;
  source : string;
  ptx : string option;
  duration_ms : float;
}

type compile_error =
  | LowerError of Lower.error
  | NvrtcError of string
  | LaunchError of string
  | VerifyError of string list

let pp_error fmt = function
  | LowerError e -> Stdlib.Format.fprintf fmt "LowerError: %a"  Lower.pp_error e
  | NvrtcError s -> Stdlib.Format.fprintf fmt "NvrtcError: %s"  s
  | LaunchError s -> Stdlib.Format.fprintf fmt "LaunchError: %s" s
  | VerifyError errs ->
    List.iter errs ~f:(fun s ->
      Stdlib.Format.fprintf fmt "VerifyError: %s\n" s)

let pp_result fmt r =
  Stdlib.Format.fprintf fmt
    "Compile.result(kernel=%s source_bytes=%d ptx=%s duration=%.2fms)"
    r.kernel_name
    (String.length r.source)
    (Option.value_map r.ptx ~default:"none" ~f:(fun s ->
      Printf.sprintf "%d bytes" (String.length s)))
    r.duration_ms

let time f =
  let t0  = Unix.gettimeofday () in
  let res = f () in
  let dt  = (Unix.gettimeofday () -. t0) *. 1000.0 in
  (res, dt)

let to_tirix (k : Kernel_ast.kernel)
  : (Tirix.tirix, compile_error) Result.t =
  match Lower.lower k with
  | Error e -> Error (LowerError e)
  | Ok (Lower.Pack desc) ->
    let tir = Desc_to_tirix.lower desc in
    match Tirix_verify.verify tir with
    | Error errs -> Error (VerifyError errs)
    | Ok () -> Ok tir

let to_source (k : Kernel_ast.kernel)
  : (result, compile_error) Result.t =
  match to_tirix k with
  | Error e -> Error e
  | Ok tir ->
    let (out, dt) = time (fun () -> Tirix_emit.emit tir) in
    Ok { kernel_name = k.Kernel_ast.name
       ; source = out.Backend_cute.full_source
       ; ptx = None
       ; duration_ms  = dt }


let to_ast (k : Kernel_ast.kernel) : (result, compile_error) Result.t =
  let tir = Ast_to_tirix.lower k in
  match Tirix_verify.verify tir with
  | Error errs -> Error (VerifyError errs)
  | Ok () ->
    let (out, dt) = time (fun () -> Tirix_emit.emit tir) in
    Ok { kernel_name = k.Kernel_ast.name
       ; source = out.Backend_cute.full_source
       ; ptx = None
       ; duration_ms  = dt }



let to_ptx (k : Kernel_ast.kernel) : (result, compile_error) Result.t =
  let run_nvrtc_compilation (r : result) =
    let arch = Nvrtc.arch_string k.Kernel_ast.arch in
    let cutlass_path =
      Option.value
        (Sys.getenv "CUTLASS_PATH")
        ~default:"/tmp/cutlass_latest/include"
    in
    let opts = [
      Printf.sprintf "--include-path=%s" cutlass_path;
      "--std=c++17";
      "-D__LP64__";
      "-D__x86_64__";
      "-default-device";
      "-arch=compute_90a";
      "--include-path=/usr/include";
      "--include-path=/usr/include/x86_64-linux-gnu";
      "--include-path=/usr/local/cuda/include";
      "--include-path=/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl";
      "--include-path=/usr/lib/gcc/x86_64-linux-gnu/11/include";
    ] in
    match Nvrtc.compile_source r.source
      ~name:(r.kernel_name ^ ".cu")
      ~arch
      ~options:opts
      () with
    | Error msg -> Error (NvrtcError msg)
    | Ok ptx    -> Ok { r with ptx = Some ptx }
  in

  match k.Kernel_ast.body with
  | Seq [] -> (
      match to_source k with
      | Error e -> Error e
      | Ok r    -> run_nvrtc_compilation r
    )
  | _ -> (
      match to_ast k with
      | Error e -> Error e
      | Ok r    -> run_nvrtc_compilation r
    )



let write_source (k : Kernel_ast.kernel) (path : string) : unit =
  match to_source k with
  | Ok result -> Out_channel.write_all path ~data:result.source
  | Error e   ->
      let msg = Stdlib.Format.asprintf "%a" pp_error e in
      failwith ("Failed to write source: " ^ msg)

let () = ignore to_ptx
