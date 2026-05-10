open Base
open Stdio

type result = {
  kernel_name : string;
  source      : string;
  ptx         : string option;
  duration_ms : float;
}

type compile_error =
  | LowerError  of Lower.error
  | NvrtcError  of string
  | LaunchError of string

let pp_error fmt = function
  | LowerError  e -> Stdlib.Format.fprintf fmt "LowerError: %a"  Lower.pp_error e
  | NvrtcError  s -> Stdlib.Format.fprintf fmt "NvrtcError: %s"  s
  | LaunchError s -> Stdlib.Format.fprintf fmt "LaunchError: %s" s

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

let to_source (k : Kernel_ast.kernel)
  : (result, compile_error) Result.t =
  match Lower.lower k with
  | Error e -> Error (LowerError e)
  | Ok (Lower.Pack desc) ->
    let (out, dt) = time (fun () -> Backend_cute.emit desc) in
    Ok { kernel_name = k.Kernel_ast.name
       ; source      = out.Backend_cute.full_source
       ; ptx         = None
       ; duration_ms = dt }

let to_source_exn (k : Kernel_ast.kernel) : result =
  match to_source k with
  | Ok r    -> r
  | Error e ->
    let msg = Stdlib.Format.asprintf "%a" pp_error e in
    failwith msg

let to_ptx (k : Kernel_ast.kernel)
  : (result, compile_error) Result.t =
  (* Phase 4 stub — nvrtc integration goes here *)
  match to_source k with
  | Error e -> Error e
  | Ok r    ->
    (* TODO Phase 4:
       let ptx = Nvrtc.compile r.source ~arch:k.arch in
       Ok { r with ptx = Some ptx } *)
    Ok { r with ptx = Some "(* nvrtc stub — Phase 4 *)" }

let write_source (k : Kernel_ast.kernel) (path : string) : unit =
  let r = to_source_exn k in
  Out_channel.write_all path ~data:r.source

let () = ignore to_ptx
