open Tesserae_runtime

let () =
  let k_total = match Stdlib.Sys.argv with
    | [| _; k |] -> int_of_string k
    | _ -> 4104 in
  match Compile.to_ast (Tesserae_blackwell_kernels.Gemm_warpspec.gemm_blackwell_ws ~k_total ~stages:4 ~multicast:false ()) with
  | Error e ->
    Stdlib.Format.printf "error: %a\n%!" Compile.pp_error e
  | Ok r ->
    print_string r.Compile.source
