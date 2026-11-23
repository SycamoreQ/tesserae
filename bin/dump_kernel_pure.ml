(* Dump the generated CUDA source for a kernel without needing NVRTC/CUDA.
   Usage: dump_kernel_pure [k_total] [stages] *)
let () =
  let k_total = match Stdlib.Sys.argv with
    | [| _; k; _ |] -> int_of_string k
    | _ -> 4096 in
  let stages = match Stdlib.Sys.argv with
    | [| _; _; s |] -> int_of_string s
    | _ -> 3 in
  let kern = Tesserae_blackwell_kernels.Gemm_warpspec.gemm_blackwell_ws
               ~k_total ~stages ~multicast:false () in
  let tir = Tesserae_tirix.Ast_to_tirix.lower kern in
  (match Tesserae_tirix.Tirix_verify.verify tir with
   | Error errs ->
     List.iter (fun s -> prerr_endline ("verify: " ^ s)) errs
   | Ok () -> ());
  ignore (Tesserae_backend.Tirix_emit.emit tir)
