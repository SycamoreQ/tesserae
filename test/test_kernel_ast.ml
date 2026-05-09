open Tesserae

let gemm_body () =
  let a    = Kernel_ast.arg "A" Kernel_ast.F16  Kernel_ast.Global in
  let b    = Kernel_ast.arg "B" Kernel_ast.F16  Kernel_ast.Global in
  let c    = Kernel_ast.arg "C" Kernel_ast.F32  Kernel_ast.Global in
  let sa   = Kernel_ast.smem "smem_A" Kernel_ast.F16 128 64 in
  let sb   = Kernel_ast.smem "smem_B" Kernel_ast.F16 256 64 in
  Kernel_ast.Seq [
    Kernel_ast.warp_dispatch [
      ( Kernel_ast.WarpIs 0,
        [ Kernel_ast.pipeline ~stages:4 ~k:"k_tiles"
            [ Kernel_ast.load ~src:a ~dst:sa ()
            ; Kernel_ast.load ~src:b ~dst:sb ()
            ; Kernel_ast.Barrier (Kernel_ast.MbarFull "full_mbar") ] ] );
      ( Kernel_ast.WarpIs 1,
        [ Kernel_ast.pipeline ~stages:4 ~k:"k_tiles"
            [ Kernel_ast.Barrier (Kernel_ast.MbarFull "full_mbar")
            ; Kernel_ast.mma sa sb c
            ; Kernel_ast.Barrier (Kernel_ast.MbarEmpty "empty_mbar") ] ] );
      ( Kernel_ast.WarpIn [2; 3; 4],
        [ Kernel_ast.store ~src:c ~dst:c () ] )
    ]
  ]

let ampere_gemm () =
  Kernel_ast.make
    ~name:"gemm_f16"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(gemm_body ())

let blackwell_gemm () =
  Kernel_ast.make
    ~name:"gemm_bf16_blackwell"
    ~arch:Kernel_ast.SM100
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 256; k = 64 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("B", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32,  Kernel_ast.Global) ]
    ~body:(gemm_body ())

(* ------------------------------------------------------------------ *)
(* make                                                                *)
(* ------------------------------------------------------------------ *)

let test_make_name () =
  Alcotest.(check string) "name" "gemm_f16"
    (ampere_gemm ()).Kernel_ast.name

let test_make_arch () =
  Alcotest.(check bool) "sm80" true
    (match (ampere_gemm ()).Kernel_ast.arch with
     | Kernel_ast.SM80 -> true | _ -> false)

let test_make_tile () =
  let t = (ampere_gemm ()).Kernel_ast.tile in
  Alcotest.(check int) "m" 128 t.Kernel_ast.m;
  Alcotest.(check int) "n" 128 t.Kernel_ast.n;
  Alcotest.(check int) "k" 32  t.Kernel_ast.k

let test_make_stages () =
  Alcotest.(check int) "stages" 4
    (ampere_gemm ()).Kernel_ast.stages

(* ------------------------------------------------------------------ *)
(* tensor_expr constructors                                            *)
(* ------------------------------------------------------------------ *)

let test_arg () =
  let a = Kernel_ast.arg "A" Kernel_ast.F16 Kernel_ast.Global in
  Alcotest.(check bool) "is arg" true
    (match a with Kernel_ast.Arg ("A", _, _) -> true | _ -> false)

let test_smem () =
  let s = Kernel_ast.smem "smem_A" Kernel_ast.F16 128 64 in
  Alcotest.(check bool) "is smem" true
    (match s with Kernel_ast.Smem ("smem_A", _, _) -> true | _ -> false)

(* ------------------------------------------------------------------ *)
(* stmt constructors                                                   *)
(* ------------------------------------------------------------------ *)

let test_load_stmt () =
  let a  = Kernel_ast.arg "A" Kernel_ast.F16 Kernel_ast.Global in
  let sa = Kernel_ast.smem "smem_A" Kernel_ast.F16 128 64 in
  let s  = Kernel_ast.load ~src:a ~dst:sa () in
  Alcotest.(check bool) "is load" true
    (match s with Kernel_ast.Load _ -> true | _ -> false)

let test_store_stmt () =
  let c = Kernel_ast.arg "C" Kernel_ast.F32 Kernel_ast.Global in
  let s = Kernel_ast.store ~src:c ~dst:c () in
  Alcotest.(check bool) "is store" true
    (match s with Kernel_ast.Store _ -> true | _ -> false)

let test_mma_stmt () =
  let a = Kernel_ast.arg "A" Kernel_ast.F16 Kernel_ast.Global in
  let b = Kernel_ast.arg "B" Kernel_ast.F16 Kernel_ast.Global in
  let c = Kernel_ast.arg "C" Kernel_ast.F32 Kernel_ast.Global in
  let s = Kernel_ast.mma a b c in
  Alcotest.(check bool) "is mma" true
    (match s with Kernel_ast.Mma _ -> true | _ -> false)

let test_pipeline_stmt () =
  let s = Kernel_ast.pipeline ~stages:4 ~k:"k_tiles"
    [ Kernel_ast.syncthreads () ] in
  Alcotest.(check bool) "is pipeline" true
    (match s with Kernel_ast.Pipeline _ -> true | _ -> false)

let test_pipeline_stages () =
  let s = Kernel_ast.pipeline ~stages:4 ~k:"k_tiles" [] in
  Alcotest.(check bool) "stages=4" true
    (match s with
     | Kernel_ast.Pipeline (pd, _) -> pd.Kernel_ast.stages = 4
     | _ -> false)

let test_syncthreads () =
  let s = Kernel_ast.syncthreads () in
  Alcotest.(check bool) "sync" true
    (match s with
     | Kernel_ast.Barrier Kernel_ast.ThreadSync -> true
     | _ -> false)

(* ------------------------------------------------------------------ *)
(* warp_dispatch                                                       *)
(* ------------------------------------------------------------------ *)

let test_warp_dispatch_seq () =
  let s = Kernel_ast.warp_dispatch
    [ (Kernel_ast.WarpIs 0, [Kernel_ast.syncthreads ()])
    ; (Kernel_ast.WarpIs 1, [Kernel_ast.syncthreads ()]) ] in
  Alcotest.(check bool) "is seq" true
    (match s with Kernel_ast.Seq _ -> true | _ -> false)

let test_warp_dispatch_length () =
  let s = Kernel_ast.warp_dispatch
    [ (Kernel_ast.WarpIs 0, [])
    ; (Kernel_ast.WarpIs 1, [])
    ; (Kernel_ast.WarpIn [2;3;4], []) ] in
  Alcotest.(check bool) "3 branches" true
    (match s with Kernel_ast.Seq xs -> List.length xs = 3 | _ -> false)

(* ------------------------------------------------------------------ *)
(* args                                                                *)
(* ------------------------------------------------------------------ *)

let test_args_count () =
  Alcotest.(check int) "3 args" 3
    (List.length (ampere_gemm ()).Kernel_ast.args)

let test_args_names () =
  let names = List.map
    (fun arg -> let (n, _, _) = arg in n)
    (ampere_gemm ()).Kernel_ast.args in
  Alcotest.(check bool) "has A" true (List.mem "A" names);
  Alcotest.(check bool) "has B" true (List.mem "B" names);
  Alcotest.(check bool) "has C" true (List.mem "C" names)

(* ------------------------------------------------------------------ *)
(* pp                                                                  *)
(* ------------------------------------------------------------------ *)

let test_pp_contains_name () =
  let s = Stdlib.Format.asprintf "%a" Kernel_ast.pp (ampere_gemm ()) in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has name" true (contains "gemm_f16" s);
  Alcotest.(check bool) "has SM80" true (contains "SM80" s)

let test_pp_blackwell () =
  let s = Stdlib.Format.asprintf "%a" Kernel_ast.pp (blackwell_gemm ()) in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has SM100"   true (contains "SM100" s);
  Alcotest.(check bool) "has 128x256" true (contains "128x256" s)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Kernel_ast" [
    "make",     [ Alcotest.test_case "name"   `Quick test_make_name
                ; Alcotest.test_case "arch"   `Quick test_make_arch
                ; Alcotest.test_case "tile"   `Quick test_make_tile
                ; Alcotest.test_case "stages" `Quick test_make_stages ];
    "tensor",   [ Alcotest.test_case "arg"    `Quick test_arg
                ; Alcotest.test_case "smem"   `Quick test_smem ];
    "stmt",     [ Alcotest.test_case "load"   `Quick test_load_stmt
                ; Alcotest.test_case "store"  `Quick test_store_stmt
                ; Alcotest.test_case "mma"    `Quick test_mma_stmt
                ; Alcotest.test_case "pipe"   `Quick test_pipeline_stmt
                ; Alcotest.test_case "stages" `Quick test_pipeline_stages
                ; Alcotest.test_case "sync"   `Quick test_syncthreads ];
    "dispatch", [ Alcotest.test_case "seq"    `Quick test_warp_dispatch_seq
                ; Alcotest.test_case "len"    `Quick test_warp_dispatch_length ];
    "args",     [ Alcotest.test_case "count"  `Quick test_args_count
                ; Alcotest.test_case "names"  `Quick test_args_names ];
    "pp",       [ Alcotest.test_case "name"   `Quick test_pp_contains_name
                ; Alcotest.test_case "blkwll" `Quick test_pp_blackwell ];
  ]
