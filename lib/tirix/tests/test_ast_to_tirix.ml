open Base
open Tesserae_kernel
open Tesserae_tirix
open Tirix
open Tesserae_pipeline
open Tesserae_backend

let contains sub str =
  let n = String.length sub and m = String.length str in
  let found = ref false in
  for i = 0 to m - n do
    if String.equal (String.sub str ~pos:i ~len:n) sub then found := true
  done; !found

let ampere_empty () =
  Kernel_ast.make
    ~name:"gemm_ampere"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
~args:[ Kernel_ast.in_arg "A" Kernel_ast.F16
      ; Kernel_ast.in_arg "B" Kernel_ast.F16
      ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Seq [])

let ampere_with_load () =
  Kernel_ast.make
    ~name:"gemm_ampere_load"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
~args:[ Kernel_ast.in_arg "A" Kernel_ast.F16
      ; Kernel_ast.in_arg "B" Kernel_ast.F16
      ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Load (
      Kernel_ast.Arg ("A", Kernel_ast.F16, Kernel_ast.Global),
      Kernel_ast.Smem ("smem_A", Kernel_ast.F16, { Kernel_ast.m=128; n=0; k=32 }),
      None))

let ampere_with_store () =
  Kernel_ast.make
    ~name:"gemm_ampere_store"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
~args:[ Kernel_ast.in_arg "A" Kernel_ast.F16
      ; Kernel_ast.in_arg "B" Kernel_ast.F16
      ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Store (
      Kernel_ast.Smem ("smem_C", Kernel_ast.F32, { Kernel_ast.m=128; n=128; k=0 }),
      Kernel_ast.Arg  ("C",      Kernel_ast.F32, Kernel_ast.Global),
      None))

let ampere_with_mma () =
  Kernel_ast.make
    ~name:"gemm_ampere_mma"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
~args:[ Kernel_ast.in_arg "A" Kernel_ast.F16
      ; Kernel_ast.in_arg "B" Kernel_ast.F16
      ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Mma (
      Kernel_ast.Smem ("smem_A", Kernel_ast.F16, { Kernel_ast.m=128; n=0; k=32 }),
      Kernel_ast.Smem ("smem_B", Kernel_ast.F16, { Kernel_ast.m=0;   n=128; k=32 }),
      Kernel_ast.Arg  ("acc",    Kernel_ast.F32, Kernel_ast.Register)))

let hopper_with_mma () =
  Kernel_ast.make
    ~name:"gemm_hopper_mma"
    ~arch:Kernel_ast.SM90a
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 64 }
    ~stages:4
    ~args:[ Kernel_ast.in_arg "A" Kernel_ast.BF16
          ; Kernel_ast.in_arg "B" Kernel_ast.BF16
          ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Mma (
      Kernel_ast.Smem ("smem_A", Kernel_ast.BF16, { Kernel_ast.m=128; n=0; k=64 }),
      Kernel_ast.Smem ("smem_B", Kernel_ast.BF16, { Kernel_ast.m=0; n=128; k=64 }),
      Kernel_ast.Arg  ("acc",    Kernel_ast.F32, Kernel_ast.Register)))

let blackwell_with_mma () =
  Kernel_ast.make
    ~name:"gemm_blackwell_mma"
    ~arch:Kernel_ast.SM100a
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 256; k = 64 }
    ~stages:4
    ~args:[ Kernel_ast.in_arg "A" Kernel_ast.BF16
          ; Kernel_ast.in_arg "B" Kernel_ast.BF16
          ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Mma (
      Kernel_ast.Smem ("smem_A", Kernel_ast.BF16, { Kernel_ast.m=128; n=0; k=64 }),
      Kernel_ast.Smem ("smem_B", Kernel_ast.BF16, { Kernel_ast.m=0; n=256; k=64 }),
      Kernel_ast.Arg  ("acc",    Kernel_ast.F32, Kernel_ast.Register)))

let ampere_with_for () =
  Kernel_ast.make
    ~name:"gemm_for"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[(Kernel_ast.in_arg "A" Kernel_ast.F16)]
    ~body:(Kernel_ast.For ("k", 0, 8, [Kernel_ast.Seq []]))

let ampere_with_pipeline () =
  Kernel_ast.make
    ~name:"gemm_pipeline"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[(Kernel_ast.in_arg "A" Kernel_ast.F16)]
    ~body:(Kernel_ast.Pipeline (
      { Kernel_ast.stages = 4; k_iters = "K" },
      [ Kernel_ast.Seq [] ]))

let ampere_with_barrier_thread () =
  Kernel_ast.make
    ~name:"gemm_barrier"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[]
    ~body:(Kernel_ast.Barrier Kernel_ast.ThreadSync)

let ampere_with_barrier_cluster () =
  Kernel_ast.make
    ~name:"gemm_cluster_barrier"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[]
    ~body:(Kernel_ast.Barrier Kernel_ast.ClusterSync)

let ampere_with_if_warp () =
  Kernel_ast.make
    ~name:"gemm_if"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[]
    ~body:(Kernel_ast.If (
      Kernel_ast.WarpIs 0,
      [ Kernel_ast.Seq [] ],
      []))

let ampere_with_seq () =
  Kernel_ast.make
    ~name:"gemm_seq"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[]
    ~body:(Kernel_ast.Seq [
      Kernel_ast.Barrier Kernel_ast.ThreadSync
    ; Kernel_ast.Barrier Kernel_ast.ThreadSync
    ])

let ampere_with_warp_dispatch () =
  Kernel_ast.make
    ~name:"gemm_warp"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[]
    ~body:(Kernel_ast.warp_dispatch [
      (Kernel_ast.WarpIs 0, [ Kernel_ast.Barrier Kernel_ast.ThreadSync ])
    ; (Kernel_ast.WarpIs 1, [ Kernel_ast.Barrier Kernel_ast.ThreadSync ])
    ])

let ampere_with_mask () =
  Kernel_ast.make
    ~name:"gemm_mask"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[Kernel_ast.in_arg "A" Kernel_ast.F16
          ; Kernel_ast.in_arg "B" Kernel_ast.F16]
    ~body:(Kernel_ast.Load (
      Kernel_ast.Arg ("A", Kernel_ast.F16, Kernel_ast.Global),
      Kernel_ast.Smem ("smem_A", Kernel_ast.F16, { Kernel_ast.m=128; n=0; k=32 }),
      Some { Kernel_ast.coord_var = "coord"; bounds = [128; 32] }))


let test_lower_name () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check string) "name" "gemm_ampere" tirix.name

let test_lower_family_ampere () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  match tirix.family with
  | Kernel_desc.Ampere -> Alcotest.(check bool) "ampere" true true
  | _ -> Alcotest.fail "wrong family"

let test_lower_family_hopper () =
  let tirix = Ast_to_tirix.lower (hopper_with_mma ()) in
  match tirix.family with
  | Kernel_desc.Hopper -> Alcotest.(check bool) "hopper" true true
  | _ -> Alcotest.fail "wrong family"

let test_lower_family_blackwell () =
  let tirix = Ast_to_tirix.lower (blackwell_with_mma ()) in
  match tirix.family with
  | Kernel_desc.Blackwell -> Alcotest.(check bool) "blackwell" true true
  | _ -> Alcotest.fail "wrong family"

let test_lower_bm () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check int) "bm=128" 128 tirix.bm

let test_lower_bn () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check int) "bn=128" 128 tirix.bn

let test_lower_bk () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check int) "bk=32" 32 tirix.bk

let test_lower_pipeline_depth () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check int) "depth=4" 4 tirix.pipeline_depth

let test_params_has_a () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check bool) "param A" true
    (List.exists tirix.params ~f:(fun p ->
      String.equal p.param_name "A"))

let test_params_has_b () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check bool) "param B" true
    (List.exists tirix.params ~f:(fun p ->
      String.equal p.param_name "B"))

let test_params_has_c () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check bool) "param C" true
    (List.exists tirix.params ~f:(fun p ->
      String.equal p.param_name "C"))

let test_params_count () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check bool) "at least 3 params" true
    (List.length tirix.params >= 3)

let test_cluster_ampere_single_sm () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check bool) "1sm" false
    (Cluster.is_2sm tirix.cluster)

let test_cluster_blackwell_two_sm () =
  let tirix = Ast_to_tirix.lower (blackwell_with_mma ()) in
  Alcotest.(check bool) "2sm" true
    (Cluster.is_2sm tirix.cluster)

let test_cluster_ampere_four_warps () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check int) "4 warps" 4
    tirix.cluster.Cluster.num_warps

let test_cluster_blackwell_six_warps () =
  let tirix = Ast_to_tirix.lower (blackwell_with_mma ()) in
  Alcotest.(check int) "6 warps" 6
    tirix.cluster.Cluster.num_warps

let test_empty_seq_body () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  match tirix.body with
  | [ SSeq [] ] | [] ->
    Alcotest.(check bool) "empty ok" true true
  | _ ->
    Alcotest.(check bool) "empty body" true true


let test_load_produces_copy () =
  let tirix = Ast_to_tirix.lower (ampere_with_load ()) in
  let rec has_copy = function
    | [] -> false
    | SOp (Copy _) :: _ -> true
    | s :: rest ->
      (match s with
       | SSeq stmts -> has_copy stmts || has_copy rest
       | _ -> has_copy rest)
  in
  Alcotest.(check bool) "has copy" true (has_copy tirix.body)

let test_load_ampere_cp_async () =
  let tirix = Ast_to_tirix.lower (ampere_with_load ()) in
  let rec find_copy = function
    | [] -> None
    | SOp (Copy c) :: _ -> Some c
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_copy stmts with
          | Some c -> Some c
          | None -> find_copy rest)
       | _ -> find_copy rest)
  in
  match find_copy tirix.body with
  | None -> Alcotest.fail "no copy found"
  | Some c ->
    Alcotest.(check bool) "cp async" true
      (match c.copy_kind with CpAsync -> true | _ -> false)

let test_load_with_mask () =
  let tirix = Ast_to_tirix.lower (ampere_with_mask ()) in
  let rec find_copy = function
    | [] -> None
    | SOp (Copy c) :: _ -> Some c
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_copy stmts with
          | Some c -> Some c
          | None -> find_copy rest)
       | _ -> find_copy rest)
  in
  match find_copy tirix.body with
  | None -> Alcotest.fail "no copy"
  | Some c ->
    Alcotest.(check bool) "has pred" true
      (Option.is_some c.pred_expr)

let test_store_produces_copy () =
  let tirix = Ast_to_tirix.lower (ampere_with_store ()) in
  let rec has_copy = function
    | [] -> false
    | SOp (Copy _) :: _ -> true
    | s :: rest ->
      (match s with
       | SSeq stmts -> has_copy stmts || has_copy rest
       | _ -> has_copy rest)
  in
  Alcotest.(check bool) "store→copy" true (has_copy tirix.body)


let test_mma_produces_mma_op () =
  let tirix = Ast_to_tirix.lower (ampere_with_mma ()) in
  let rec has_mma = function
    | [] -> false
    | SOp (Mma _) :: _ -> true
    | s :: rest ->
      (match s with
       | SSeq stmts -> has_mma stmts || has_mma rest
       | _ -> has_mma rest)
  in
  Alcotest.(check bool) "has mma" true (has_mma tirix.body)

let test_mma_ampere_sm80 () =
  let tirix = Ast_to_tirix.lower (ampere_with_mma ()) in
  let rec find_mma = function
    | [] -> None
    | SOp (Mma m) :: _ -> Some m
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_mma stmts with
          | Some m -> Some m | None -> find_mma rest)
       | _ -> find_mma rest)
  in
  match find_mma tirix.body with
  | None -> Alcotest.fail "no mma"
  | Some m ->
    Alcotest.(check bool) "sm80" true
      (match m.mma_kind with Sm80Mma -> true | _ -> false)

let test_mma_hopper_wgmma () =
  let tirix = Ast_to_tirix.lower (hopper_with_mma ()) in
  let rec find_mma = function
    | [] -> None
    | SOp (Mma m) :: _ -> Some m
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_mma stmts with
          | Some m -> Some m | None -> find_mma rest)
       | _ -> find_mma rest)
  in
  match find_mma tirix.body with
  | None -> Alcotest.fail "no mma"
  | Some m ->
    Alcotest.(check bool) "wgmma" true
      (match m.mma_kind with Sm90Wgmma -> true | _ -> false)

let test_mma_blackwell_tcgen05 () =
  let tirix = Ast_to_tirix.lower (blackwell_with_mma ()) in
  let rec find_mma = function
    | [] -> None
    | SOp (Mma m) :: _ -> Some m
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_mma stmts with
          | Some m -> Some m | None -> find_mma rest)
       | _ -> find_mma rest)
  in
  match find_mma tirix.body with
  | None -> Alcotest.fail "no mma"
  | Some m ->
    Alcotest.(check bool) "tcgen05" true
      (match m.mma_kind with Sm100Tcgen05 -> true | _ -> false)

let test_mma_accum_flag () =
  let tirix = Ast_to_tirix.lower (ampere_with_mma ()) in
  let rec find_mma = function
    | [] -> None
    | SOp (Mma m) :: _ -> Some m
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_mma stmts with
          | Some m -> Some m | None -> find_mma rest)
       | _ -> find_mma rest)
  in
  match find_mma tirix.body with
  | None -> Alcotest.fail "no mma"
  | Some m ->
    Alcotest.(check bool) "accum flag" true m.accum_flag


let test_for_produces_sfor () =
  let tirix = Ast_to_tirix.lower (ampere_with_for ()) in
  let rec has_for = function
    | [] -> false
    | SFor _ :: _ -> true
    | s :: rest ->
      (match s with
       | SSeq stmts -> has_for stmts || has_for rest
       | _ -> has_for rest)
  in
  Alcotest.(check bool) "has SFor" true (has_for tirix.body)

let test_for_var_name () =
  let tirix = Ast_to_tirix.lower (ampere_with_for ()) in
  let rec find_for = function
    | [] -> None
    | SFor { var; _ } :: _ -> Some var.var_name
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_for stmts with
          | Some name -> Some name
          | None -> find_for rest)
       | _ -> find_for rest)
  in
  match find_for tirix.body with
  | None -> Alcotest.fail "no for"
  | Some name ->
    Alcotest.(check string) "var k" "k" name

let test_for_bounds () =
  let tirix = Ast_to_tirix.lower (ampere_with_for ()) in
  let rec find_for = function
    | [] -> None
    | SFor { start; stop; _ } :: _ -> Some (start, stop)
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_for stmts with
          | Some bounds -> Some bounds
          | None -> find_for rest)
       | _ -> find_for rest)
  in
  match find_for tirix.body with
  | None -> Alcotest.fail "no for"
  | Some (start, stop) ->
    (match start, stop with
     | Const (S32, s), Const (S32, e) ->
       Alcotest.(check int) "start=0" 0 (Int32.to_int_exn s);
       Alcotest.(check int) "stop=8"  8 (Int32.to_int_exn e)
     | _ -> Alcotest.fail "wrong bounds")

let test_pipeline_produces_spipeline () =
  let tirix = Ast_to_tirix.lower (ampere_with_pipeline ()) in
  let rec has_pipeline = function
    | [] -> false
    | SPipeline _ :: _ -> true
    | s :: rest ->
      (match s with
       | SSeq stmts -> has_pipeline stmts || has_pipeline rest
       | _ -> has_pipeline rest)
  in
  Alcotest.(check bool) "has SPipeline" true (has_pipeline tirix.body)

let test_pipeline_stages () =
  let tirix = Ast_to_tirix.lower (ampere_with_pipeline ()) in
  let rec find_pipeline = function
    | [] -> None
    | SPipeline { stages; _ } :: _ -> Some stages
    | s :: rest ->
      (match s with
       | SSeq stmts ->
         (match find_pipeline stmts with
          | Some stages -> Some stages
          | None -> find_pipeline rest)
       | _ -> find_pipeline rest)
  in
  match find_pipeline tirix.body with
  | None -> Alcotest.fail "no pipeline"
  | Some stages ->
    Alcotest.(check int) "stages=4" 4 stages

let test_barrier_thread_sync () =
  let tirix = Ast_to_tirix.lower (ampere_with_barrier_thread ()) in
  let rec has_cta = function
    | [] -> false
    | SOp (Barrier CtaSync) :: _ -> true
    | s :: rest ->
      (match s with
       | SSeq stmts -> has_cta stmts || has_cta rest
       | _ -> has_cta rest)
  in
  Alcotest.(check bool) "CtaSync" true (has_cta tirix.body)

let test_barrier_cluster_sync () =
  let tirix = Ast_to_tirix.lower (ampere_with_barrier_cluster ()) in
  let rec has_cl = function
    | [] -> false
    | SOp (Barrier ClusterArrive) :: _ -> true
    | s :: rest ->
      (match s with
       | SSeq stmts -> has_cl stmts || has_cl rest
       | _ -> has_cl rest)
  in
  Alcotest.(check bool) "ClusterArrive" true (has_cl tirix.body)

let test_if_warp_is () =
  let tirix = Ast_to_tirix.lower (ampere_with_if_warp ()) in
  let rec has_warp_group = function
    | [] -> false
    | Tirix.SWarpGroup _ :: _ -> true
    | s :: rest ->
      (match s with
       | Tirix.SSeq stmts -> has_warp_group stmts || has_warp_group rest
       | _ -> has_warp_group rest)
  in
  Alcotest.(check bool) "has SWarpGroup" true (has_warp_group tirix.body)

let test_if_warp_is_cond_uses_role () =
  let tirix = Ast_to_tirix.lower (ampere_with_if_warp ()) in
  let rec has_producer = function
    | [] -> false
    | Tirix.SWarpGroup (Cluster.Producer, _) :: _ -> true
    | s :: rest ->
      (match s with
       | Tirix.SSeq stmts -> has_producer stmts || has_producer rest
       | _ -> has_producer rest)
  in
  Alcotest.(check bool) "producer role in dispatch" true
    (has_producer tirix.body)


let test_seq_produces_sseq () =
  let tirix = Ast_to_tirix.lower (ampere_with_seq ()) in
  let rec has_seq = function
    | [] -> false
    | SSeq _ :: _ -> true
    | _ :: rest -> has_seq rest
  in
  Alcotest.(check bool) "has SSeq" true (has_seq tirix.body)

let test_seq_length () =
  let tirix = Ast_to_tirix.lower (ampere_with_seq ()) in
  let rec find_seq = function
    | [] -> None
    | SSeq ss :: _ -> Some ss
    | _ :: rest -> find_seq rest
  in
  match find_seq tirix.body with
  | None -> Alcotest.fail "no seq"
  | Some ss ->
    Alcotest.(check int) "2 stmts" 2 (List.length ss)

let test_warp_dispatch_produces_sif () =
  let tirix = Ast_to_tirix.lower (ampere_with_warp_dispatch ()) in
  let rec count_warp_groups = function
    | [] -> 0
    | Tirix.SWarpGroup _ :: rest -> 1 + count_warp_groups rest
    | s :: rest ->
      (match s with
       | Tirix.SSeq stmts -> count_warp_groups stmts + count_warp_groups rest
       | _ -> count_warp_groups rest)
  in
  let n = count_warp_groups tirix.body in
  Alcotest.(check int) "two warp groups" 2 n

let test_verify_empty () =
  let tirix = Ast_to_tirix.lower (ampere_empty ()) in
  Alcotest.(check bool) "verify ok" true
    (Result.is_ok (Tirix_verify.verify tirix))

let test_verify_with_mma () =
  let tirix = Ast_to_tirix.lower (ampere_with_mma ()) in
  Alcotest.(check bool) "verify ok" true
    (Result.is_ok (Tirix_verify.verify tirix))

let test_verify_with_for () =
  let tirix = Ast_to_tirix.lower (ampere_with_for ()) in
  Alcotest.(check bool) "verify ok" true
    (Result.is_ok (Tirix_verify.verify tirix))

let test_verify_with_pipeline () =
  let tirix = Ast_to_tirix.lower (ampere_with_pipeline ()) in
  Alcotest.(check bool) "verify ok" true
    (Result.is_ok (Tirix_verify.verify tirix))

let test_emit_ampere_mma () =
  let tirix = Ast_to_tirix.lower (ampere_with_mma ()) in
  let out = Tirix_emit.emit tirix in
  Alcotest.(check bool) "__global__" true
    (contains "__global__" out.Backend_cute.full_source)

let test_emit_has_kernel_name () =
  let tirix = Ast_to_tirix.lower (ampere_with_mma ()) in
  let out = Tirix_emit.emit tirix in
  Alcotest.(check bool) "kernel name" true
    (contains "gemm_ampere_mma" out.Backend_cute.full_source)

let test_emit_hopper_has_bf16 () =
  let tirix = Ast_to_tirix.lower (hopper_with_mma ()) in
  let out = Tirix_emit.emit tirix in
  Alcotest.(check bool) "bf16" true
    (contains "bf16" out.Backend_cute.full_source)

let test_emit_blackwell_has_tcgen05 () =
  let tirix = Ast_to_tirix.lower (blackwell_with_mma ()) in
  let out = Tirix_emit.emit tirix in
  Alcotest.(check bool) "tcgen05" true
    (contains "tcgen05" out.Backend_cute.full_source)

let () =
  Alcotest.run "Ast_to_tirix" [
    "structure",  [ Alcotest.test_case "name"          `Quick test_lower_name
                  ; Alcotest.test_case "family-ampere" `Quick test_lower_family_ampere
                  ; Alcotest.test_case "family-hopper" `Quick test_lower_family_hopper
                  ; Alcotest.test_case "family-blkwll" `Quick test_lower_family_blackwell
                  ; Alcotest.test_case "bm"            `Quick test_lower_bm
                  ; Alcotest.test_case "bn"            `Quick test_lower_bn
                  ; Alcotest.test_case "bk"            `Quick test_lower_bk
                  ; Alcotest.test_case "depth"         `Quick test_lower_pipeline_depth ];
    "params",     [ Alcotest.test_case "has-a"         `Quick test_params_has_a
                  ; Alcotest.test_case "has-b"         `Quick test_params_has_b
                  ; Alcotest.test_case "has-c"         `Quick test_params_has_c
                  ; Alcotest.test_case "count"         `Quick test_params_count ];
    "cluster",    [ Alcotest.test_case "1sm-ampere"    `Quick test_cluster_ampere_single_sm
                  ; Alcotest.test_case "2sm-blackwell" `Quick test_cluster_blackwell_two_sm
                  ; Alcotest.test_case "4-warps"       `Quick test_cluster_ampere_four_warps
                  ; Alcotest.test_case "6-warps"       `Quick test_cluster_blackwell_six_warps ];
    "empty",      [ Alcotest.test_case "seq-empty"     `Quick test_empty_seq_body ];
    "load",       [ Alcotest.test_case "produces-copy" `Quick test_load_produces_copy
                  ; Alcotest.test_case "cp-async"      `Quick test_load_ampere_cp_async
                  ; Alcotest.test_case "mask"          `Quick test_load_with_mask ];
    "store",      [ Alcotest.test_case "produces-copy" `Quick test_store_produces_copy ];
    "mma",        [ Alcotest.test_case "produces-mma"  `Quick test_mma_produces_mma_op
                  ; Alcotest.test_case "sm80"          `Quick test_mma_ampere_sm80
                  ; Alcotest.test_case "wgmma"         `Quick test_mma_hopper_wgmma
                  ; Alcotest.test_case "tcgen05"       `Quick test_mma_blackwell_tcgen05
                  ; Alcotest.test_case "accum"         `Quick test_mma_accum_flag ];
    "for",        [ Alcotest.test_case "produces-sfor" `Quick test_for_produces_sfor
                  ; Alcotest.test_case "var-name"      `Quick test_for_var_name
                  ; Alcotest.test_case "bounds"        `Quick test_for_bounds ];
    "pipeline",   [ Alcotest.test_case "produces"      `Quick test_pipeline_produces_spipeline
                  ; Alcotest.test_case "stages"        `Quick test_pipeline_stages ];
    "barrier",    [ Alcotest.test_case "thread-sync"   `Quick test_barrier_thread_sync
                  ; Alcotest.test_case "cluster-sync"  `Quick test_barrier_cluster_sync ];
    "if",         [ Alcotest.test_case "warp-is"       `Quick test_if_warp_is
                  ; Alcotest.test_case "warp-id-cond"  `Quick test_if_warp_is_cond_uses_role ];
    "seq",        [ Alcotest.test_case "produces-sseq" `Quick test_seq_produces_sseq
                  ; Alcotest.test_case "length"        `Quick test_seq_length ];
    "dispatch",   [ Alcotest.test_case "warp-dispatch" `Quick test_warp_dispatch_produces_sif ];
    "verify",     [ Alcotest.test_case "empty"         `Quick test_verify_empty
                  ; Alcotest.test_case "mma"           `Quick test_verify_with_mma
                  ; Alcotest.test_case "for"           `Quick test_verify_with_for
                  ; Alcotest.test_case "pipeline"      `Quick test_verify_with_pipeline ];
    "e2e",        [ Alcotest.test_case "ampere-mma"    `Quick test_emit_ampere_mma
                  ; Alcotest.test_case "kernel-name"   `Quick test_emit_has_kernel_name
                  ; Alcotest.test_case "hopper-bf16"   `Quick test_emit_hopper_has_bf16
                  ; Alcotest.test_case "blackwell-tc"  `Quick test_emit_blackwell_has_tcgen05 ];
  ]
