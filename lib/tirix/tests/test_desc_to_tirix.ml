open Base
open Tesserae_core
open Tesserae_pipeline
open Tesserae_kernel
open Tesserae_tirix
open Tirix


let ampere_kernel () =
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

let hopper_kernel () =
  Kernel_ast.make
    ~name:"gemm_hopper"
    ~arch:Kernel_ast.SM90a
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 64 }
    ~stages:4
    ~args:[ Kernel_ast.in_arg "A" Kernel_ast.BF16
          ; Kernel_ast.in_arg "B" Kernel_ast.F16
          ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Seq [])

let blackwell_kernel () =
  Kernel_ast.make
    ~name:"gemm_blackwell"
    ~arch:Kernel_ast.SM100a
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 256; k = 64 }
    ~stages:4
    ~args:[ Kernel_ast.in_arg "A" Kernel_ast.BF16
          ; Kernel_ast.in_arg "B" Kernel_ast.F16
          ; Kernel_ast.out_arg "C" Kernel_ast.F32 ]
    ~body:(Kernel_ast.Seq [])

let lower k =
  let (Lower.Pack desc) = Lower.lower_exn k in
  Desc_to_tirix.lower desc

(* --- AST Deep Inspectors --- *)

let matches_role role = function
  | Cluster.Producer -> (match role with Cluster.Producer -> true | _ -> false)
  | Cluster.Consumer -> (match role with Cluster.Consumer -> true | _ -> false)
  | Cluster.Epilogue -> (match role with Cluster.Epilogue -> true | _ -> false)
  | Cluster.Scheduler -> (match role with Cluster.Scheduler -> true | _ -> false)

let has_warp_role body role =
  let rec search = function
    | [] -> false
    | SWarpGroup (r, _) :: rest -> matches_role role r || search rest
    | SIf (_, thn, els) :: rest -> search thn || search els || search rest
    | SFor { body; _ } :: rest  -> search body || search rest
    | SSeq ss :: rest -> search ss   || search rest
    | _ :: rest -> search rest
  in
  search body

let find_in_body body pred =
  let rec go = function
    | [] -> false
    | s :: rest ->
      if pred s then true
      else match s with
        | SIf (_, thn, els) -> go thn || go els || go rest
        | SFor { body; _ }  -> go body || go rest
        | SWarpGroup (_, b) -> go b    || go rest
        | SSeq ss -> go ss   || go rest
        | _  -> go rest
  in
  go body


let test_name_ampere () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check string) "name" "gemm_ampere" tirix.name

let test_name_hopper () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check string) "name" "gemm_hopper" tirix.name

let test_name_blackwell () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check string) "name" "gemm_blackwell" tirix.name

let test_family_ampere () =
  let tirix = lower (ampere_kernel ()) in
  match tirix.family with
  | Kernel_desc.Ampere -> Alcotest.(check bool) "ampere" true true
  | _ -> Alcotest.fail "wrong family"

let test_family_hopper () =
  let tirix = lower (hopper_kernel ()) in
  match tirix.family with
  | Kernel_desc.Hopper -> Alcotest.(check bool) "hopper" true true
  | _ -> Alcotest.fail "wrong family"

let test_family_blackwell () =
  let tirix = lower (blackwell_kernel ()) in
  match tirix.family with
  | Kernel_desc.Blackwell -> Alcotest.(check bool) "blackwell" true true
  | _ -> Alcotest.fail "wrong family"

let test_bm_ampere () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check int) "bm=128" 128 tirix.bm

let test_bn_ampere () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check int) "bn=128" 128 tirix.bn

let test_bk_ampere () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check int) "bk=32" 32 tirix.bk

let test_bk_hopper () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check int) "bk=64" 64 tirix.bk

let test_smem_bytes_positive () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "smem>0" true (tirix.smem_bytes > 0)

let test_pipeline_depth () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check int) "depth=4" 4 tirix.pipeline_depth

let test_body_nonempty () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "body not empty" true (List.length tirix.body > 0)

(* --- Tensors Tests --- *)

let test_has_smem_a () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "smem_A" true
    (List.exists tirix.tensors ~f:(fun (n,_) -> String.equal n "smem_A"))

let test_has_smem_b () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "smem_B" true
    (List.exists tirix.tensors ~f:(fun (n,_) -> String.equal n "smem_B"))

let test_smem_a_is_shared () =
  let tirix = lower (ampere_kernel ()) in
  match List.find tirix.tensors ~f:(fun (n,_) -> String.equal n "smem_A") with
  | None -> Alcotest.fail "smem_A not found"
  | Some (_, Tensor t) ->
    Alcotest.(check string) "shared" "shared" (Memspace.name t.tensor_memspace)

let test_smem_layout_size_positive () =
  let tirix = lower (ampere_kernel ()) in
  match List.find tirix.tensors ~f:(fun (n,_) -> String.equal n "smem_A") with
  | None -> Alcotest.fail "smem_A not found"
  | Some (_, Tensor t) ->
    Alcotest.(check bool) "size>0" true (Layout.size t.tensor_layout > 0)

let test_blackwell_bn_smem_halved () =
  let tirix = lower (blackwell_kernel ()) in
  match List.find tirix.tensors ~f:(fun (n,_) -> String.equal n "smem_B") with
  | None -> Alcotest.fail "smem_B not found"
  | Some (_, Tensor t) ->
    let shape_flat = Modes.flatten t.tensor_layout.Layout.shape in
    let bn_in_layout = List.nth_exn shape_flat 1 in
    Alcotest.(check int) "bn/2=128" 128 bn_in_layout

(* --- Parameters Tests --- *)

let test_has_param_a () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "A" true (List.exists tirix.params ~f:(fun p -> String.equal p.param_name "A"))

let test_has_param_b () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "B" true (List.exists tirix.params ~f:(fun p -> String.equal p.param_name "B"))

let test_has_param_c () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "C" true (List.exists tirix.params ~f:(fun p -> String.equal p.param_name "C"))

let test_has_param_m () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "M" true (List.exists tirix.params ~f:(fun p -> String.equal p.param_name "M"))

let test_has_param_n () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "N" true (List.exists tirix.params ~f:(fun p -> String.equal p.param_name "N"))

let test_has_param_k () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "K" true (List.exists tirix.params ~f:(fun p -> String.equal p.param_name "K"))

let test_ampere_no_tma_params () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "no tma" false (List.exists tirix.params ~f:(fun p -> p.param_is_tma))

let test_hopper_has_tma_params () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check bool) "tma" true (List.exists tirix.params ~f:(fun p -> p.param_is_tma))

let test_blackwell_has_tma_params () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "tma" true (List.exists tirix.params ~f:(fun p -> p.param_is_tma))


let test_has_producer () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "producer" true (has_warp_role tirix.body Cluster.Producer)

let test_has_consumer () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "consumer" true (has_warp_role tirix.body Cluster.Consumer)

let test_has_epilogue () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "epilogue" true (has_warp_role tirix.body Cluster.Epilogue)

let test_blackwell_has_scheduler () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "scheduler" true (has_warp_role tirix.body Cluster.Scheduler)


let test_ampere_cp_async () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "cp async" true
    (find_in_body tirix.body (function
      | SOp (Copy { copy_kind = CpAsync; _ }) -> true
      | _ -> false))

let test_hopper_tma_load () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check bool) "tma load" true
    (find_in_body tirix.body (function
      | SOp (Copy { copy_kind = TmaLoad; _ }) -> true
      | _ -> false))

let test_blackwell_tma_multicast () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "tma multicast" true
    (find_in_body tirix.body (function
      | SOp (Copy { copy_kind = TmaMulticast; _ }) -> true
      | _ -> false))

(* --- MMA Instruction Target Routing --- *)

let test_ampere_sm80_mma () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "sm80" true
    (find_in_body tirix.body (function
      | SOp (Mma { mma_kind = Sm80Mma; _ }) -> true
      | _ -> false))

let test_hopper_wgmma () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check bool) "wgmma" true
    (find_in_body tirix.body (function
      | SOp (Mma { mma_kind = Sm90Wgmma; _ }) -> true
      | _ -> false))

let test_blackwell_tcgen05 () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "tcgen05" true
    (find_in_body tirix.body (function
      | SOp (Mma { mma_kind = Sm100Tcgen05; _ }) -> true
      | _ -> false))


let test_ampere_no_tmem_alloc () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "no tmem alloc" false
    (List.exists tirix.body ~f:(function SOp (TmemAlloc _) -> true | _ -> false))

let test_blackwell_has_tmem_alloc () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "tmem alloc" true
    (List.exists tirix.body ~f:(function SOp (TmemAlloc _) -> true | _ -> false))

let test_blackwell_has_tmem_dealloc () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "tmem dealloc" true
    (List.exists tirix.body ~f:(function SOp (TmemDealloc _) -> true | _ -> false))

let test_blackwell_tmem_col_count () =
  let tirix = lower (blackwell_kernel ()) in
  match List.find tirix.body ~f:(function SOp (TmemAlloc _) -> true | _ -> false) with
  | Some (SOp (TmemAlloc { col_count; _ })) -> Alcotest.(check int) "col_count=128" 128 col_count
  | _ -> Alcotest.fail "no tmem alloc"

let test_ampere_no_helpers () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check int) "no helpers" 0 (List.length tirix.helpers)

let test_hopper_has_make_smem_desc () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check bool) "make_smem_desc" true
    (List.exists tirix.helpers ~f:(fun h -> String.equal h.hf_name "make_smem_desc"))

let test_hopper_has_tma_load_helper () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check bool) "tma_2d_gmem2smem" true
    (List.exists tirix.helpers ~f:(fun h -> String.equal h.hf_name "tma_2d_gmem2smem"))

let test_blackwell_has_multicast_helper () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "tma_multicast" true
    (List.exists tirix.helpers ~f:(fun h -> String.equal h.hf_name "tma_2d_gmem2smem_multicast"))

let test_blackwell_helper_count () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check int) "3 helpers" 3 (List.length tirix.helpers)

let test_hopper_helper_count () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check int) "2 helpers" 2 (List.length tirix.helpers)


let test_ampere_single_sm () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "1sm" false (Cluster.is_2sm tirix.cluster)

let test_blackwell_two_sm () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "2sm" true (Cluster.is_2sm tirix.cluster)

let test_ampere_four_warps () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check int) "4 warps" 4 tirix.cluster.Cluster.num_warps

let test_blackwell_six_warps () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check int) "6 warps" 6 tirix.cluster.Cluster.num_warps

(* --- Body Register Scopes --- *)

let test_warp_id_declared () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "warp_id decl" true
    (List.exists tirix.body ~f:(function SLet (v, _) -> String.equal v.var_name "warp_id" | _ -> false))

let test_lane_id_declared () =
  let tirix = lower (ampere_kernel ()) in
  Alcotest.(check bool) "lane_id decl" true
    (List.exists tirix.body ~f:(function SLet (v, _) -> String.equal v.var_name "lane_id" | _ -> false))

let test_hopper_full_mbar_declared () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check bool) "full_mbar decl" true
    (List.exists tirix.body ~f:(function SLet (v, _) -> String.equal v.var_name "full_mbar" | _ -> false))

let test_hopper_empty_mbar_declared () =
  let tirix = lower (hopper_kernel ()) in
  Alcotest.(check bool) "empty_mbar decl" true
    (List.exists tirix.body ~f:(function SLet (v, _) -> String.equal v.var_name "empty_mbar" | _ -> false))

let test_blackwell_tmem_addr_declared () =
  let tirix = lower (blackwell_kernel ()) in
  Alcotest.(check bool) "tmem_addr decl" true
    (List.exists tirix.body ~f:(function
      | SLet (v, _) | SLetMut (v, _) -> String.equal v.var_name "tmem_addr"
      | _ -> false))

let () =
  Alcotest.run "Desc_to_tirix" [
    "structure",  [ Alcotest.test_case "name-ampere"    `Quick test_name_ampere
                  ; Alcotest.test_case "name-hopper"    `Quick test_name_hopper
                  ; Alcotest.test_case "name-blackwell" `Quick test_name_blackwell
                  ; Alcotest.test_case "family-ampere"  `Quick test_family_ampere
                  ; Alcotest.test_case "family-hopper"  `Quick test_family_hopper
                  ; Alcotest.test_case "family-blkwll"  `Quick test_family_blackwell
                  ; Alcotest.test_case "bm"             `Quick test_bm_ampere
                  ; Alcotest.test_case "bn"             `Quick test_bn_ampere
                  ; Alcotest.test_case "bk-ampere"      `Quick test_bk_ampere
                  ; Alcotest.test_case "bk-hopper"      `Quick test_bk_hopper
                  ; Alcotest.test_case "smem-bytes"     `Quick test_smem_bytes_positive
                  ; Alcotest.test_case "depth"          `Quick test_pipeline_depth
                  ; Alcotest.test_case "body-nonempty"  `Quick test_body_nonempty ];
    "tensors",    [ Alcotest.test_case "smem-a"         `Quick test_has_smem_a
                  ; Alcotest.test_case "smem-b"         `Quick test_has_smem_b
                  ; Alcotest.test_case "smem-a-shared"  `Quick test_smem_a_is_shared
                  ; Alcotest.test_case "layout-size"    `Quick test_smem_layout_size_positive
                  ; Alcotest.test_case "bw-bn-halved"   `Quick test_blackwell_bn_smem_halved ];
    "params",     [ Alcotest.test_case "A"              `Quick test_has_param_a
                  ; Alcotest.test_case "B"              `Quick test_has_param_b
                  ; Alcotest.test_case "C"              `Quick test_has_param_c
                  ; Alcotest.test_case "M"              `Quick test_has_param_m
                  ; Alcotest.test_case "N"              `Quick test_has_param_n
                  ; Alcotest.test_case "K"              `Quick test_has_param_k
                  ; Alcotest.test_case "no-tma-ampere"  `Quick test_ampere_no_tma_params
                  ; Alcotest.test_case "tma-hopper"     `Quick test_hopper_has_tma_params
                  ; Alcotest.test_case "tma-blackwell"  `Quick test_blackwell_has_tma_params ];
    "warp",       [ Alcotest.test_case "producer"       `Quick test_has_producer
                  ; Alcotest.test_case "consumer"       `Quick test_has_consumer
                  ; Alcotest.test_case "epilogue"       `Quick test_has_epilogue
                  ; Alcotest.test_case "scheduler"      `Quick test_blackwell_has_scheduler ];
    "copy",       [ Alcotest.test_case "cp-async"       `Quick test_ampere_cp_async
                  ; Alcotest.test_case "tma-load"       `Quick test_hopper_tma_load
                  ; Alcotest.test_case "multicast"      `Quick test_blackwell_tma_multicast ];
    "mma",        [ Alcotest.test_case "sm80"           `Quick test_ampere_sm80_mma
                  ; Alcotest.test_case "wgmma"          `Quick test_hopper_wgmma
                  ; Alcotest.test_case "tcgen05"        `Quick test_blackwell_tcgen05 ];
    "tmem",       [ Alcotest.test_case "no-alloc-amp"   `Quick test_ampere_no_tmem_alloc
                  ; Alcotest.test_case "alloc-blkwll"   `Quick test_blackwell_has_tmem_alloc
                  ; Alcotest.test_case "dealloc-blkwll" `Quick test_blackwell_has_tmem_dealloc
                  ; Alcotest.test_case "col-count"      `Quick test_blackwell_tmem_col_count ];
    "helpers",    [ Alcotest.test_case "none-ampere"    `Quick test_ampere_no_helpers
                  ; Alcotest.test_case "smem-desc"      `Quick test_hopper_has_make_smem_desc
                  ; Alcotest.test_case "tma-load"       `Quick test_hopper_has_tma_load_helper
                  ; Alcotest.test_case "multicast"      `Quick test_blackwell_has_multicast_helper
                  ; Alcotest.test_case "count-blkwll"   `Quick test_blackwell_helper_count
                  ; Alcotest.test_case "count-hopper"   `Quick test_hopper_helper_count ];
    "cluster",    [ Alcotest.test_case "1sm-ampere"     `Quick test_ampere_single_sm
                  ; Alcotest.test_case "2sm-blackwell"  `Quick test_blackwell_two_sm
                  ; Alcotest.test_case "4-warps"        `Quick test_ampere_four_warps
                  ; Alcotest.test_case "6-warps"        `Quick test_blackwell_six_warps ];
    "vars",       [ Alcotest.test_case "warp-id"        `Quick test_warp_id_declared
                  ; Alcotest.test_case "lane-id"        `Quick test_lane_id_declared
                  ; Alcotest.test_case "full-mbar"      `Quick test_hopper_full_mbar_declared
                  ; Alcotest.test_case "empty-mbar"     `Quick test_hopper_empty_mbar_declared
                  ; Alcotest.test_case "tmem-addr"      `Quick test_blackwell_tmem_addr_declared ];
  ]
