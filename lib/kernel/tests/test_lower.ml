open Tesserae

let ampere_kernel () =
  Kernel_ast.make
    ~name:"gemm_ampere"
    ~arch:Kernel_ast.SM80
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])

let hopper_kernel () =
  Kernel_ast.make
    ~name:"gemm_hopper"
    ~arch:Kernel_ast.SM90
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 64 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("B", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32,  Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])

let blackwell_kernel () =
  Kernel_ast.make
    ~name:"gemm_blackwell"
    ~arch:Kernel_ast.SM100
    ~elem:Kernel_ast.BF16
    ~tile:{ Kernel_ast.m = 128; n = 256; k = 64 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("B", Kernel_ast.BF16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32,  Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])

let test_strategy_sm80 () =
  Alcotest.(check bool) "cp_async" true
    (match Lower.arch_to_strategy Kernel_ast.SM80 with
     | Tile_io.CpAsync -> true | _ -> false)

let test_strategy_sm90 () =
  Alcotest.(check bool) "tma" true
    (match Lower.arch_to_strategy Kernel_ast.SM90 with
     | Tile_io.TmaLoad -> true | _ -> false)

let test_strategy_sm100 () =
  Alcotest.(check bool) "multicast" true
    (match Lower.arch_to_strategy Kernel_ast.SM100 with
     | Tile_io.TmaMulticast -> true | _ -> false)

let test_accum_sm80 () =
  Alcotest.(check bool) "registers" true
    (match Lower.arch_to_accum Kernel_ast.SM80 with
     | Tile_op.Registers -> true | _ -> false)

let test_accum_sm100 () =
  Alcotest.(check bool) "tmem" true
    (match Lower.arch_to_accum Kernel_ast.SM100 with
     | Tile_op.TensorMem -> true | _ -> false)

let test_lower_ampere_family () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "ampere" true
    (match desc.Kernel_desc.family with
     | Kernel_desc.Ampere -> true | _ -> false)

let test_lower_hopper_family () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "hopper" true
    (match desc.Kernel_desc.family with
     | Kernel_desc.Hopper -> true | _ -> false)

let test_lower_blackwell_family () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "blackwell" true
    (match desc.Kernel_desc.family with
     | Kernel_desc.Blackwell -> true | _ -> false)

let test_lower_dims_ampere () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check int) "bm" 128 desc.Kernel_desc.bm;
  Alcotest.(check int) "bn" 128 desc.Kernel_desc.bn;
  Alcotest.(check int) "bk" 32  desc.Kernel_desc.bk

let test_lower_dims_blackwell () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check int) "bm" 128 desc.Kernel_desc.bm;
  Alcotest.(check int) "bn" 256 desc.Kernel_desc.bn;
  Alcotest.(check int) "bk" 64  desc.Kernel_desc.bk


let test_lower_name () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check string) "name" "gemm_ampere" desc.Kernel_desc.name

let test_lower_warps_ampere () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "warps > 0" true
    (desc.Kernel_desc.cluster.Cluster.num_warps > 0)

let test_lower_warps_blackwell () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check int) "6 warps" 6
    desc.Kernel_desc.cluster.Cluster.num_warps

let test_lower_cp_async () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "cp_async" true
    (not (Tile_io.is_tma desc.Kernel_desc.tile_io))

let test_lower_tma () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "tma" true
    (Tile_io.is_tma desc.Kernel_desc.tile_io)

let test_lower_multicast () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "multicast" true
    (Tile_io.is_tma desc.Kernel_desc.tile_io)

let test_lower_accum_registers () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "registers" true
    (not (Tile_op.is_tmem desc.Kernel_desc.tile_op))

let test_lower_accum_tmem () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "tmem" true
    (Tile_op.is_tmem desc.Kernel_desc.tile_op)

let test_lower_valid_ampere () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "valid" true
    (Result.is_ok (Kernel_desc.validate desc))

let test_lower_valid_hopper () =
  let (Lower.Pack desc) = Lower.lower_exn (hopper_kernel ()) in
  Alcotest.(check bool) "valid" true
    (Result.is_ok (Kernel_desc.validate desc))

let test_lower_valid_blackwell () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "valid" true
    (Result.is_ok (Kernel_desc.validate desc))

let test_lower_2sm () =
  let (Lower.Pack desc) = Lower.lower_exn (blackwell_kernel ()) in
  Alcotest.(check bool) "2sm" true
    (Cluster.is_2sm desc.Kernel_desc.cluster)

let test_lower_1sm_ampere () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Alcotest.(check bool) "1sm" false
    (Cluster.is_2sm desc.Kernel_desc.cluster)

let test_lower_ok () =
  Alcotest.(check bool) "ok" true
    (Result.is_ok (Lower.lower (ampere_kernel ())))

let () =
  Alcotest.run "Lower" [
    "strategy",  [ Alcotest.test_case "sm80"  `Quick test_strategy_sm80
                 ; Alcotest.test_case "sm90"  `Quick test_strategy_sm90
                 ; Alcotest.test_case "sm100" `Quick test_strategy_sm100 ];
    "accum",     [ Alcotest.test_case "sm80"  `Quick test_accum_sm80
                 ; Alcotest.test_case "sm100" `Quick test_accum_sm100 ];
    "family",    [ Alcotest.test_case "ampere"    `Quick test_lower_ampere_family
                 ; Alcotest.test_case "hopper"    `Quick test_lower_hopper_family
                 ; Alcotest.test_case "blackwell" `Quick test_lower_blackwell_family ];
    "dims",      [ Alcotest.test_case "ampere"    `Quick test_lower_dims_ampere
                 ; Alcotest.test_case "blackwell" `Quick test_lower_dims_blackwell ];
    "name",      [ Alcotest.test_case "preserved" `Quick test_lower_name ];
    "warps",     [ Alcotest.test_case "ampere"    `Quick test_lower_warps_ampere
                 ; Alcotest.test_case "blackwell" `Quick test_lower_warps_blackwell ];
    "io",        [ Alcotest.test_case "cp_async"  `Quick test_lower_cp_async
                 ; Alcotest.test_case "tma"       `Quick test_lower_tma
                 ; Alcotest.test_case "multicast" `Quick test_lower_multicast ];
    "accum_loc", [ Alcotest.test_case "registers" `Quick test_lower_accum_registers
                 ; Alcotest.test_case "tmem"      `Quick test_lower_accum_tmem ];
    "validate",  [ Alcotest.test_case "ampere"    `Quick test_lower_valid_ampere
                 ; Alcotest.test_case "hopper"    `Quick test_lower_valid_hopper
                 ; Alcotest.test_case "blackwell" `Quick test_lower_valid_blackwell ];
    "cluster",   [ Alcotest.test_case "2sm"       `Quick test_lower_2sm
                 ; Alcotest.test_case "1sm"       `Quick test_lower_1sm_ampere ];
    "result",    [ Alcotest.test_case "ok"        `Quick test_lower_ok ];
  ]
