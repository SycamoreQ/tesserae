open Tesserae

(* ------------------------------------------------------------------ *)
(* constructors                                                        *)
(* ------------------------------------------------------------------ *)

let ampere () =
  Kernel_desc.make_ampere
    ~name:"gemm_ampere"
    ~bm:128 ~bn:128 ~bk:32
    ~elem:Elemtype.Float16
    ~m:4096 ~n:4096 ~k:4096

let hopper () =
  Kernel_desc.make_hopper
    ~name:"gemm_hopper"
    ~bm:128 ~bn:128 ~bk:64
    ~elem:Elemtype.Bfloat16
    ~m:4096 ~n:4096 ~k:4096

let blackwell () =
  Kernel_desc.make_blackwell
    ~name:"gemm_blackwell"
    ~bm:128 ~bn:256 ~bk:64
    ~elem:Elemtype.Bfloat16
    ~m:8192 ~n:8192 ~k:8192

(* ------------------------------------------------------------------ *)
(* family                                                              *)
(* ------------------------------------------------------------------ *)

let test_family_ampere () =
  Alcotest.(check bool) "ampere" true
    (match (ampere ()).Kernel_desc.family with
     | Kernel_desc.Ampere -> true | _ -> false)

let test_family_hopper () =
  Alcotest.(check bool) "hopper" true
    (match (hopper ()).Kernel_desc.family with
     | Kernel_desc.Hopper -> true | _ -> false)

let test_family_blackwell () =
  Alcotest.(check bool) "blackwell" true
    (match (blackwell ()).Kernel_desc.family with
     | Kernel_desc.Blackwell -> true | _ -> false)

(* ------------------------------------------------------------------ *)
(* dimensions                                                          *)
(* ------------------------------------------------------------------ *)

let test_dims_ampere () =
  let k = ampere () in
  Alcotest.(check int) "bm" 128 k.Kernel_desc.bm;
  Alcotest.(check int) "bn" 128 k.Kernel_desc.bn;
  Alcotest.(check int) "bk" 32  k.Kernel_desc.bk

let test_dims_blackwell () =
  let k = blackwell () in
  Alcotest.(check int) "bm" 128 k.Kernel_desc.bm;
  Alcotest.(check int) "bn" 256 k.Kernel_desc.bn;
  Alcotest.(check int) "bk" 64  k.Kernel_desc.bk

(* ------------------------------------------------------------------ *)
(* validate                                                            *)
(* ------------------------------------------------------------------ *)

let test_validate_ampere () =
  Alcotest.(check bool) "valid" true
    (Result.is_ok (Kernel_desc.validate (ampere ())))

let test_validate_hopper () =
  Alcotest.(check bool) "valid" true
    (Result.is_ok (Kernel_desc.validate (hopper ())))

let test_validate_blackwell () =
  Alcotest.(check bool) "valid" true
    (Result.is_ok (Kernel_desc.validate (blackwell ())))

(* ------------------------------------------------------------------ *)
(* arithmetic_intensity                                                *)
(* ------------------------------------------------------------------ *)

let test_ai_ampere () =
  (* 2*128*128*32 / ((128+128)*32*2) = 1048576 / 16384 = 64.0 *)
  let ai = Kernel_desc.arithmetic_intensity (ampere ()) in
  Alcotest.(check bool) "ai > 0" true (ai > 0.0)

let test_ai_blackwell_higher () =
  (* 2SM halves B → higher AI *)
  let ai_b = Kernel_desc.arithmetic_intensity (blackwell ()) in
  let ai_h = Kernel_desc.arithmetic_intensity (hopper ()) in
  Alcotest.(check bool) "blackwell > hopper AI" true (ai_b > ai_h)

(* ------------------------------------------------------------------ *)
(* smem_bytes                                                          *)
(* ------------------------------------------------------------------ *)

let test_smem_ampere () =
  (* pipeline_depth * (BM + BN) * BK * 2 bytes *)
  let s = Kernel_desc.smem_bytes (ampere ()) in
  Alcotest.(check bool) "smem > 0" true (s > 0)

let test_smem_fits () =
  (* must fit in 227KB *)
  let s = Kernel_desc.smem_bytes (blackwell ()) in
  Alcotest.(check bool) "fits 227KB" true (s <= 227 * 1024)

(* ------------------------------------------------------------------ *)
(* num_warps                                                           *)
(* ------------------------------------------------------------------ *)

let test_num_warps_ampere () =
  (* 1 warp for mma.sync *)
  Alcotest.(check bool) "warps > 0" true
    (Kernel_desc.num_warps (ampere ()) > 0)

let test_num_warps_blackwell () =
  (* 6 warps: producer + consumer + 3 epilogue + scheduler *)
  Alcotest.(check int) "6 warps" 6
    (Kernel_desc.num_warps (blackwell ()))

(* ------------------------------------------------------------------ *)
(* emit_kernel_params                                                  *)
(* ------------------------------------------------------------------ *)

let test_emit_params_ampere () =
  let s = Kernel_desc.emit_kernel_params (ampere ()) in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has A"   true (contains "A" s);
  Alcotest.(check bool) "has B"   true (contains "B" s);
  Alcotest.(check bool) "has C"   true (contains "C" s);
  Alcotest.(check bool) "has M"   true (contains "M" s)

let test_emit_params_tma () =
  let s = Kernel_desc.emit_kernel_params (hopper ()) in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has tmap" true (contains "tmap" s)

(* ------------------------------------------------------------------ *)
(* emit_launch_config                                                  *)
(* ------------------------------------------------------------------ *)

let test_launch_ampere () =
  let s = Kernel_desc.emit_launch_config (ampere ()) 4096 4096 in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has dim3" true (contains "dim3" s)

let test_launch_blackwell () =
  let s = Kernel_desc.emit_launch_config (blackwell ()) 8192 8192 in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has cluster" true (contains "cluster" s)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Kernel_desc" [
    "family",  [ Alcotest.test_case "ampere"    `Quick test_family_ampere
               ; Alcotest.test_case "hopper"    `Quick test_family_hopper
               ; Alcotest.test_case "blackwell" `Quick test_family_blackwell ];
    "dims",    [ Alcotest.test_case "ampere"    `Quick test_dims_ampere
               ; Alcotest.test_case "blackwell" `Quick test_dims_blackwell ];
    "validate",[ Alcotest.test_case "ampere"    `Quick test_validate_ampere
               ; Alcotest.test_case "hopper"    `Quick test_validate_hopper
               ; Alcotest.test_case "blackwell" `Quick test_validate_blackwell ];
    "ai",      [ Alcotest.test_case "positive"  `Quick test_ai_ampere
               ; Alcotest.test_case "2sm-higher"`Quick test_ai_blackwell_higher ];
    "smem",    [ Alcotest.test_case "positive"  `Quick test_smem_ampere
               ; Alcotest.test_case "fits"      `Quick test_smem_fits ];
    "warps",   [ Alcotest.test_case "positive"  `Quick test_num_warps_ampere
               ; Alcotest.test_case "blackwell" `Quick test_num_warps_blackwell ];
    "params",  [ Alcotest.test_case "ampere"    `Quick test_emit_params_ampere
               ; Alcotest.test_case "tma"       `Quick test_emit_params_tma ];
    "launch",  [ Alcotest.test_case "ampere"    `Quick test_launch_ampere
               ; Alcotest.test_case "blackwell" `Quick test_launch_blackwell ];
  ]
