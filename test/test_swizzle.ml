open Tesserae

let test_make_valid () =
  let sw = Swizzle.make 2 4 3 in
  Alcotest.(check int) "b" 2 sw.Swizzle.b;
  Alcotest.(check int) "m" 4 sw.Swizzle.m;
  Alcotest.(check int) "s" 3 sw.Swizzle.s

let test_make_invalid () =
  Alcotest.check_raises "negative b"
    (Invalid_argument "swizzle parameters must be non-negative")
    (fun () -> ignore (Swizzle.make (-1) 4 3))

let test_apply_identity () =
  (* Swizzle<0,4,3>: yyy_msk = 0, always identity *)
  let sw = Swizzle.make 0 4 3 in
  Alcotest.(check int) "0"   0   (Swizzle.apply sw 0);
  Alcotest.(check int) "127" 127 (Swizzle.apply sw 127);
  Alcotest.(check int) "255" 255 (Swizzle.apply sw 255)

let test_apply_b1 () =
  (* Swizzle<1,4,3>: 32B swizzle
     yyy_msk = 1 << (4+3) = 128 = 0b10000000
     apply(x) = x XOR ((x AND 128) >> 3)
     apply(255) = 255 XOR ((255 AND 128) >> 3)
               = 255 XOR (128 >> 3)
               = 255 XOR 16
               = 239 *)
  let sw = Swizzle.make 1 4 3 in
  Alcotest.(check int) "0"        0   (Swizzle.apply sw 0);
  Alcotest.(check int) "127->127" 127 (Swizzle.apply sw 127);
  Alcotest.(check int) "255->239" 239 (Swizzle.apply sw 255)

let test_apply_b2 () =
  (* Swizzle<2,4,3>: 64B swizzle
     yyy_msk = 3 << (4+3) = 3 << 7 = 384 = 0b110000000
     apply(511) = 511 XOR ((511 AND 384) >> 3)
               = 511 XOR (384 >> 3)
               = 511 XOR 48
               = 463
     apply(255) = 255 XOR ((255 AND 384) >> 3)
               = 255 XOR 0
               = 255  (384 > 255, AND = 0) *)
  let sw = Swizzle.make 2 4 3 in
  Alcotest.(check int) "0"        0   (Swizzle.apply sw 0);
  Alcotest.(check int) "511->463" 463 (Swizzle.apply sw 511)

let test_apply_b3 () =
  (* Swizzle<3,4,3>: 128B swizzle
     yyy_msk = 7 << (4+3) = 7 << 7 = 896 = 0b1110000000
     apply(1023) = 1023 XOR ((1023 AND 896) >> 3)
                = 1023 XOR (896 >> 3)
                = 1023 XOR 112
                = 911 *)
  let sw = Swizzle.make 3 4 3 in
  Alcotest.(check int) "0"         0   (Swizzle.apply sw 0);
  Alcotest.(check int) "1023->911" 911 (Swizzle.apply sw 1023)

let test_self_inverse_b1 () =
  let sw = Swizzle.make 1 4 3 in
  for x = 0 to 255 do
    let y = Swizzle.apply sw x in
    let z = Swizzle.apply sw y in
    if z <> x then
      Alcotest.failf "not self-inverse at x=%d: got %d" x z
  done

let test_self_inverse_b3 () =
  let sw = Swizzle.make 3 4 3 in
  for x = 0 to 1023 do
    let y = Swizzle.apply sw x in
    let z = Swizzle.apply sw y in
    if z <> x then
      Alcotest.failf "not self-inverse at x=%d: got %d" x z
  done

let test_is_identity_true () =
  Alcotest.(check bool) "b=0" true
    (Swizzle.is_identity (Swizzle.make 0 4 3))

let test_is_identity_false () =
  Alcotest.(check bool) "b=2" false
    (Swizzle.is_identity (Swizzle.make 2 4 3))

let test_mask_bits () =
  Alcotest.(check int) "b=2 -> 4" 4
    (Swizzle.mask_bits (Swizzle.make 2 4 3));
  Alcotest.(check int) "b=3 -> 8" 8
    (Swizzle.mask_bits (Swizzle.make 3 4 3))

let test_compose_valid () =
  let sw1 = Swizzle.make 1 4 3 in
  let sw2 = Swizzle.make 1 4 3 in
  let sw  = Swizzle.compose sw1 sw2 in
  Alcotest.(check int) "b=2" 2 sw.Swizzle.b;
  Alcotest.(check int) "m=4" 4 sw.Swizzle.m;
  Alcotest.(check int) "s=3" 3 sw.Swizzle.s

let test_compose_invalid () =
  let sw1 = Swizzle.make 1 4 3 in
  let sw2 = Swizzle.make 1 3 3 in
  Alcotest.check_raises "incompatible"
    (Invalid_argument "incompatible swizzle shifts for composition")
    (fun () -> ignore (Swizzle.compose sw1 sw2))

let test_smem_selector_f16_128 () =
  (* float16, 128B tile: tile_k*2=128 → B=3, M=4, S=3 *)
  let sw = Swizzle.smem_selector Elemtype.Float16 64 64 in
  Alcotest.(check int) "b" 3 sw.Swizzle.b;
  Alcotest.(check int) "m" 4 sw.Swizzle.m;
  Alcotest.(check int) "s" 3 sw.Swizzle.s

let test_smem_selector_f16_64 () =
  (* float16, 64B tile: tile_k*2=64 → B=2, M=4, S=3 *)
  let sw = Swizzle.smem_selector Elemtype.Float16 64 32 in
  Alcotest.(check int) "b" 2 sw.Swizzle.b;
  Alcotest.(check int) "m" 4 sw.Swizzle.m;
  Alcotest.(check int) "s" 3 sw.Swizzle.s

let test_smem_selector_f32_128 () =
  (* float32, 128B tile: tile_k*4=128 → B=3, M=4, S=3 *)
  let sw = Swizzle.smem_selector Elemtype.Float32 64 32 in
  Alcotest.(check int) "b" 3 sw.Swizzle.b;
  Alcotest.(check int) "m" 4 sw.Swizzle.m;
  Alcotest.(check int) "s" 3 sw.Swizzle.s

let test_smem_selector_identity () =
  (* float32, very small tile: tile_k*4=4 < 16 → B=0, identity *)
  let sw = Swizzle.smem_selector Elemtype.Float32 4 1 in
  Alcotest.(check bool) "identity" true (Swizzle.is_identity sw)

let test_pp () =
  let sw = Swizzle.make 2 4 3 in
  let s  = Stdlib.Format.asprintf "%a" Swizzle.pp sw in
  Alcotest.(check string) "pp" "Swizzle<2,4,3>" s

let test_emit_cpp_identity () =
  let sw = Swizzle.make 0 4 3 in
  Alcotest.(check string) "identity" "Swizzle<0,4,3>"
    (Swizzle.emit_cpp sw)

let test_emit_cpp_128b () =
  let sw = Swizzle.make 3 4 3 in
  Alcotest.(check string) "128B" "Swizzle<3,4,3>"
    (Swizzle.emit_cpp sw)

let () =
  Alcotest.run "Swizzle" [
    "make",     [ Alcotest.test_case "valid"   `Quick test_make_valid
                ; Alcotest.test_case "invalid" `Quick test_make_invalid ];
    "apply",    [ Alcotest.test_case "identity" `Quick test_apply_identity
                ; Alcotest.test_case "b1"       `Quick test_apply_b1
                ; Alcotest.test_case "b2"       `Quick test_apply_b2
                ; Alcotest.test_case "b3"       `Quick test_apply_b3 ];
    "inverse",  [ Alcotest.test_case "b1" `Quick test_self_inverse_b1
                ; Alcotest.test_case "b3" `Quick test_self_inverse_b3 ];
    "identity", [ Alcotest.test_case "true"  `Quick test_is_identity_true
                ; Alcotest.test_case "false" `Quick test_is_identity_false ];
    "mask",     [ Alcotest.test_case "bits"  `Quick test_mask_bits ];
    "compose",  [ Alcotest.test_case "valid"   `Quick test_compose_valid
                ; Alcotest.test_case "invalid" `Quick test_compose_invalid ];
    "selector", [ Alcotest.test_case "f16-128" `Quick test_smem_selector_f16_128
                ; Alcotest.test_case "f16-64"  `Quick test_smem_selector_f16_64
                ; Alcotest.test_case "f32-128" `Quick test_smem_selector_f32_128
                ; Alcotest.test_case "identity"`Quick test_smem_selector_identity ];
    "pp",       [ Alcotest.test_case "pp"      `Quick test_pp ];
    "emit",     [ Alcotest.test_case "identity" `Quick test_emit_cpp_identity
                ; Alcotest.test_case "128b"     `Quick test_emit_cpp_128b ];
  ]
