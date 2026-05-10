open Tesserae

let test_sw_none () =
  let sw = Swizzle.make 0 4 3 in
  Alcotest.(check bool) "no swizzle" true
    (Smem_desc.swizzle_mode_of sw = Smem_desc.NoSwizzle)

let test_sw_32b () =
  let sw = Swizzle.make 1 4 3 in
  Alcotest.(check bool) "32B" true
    (Smem_desc.swizzle_mode_of sw = Smem_desc.Swizzle32B)

let test_sw_64b () =
  let sw = Swizzle.make 2 4 3 in
  Alcotest.(check bool) "64B" true
    (Smem_desc.swizzle_mode_of sw = Smem_desc.Swizzle64B)

let test_sw_128b () =
  let sw = Swizzle.make 3 4 3 in
  Alcotest.(check bool) "128B" true
    (Smem_desc.swizzle_mode_of sw = Smem_desc.Swizzle128B)

let test_bits_none () =
  Alcotest.(check int) "none=0" 0
    (Smem_desc.swizzle_mode_bits Smem_desc.NoSwizzle)

let test_bits_32b () =
  Alcotest.(check int) "32b=1" 1
    (Smem_desc.swizzle_mode_bits Smem_desc.Swizzle32B)

let test_bits_64b () =
  Alcotest.(check int) "64b=2" 2
    (Smem_desc.swizzle_mode_bits Smem_desc.Swizzle64B)

let test_bits_128b () =
  Alcotest.(check int) "128b=3" 3
    (Smem_desc.swizzle_mode_bits Smem_desc.Swizzle128B)

let test_encode_zero () =
  let d = Smem_desc.make
    ~base_addr:0 ~leading_off:0 ~stride_off:0
    ~swizzle_mode:Smem_desc.NoSwizzle in
  Alcotest.(check int) "zero" 0 (Smem_desc.encode d)

let test_encode_base () =
  (* base_addr=1 → bits[13:0]=1 *)
  let d = Smem_desc.make
    ~base_addr:1 ~leading_off:0 ~stride_off:0
    ~swizzle_mode:Smem_desc.NoSwizzle in
  Alcotest.(check int) "base" 1 (Smem_desc.encode d)

let test_encode_leading () =
  (* leading_off=1 → bits[29:16]=1 → value = 1 lsl 16 = 65536 *)
  let d = Smem_desc.make
    ~base_addr:0 ~leading_off:1 ~stride_off:0
    ~swizzle_mode:Smem_desc.NoSwizzle in
  Alcotest.(check int) "leading" 65536 (Smem_desc.encode d)

let test_encode_swizzle128 () =
  (* swizzle=3 → bits[62:61]=3 → value = 3 lsl 61 *)
  let d = Smem_desc.make
    ~base_addr:0 ~leading_off:0 ~stride_off:0
    ~swizzle_mode:Smem_desc.Swizzle128B in
  Alcotest.(check int) "swizzle128"
    (3 lsl 61)
    (Smem_desc.encode d)

let test_emit_make_smem_desc () =
  let s = Smem_desc.emit_make_smem_desc "A_smem" 128 16 Smem_desc.Swizzle128B in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has make_smem_desc" true (contains "make_smem_desc" s);
  Alcotest.(check bool) "has A_smem"         true (contains "A_smem" s)

let test_emit_cpp_helper () =
  let s = Smem_desc.emit_cpp_helper () in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has uint64_t" true (contains "uint64_t" s);
  Alcotest.(check bool) "has make_smem_desc" true (contains "make_smem_desc" s)

let () =
  Alcotest.run "Smem_desc" [
    "sw_mode",  [ Alcotest.test_case "none"  `Quick test_sw_none
                ; Alcotest.test_case "32b"   `Quick test_sw_32b
                ; Alcotest.test_case "64b"   `Quick test_sw_64b
                ; Alcotest.test_case "128b"  `Quick test_sw_128b ];
    "bits",     [ Alcotest.test_case "none"  `Quick test_bits_none
                ; Alcotest.test_case "32b"   `Quick test_bits_32b
                ; Alcotest.test_case "64b"   `Quick test_bits_64b
                ; Alcotest.test_case "128b"  `Quick test_bits_128b ];
    "encode",   [ Alcotest.test_case "zero"     `Quick test_encode_zero
                ; Alcotest.test_case "base"     `Quick test_encode_base
                ; Alcotest.test_case "leading"  `Quick test_encode_leading
                ; Alcotest.test_case "swizzle"  `Quick test_encode_swizzle128 ];
    "emit",     [ Alcotest.test_case "desc"   `Quick test_emit_make_smem_desc
                ; Alcotest.test_case "helper" `Quick test_emit_cpp_helper ];
  ]
