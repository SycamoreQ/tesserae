open Tesserae


let test_make_valid_cta1 () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:128 ~num_rows:128 in
  Alcotest.(check int) "cols" 128 t.Tmem.num_cols;
  Alcotest.(check int) "rows" 128 t.Tmem.num_rows

let test_make_valid_cta2 () =
  let t = Tmem.make ~cta_group:Tmem.CTA2 ~num_cols:256 ~num_rows:128 in
  Alcotest.(check int) "cols" 256 t.Tmem.num_cols

let test_make_invalid_rows () =
  Alcotest.check_raises "rows > 128"
    (Invalid_argument "num_rows must be <= 128")
    (fun () -> ignore (Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:64 ~num_rows:256))

let test_make_invalid_cols () =
  Alcotest.check_raises "cols > 512"
    (Invalid_argument "num_cols must be <= 512")
    (fun () -> ignore (Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:768 ~num_rows:128))

let test_make_invalid_cta1_cols () =
  Alcotest.check_raises "cta1 cols > 256"
    (Invalid_argument "CTA1 num_cols must be <= 256")
    (fun () -> ignore (Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:512 ~num_rows:128))

let test_address_origin () =
  Alcotest.(check int) "0,0" 0 (Tmem.address ~row:0 ~col:0)

let test_address_row () =
  (* row=1, col=0 → (1 lsl 16) lor 0 = 65536 *)
  Alcotest.(check int) "1,0" 65536 (Tmem.address ~row:1 ~col:0)

let test_address_col () =
  (* row=0, col=8 → 8 *)
  Alcotest.(check int) "0,8" 8 (Tmem.address ~row:0 ~col:8)

let test_address_both () =
  (* row=2, col=16 → (2 lsl 16) lor 16 = 131088 *)
  Alcotest.(check int) "2,16" 131088 (Tmem.address ~row:2 ~col:16)

let test_warp_offset_0 () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:128 ~num_rows:128 in
  Alcotest.(check int) "warp 0" 0 (Tmem.warp_row_offset t 0)

let test_warp_offset_1 () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:128 ~num_rows:128 in
  Alcotest.(check int) "warp 1" 32 (Tmem.warp_row_offset t 1)

let test_warp_offset_3 () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:128 ~num_rows:128 in
  Alcotest.(check int) "warp 3" 96 (Tmem.warp_row_offset t 3)

let test_elems_per_thread () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:128 ~num_rows:128 in
  Alcotest.(check int) "8 elems" 8 (Tmem.elems_per_thread_per_load t)

let test_num_loads_64 () =
  (* 64 cols / 8 = 8 loads per warp *)
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:64 ~num_rows:128 in
  Alcotest.(check int) "8 loads" 8 (Tmem.num_loads_per_warp t)

let test_num_loads_128 () =
  (* 128 cols / 8 = 16 loads per warp *)
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:128 ~num_rows:128 in
  Alcotest.(check int) "16 loads" 16 (Tmem.num_loads_per_warp t)

let test_total_elems () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:64 ~num_rows:128 in
  Alcotest.(check int) "8192" 8192 (Tmem.total_elems t)

let test_bytes () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:64 ~num_rows:128 in
  Alcotest.(check int) "32768" 32768 (Tmem.bytes t)

let test_alloc_ptx_cta1 () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:128 ~num_rows:128 in
  let s = Tmem.alloc_ptx t "tmem_addr" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has alloc"       true (contains "tcgen05.alloc" s);
  Alcotest.(check bool) "has cta_group::1" true (contains "cta_group::1" s);
  Alcotest.(check bool) "has tmem_addr"   true (contains "tmem_addr" s);
  Alcotest.(check bool) "has 128"         true (contains "128" s)

let test_alloc_ptx_cta2 () =
  let t = Tmem.make ~cta_group:Tmem.CTA2 ~num_cols:128 ~num_rows:128 in
  let s = Tmem.alloc_ptx t "tmem_addr" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has cta_group::2" true (contains "cta_group::2" s)

let test_dealloc_ptx () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:64 ~num_rows:128 in
  let s = Tmem.dealloc_ptx t "taddr" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has dealloc" true (contains "tcgen05.dealloc" s);
  Alcotest.(check bool) "has taddr"   true (contains "taddr" s);
  Alcotest.(check bool) "has 64"      true (contains "64" s)

let test_commit_ptx () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:64 ~num_rows:128 in
  let s = Tmem.commit_ptx t "mbar" in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has commit"   true (contains "tcgen05.commit" s);
  Alcotest.(check bool) "has mbar"     true (contains "mbar" s);
  Alcotest.(check bool) "has arrive"   true (contains "arrive" s)

let test_ld_ptx () =
  let t = Tmem.make ~cta_group:Tmem.CTA1 ~num_cols:64 ~num_rows:128 in
  let regs = ["r0";"r1";"r2";"r3";"r4";"r5";"r6";"r7"] in
  let s = Tmem.ld_ptx t "taddr" regs 0 in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has tcgen05.ld"  true (contains "tcgen05.ld" s);
  Alcotest.(check bool) "has 32x32b"      true (contains "32x32b" s);
  Alcotest.(check bool) "has taddr"       true (contains "taddr" s)

let () =
  Alcotest.run "Tmem" [
    "make",       [ Alcotest.test_case "cta1"        `Quick test_make_valid_cta1
                  ; Alcotest.test_case "cta2"        `Quick test_make_valid_cta2
                  ; Alcotest.test_case "rows"        `Quick test_make_invalid_rows
                  ; Alcotest.test_case "cols"        `Quick test_make_invalid_cols
                  ; Alcotest.test_case "cta1-cols"   `Quick test_make_invalid_cta1_cols ];
    "address",    [ Alcotest.test_case "origin"  `Quick test_address_origin
                  ; Alcotest.test_case "row"     `Quick test_address_row
                  ; Alcotest.test_case "col"     `Quick test_address_col
                  ; Alcotest.test_case "both"    `Quick test_address_both ];
    "warp_off",   [ Alcotest.test_case "warp0"   `Quick test_warp_offset_0
                  ; Alcotest.test_case "warp1"   `Quick test_warp_offset_1
                  ; Alcotest.test_case "warp3"   `Quick test_warp_offset_3 ];
    "load",       [ Alcotest.test_case "elems"   `Quick test_elems_per_thread
                  ; Alcotest.test_case "loads64" `Quick test_num_loads_64
                  ; Alcotest.test_case "loads128"`Quick test_num_loads_128 ];
    "size",       [ Alcotest.test_case "elems"   `Quick test_total_elems
                  ; Alcotest.test_case "bytes"   `Quick test_bytes ];
    "ptx",        [ Alcotest.test_case "alloc1"  `Quick test_alloc_ptx_cta1
                  ; Alcotest.test_case "alloc2"  `Quick test_alloc_ptx_cta2
                  ; Alcotest.test_case "dealloc" `Quick test_dealloc_ptx
                  ; Alcotest.test_case "commit"  `Quick test_commit_ptx
                  ; Alcotest.test_case "ld"      `Quick test_ld_ptx ];
  ]
