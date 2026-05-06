open Tesserae

let i n    = Modes.Int n
let tup ts = Modes.Tuple ts
let lay s d = Layout.make s d

(* ------------------------------------------------------------------ *)
(* make / validation                                                   *)
(* ------------------------------------------------------------------ *)

let test_make_valid () =
  (* layout rank 2, bounds has 2 entries — valid *)
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [6; 6] in
  Alcotest.(check int) "rank" 2 (Layout.rank p.Predicate.layout)

let test_make_invalid () =
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  Alcotest.check_raises "bounds mismatch"
    (Invalid_argument "bounds length must match layout rank")
    (fun () -> ignore (Predicate.make l [6]))

(* ------------------------------------------------------------------ *)
(* is_in_bounds                                                        *)
(* ------------------------------------------------------------------ *)

let test_in_bounds_true () =
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [6; 6] in
  Alcotest.(check bool) "in bounds" true
    (Predicate.is_in_bounds p [3; 5])

let test_in_bounds_false_row () =
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [3; 6] in
  (* row coord 3 >= bound 3 → out of bounds *)
  Alcotest.(check bool) "out of bounds row" false
    (Predicate.is_in_bounds p [3; 2])

let test_in_bounds_false_col () =
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [6; 2] in
  Alcotest.(check bool) "out of bounds col" false
    (Predicate.is_in_bounds p [0; 3])

let test_in_bounds_zero () =
  let l = lay (i 8) (i 1) in
  let p = Predicate.make l [5] in
  Alcotest.(check bool) "coord 4 in bounds" true
    (Predicate.is_in_bounds p [4]);
  Alcotest.(check bool) "coord 5 out" false
    (Predicate.is_in_bounds p [5])

(* ------------------------------------------------------------------ *)
(* count_valid                                                         *)
(* ------------------------------------------------------------------ *)

let test_count_valid_full () =
  (* layout 4x4, bounds 4x4 — all 16 valid *)
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [4; 4] in
  Alcotest.(check int) "all valid" 16 (Predicate.count_valid p)

let test_count_valid_partial () =
  (* layout 4x4, bounds 3x3 — 9 valid *)
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [3; 3] in
  Alcotest.(check int) "9 valid" 9 (Predicate.count_valid p)

let test_count_valid_1d () =
  (* layout 8:1, bounds 5 — 5 valid *)
  let l = lay (i 8) (i 1) in
  let p = Predicate.make l [5] in
  Alcotest.(check int) "5 valid" 5 (Predicate.count_valid p)

(* ------------------------------------------------------------------ *)
(* needs_predication                                                   *)
(* ------------------------------------------------------------------ *)

let test_needs_predication_false () =
  (* bounds exactly divide shape — no predication needed *)
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [4; 4] in
  Alcotest.(check bool) "no pred" false (Predicate.needs_predication p)

let test_needs_predication_true () =
  (* bounds don't divide — predication needed *)
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [3; 4] in
  Alcotest.(check bool) "needs pred" true (Predicate.needs_predication p)

(* ------------------------------------------------------------------ *)
(* residue                                                             *)
(* ------------------------------------------------------------------ *)

let test_residue_zero () =
  (* 8 mod 4 = 0 — full tile *)
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [8; 8] in
  Alcotest.(check (list int)) "no residue" [0; 0]
    (Predicate.residue p)

let test_residue_partial () =
  (* bounds 6, shape 4: 6 mod 4 = 2 *)
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [6; 7] in
  Alcotest.(check (list int)) "residue" [2; 3]
    (Predicate.residue p)

(* ------------------------------------------------------------------ *)
(* emit_predicate_check                                                *)
(* ------------------------------------------------------------------ *)

let test_emit_check_1d () =
  let l = lay (i 8) (i 1) in
  let p = Predicate.make l [5] in
  Alcotest.(check string) "1d check"
    "get<0>(coord) < 5"
    (Predicate.emit_predicate_check p "coord")

let test_emit_check_2d () =
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let p = Predicate.make l [6; 8] in
  Alcotest.(check string) "2d check"
    "get<0>(coord) < 6 && get<1>(coord) < 8"
    (Predicate.emit_predicate_check p "coord")

(* ------------------------------------------------------------------ *)
(* emit_identity_tensor                                                *)
(* ------------------------------------------------------------------ *)

let test_emit_identity () =
  let l = lay (tup [i 4; i 4]) (tup [i 1; i 4]) in
  let s = Predicate.emit_identity_tensor "cta_coord" l in
  let contains sub str =
    let n = String.length sub and m = String.length str in
    let found = ref false in
    for i = 0 to m - n do
      if String.sub str i n = sub then found := true
    done; !found
  in
  Alcotest.(check bool) "has make_identity_tensor" true
    (contains "make_identity_tensor" s);
  Alcotest.(check bool) "has var name" true
    (contains "cta_coord" s)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Predicate" [
    "make",       [ Alcotest.test_case "valid"   `Quick test_make_valid
                  ; Alcotest.test_case "invalid" `Quick test_make_invalid ];
    "in_bounds",  [ Alcotest.test_case "true"    `Quick test_in_bounds_true
                  ; Alcotest.test_case "false-r" `Quick test_in_bounds_false_row
                  ; Alcotest.test_case "false-c" `Quick test_in_bounds_false_col
                  ; Alcotest.test_case "zero"    `Quick test_in_bounds_zero ];
    "count",      [ Alcotest.test_case "full"    `Quick test_count_valid_full
                  ; Alcotest.test_case "partial" `Quick test_count_valid_partial
                  ; Alcotest.test_case "1d"      `Quick test_count_valid_1d ];
    "needs_pred", [ Alcotest.test_case "false"   `Quick test_needs_predication_false
                  ; Alcotest.test_case "true"    `Quick test_needs_predication_true ];
    "residue",    [ Alcotest.test_case "zero"    `Quick test_residue_zero
                  ; Alcotest.test_case "partial" `Quick test_residue_partial ];
    "emit_check", [ Alcotest.test_case "1d"      `Quick test_emit_check_1d
                  ; Alcotest.test_case "2d"      `Quick test_emit_check_2d ];
    "emit_ident", [ Alcotest.test_case "basic"   `Quick test_emit_identity ];
  ]
