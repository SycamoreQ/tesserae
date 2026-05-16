open Tesserae

(* --- name --- *)
let test_name_global () =
  Alcotest.(check string) "global" "global" (Memspace.name Memspace.Global)

let test_name_shared () =
  Alcotest.(check string) "shared" "shared" (Memspace.name Memspace.Shared)

let test_name_register () =
  Alcotest.(check string) "register" "register" (Memspace.name Memspace.Register)

(* --- can_transfer --- *)
let test_transfer_global_shared () =
  Alcotest.(check bool) "global->shared" true
    (Memspace.can_transfer ~src:Memspace.Global ~dst:Memspace.Shared)

let test_transfer_global_register () =
  Alcotest.(check bool) "global->register" true
    (Memspace.can_transfer ~src:Memspace.Global ~dst:Memspace.Register)

let test_transfer_shared_register () =
  Alcotest.(check bool) "shared->register" true
    (Memspace.can_transfer ~src:Memspace.Shared ~dst:Memspace.Register)

let test_transfer_register_shared () =
  Alcotest.(check bool) "register->shared" true
    (Memspace.can_transfer ~src:Memspace.Register ~dst:Memspace.Shared)

let test_transfer_register_global () =
  Alcotest.(check bool) "register->global" true
    (Memspace.can_transfer ~src:Memspace.Register ~dst:Memspace.Global)

let test_transfer_shared_global () =
  Alcotest.(check bool) "shared->global invalid" false
    (Memspace.can_transfer ~src:Memspace.Shared ~dst:Memspace.Global)

let test_transfer_same_global () =
  Alcotest.(check bool) "global->global" true
    (Memspace.can_transfer ~src:Memspace.Global ~dst:Memspace.Global)

let test_transfer_same_shared () =
  Alcotest.(check bool) "shared->shared" true
    (Memspace.can_transfer ~src:Memspace.Shared ~dst:Memspace.Shared)

let test_transfer_same_register () =
  Alcotest.(check bool) "register->register" true
    (Memspace.can_transfer ~src:Memspace.Register ~dst:Memspace.Register)

(* --- pp --- *)
let test_pp_global () =
  let s = Stdlib.Format.asprintf "%a" Memspace.pp Memspace.Global in
  Alcotest.(check string) "pp global" "global" s

(* --- runner --- *)
let () =
  Alcotest.run "Memspace" [
    "name",     [ Alcotest.test_case "global"   `Quick test_name_global
                ; Alcotest.test_case "shared"   `Quick test_name_shared
                ; Alcotest.test_case "register" `Quick test_name_register ];
    "transfer", [ Alcotest.test_case "g->s"   `Quick test_transfer_global_shared
                ; Alcotest.test_case "g->r"   `Quick test_transfer_global_register
                ; Alcotest.test_case "s->r"   `Quick test_transfer_shared_register
                ; Alcotest.test_case "r->s"   `Quick test_transfer_register_shared
                ; Alcotest.test_case "r->g"   `Quick test_transfer_register_global
                ; Alcotest.test_case "s->g"   `Quick test_transfer_shared_global
                ; Alcotest.test_case "g->g"   `Quick test_transfer_same_global
                ; Alcotest.test_case "s->s"   `Quick test_transfer_same_shared
                ; Alcotest.test_case "r->r"   `Quick test_transfer_same_register ];
    "pp",       [ Alcotest.test_case "global"   `Quick test_pp_global ];
  ]
