open Tesserae

(* Standard 2SM warp-specialized cluster:
   6 warps: 0=producer, 1=consumer, 2-4=epilogue, 5=scheduler *)
let two_sm () =
  Cluster.make
    { Cluster.x = 2; y = 1; z = 1 }
    6
    [ (0, Cluster.Producer)
    ; (1, Cluster.Consumer)
    ; (2, Cluster.Epilogue)
    ; (3, Cluster.Epilogue)
    ; (4, Cluster.Epilogue)
    ; (5, Cluster.Scheduler) ]

let single_sm () =
  Cluster.make
    { Cluster.x = 1; y = 1; z = 1 }
    4
    [ (0, Cluster.Producer)
    ; (1, Cluster.Consumer)
    ; (2, Cluster.Epilogue)
    ; (3, Cluster.Epilogue) ]

let test_make_valid () =
  let t = two_sm () in
  Alcotest.(check int) "warps" 6 t.Cluster.num_warps

let test_make_invalid_cta_count () =
  Alcotest.check_raises "too many CTAs"
    (Invalid_argument "cluster CTA count must be <= 8")
    (fun () ->
       ignore (Cluster.make
         { Cluster.x = 3; y = 3; z = 2 }
         4 []))

let test_make_invalid_warp_id () =
  Alcotest.check_raises "warp_id out of range"
    (Invalid_argument "warp_id out of range")
    (fun () ->
       ignore (Cluster.make
         { Cluster.x = 1; y = 1; z = 1 }
         4
         [(5, Cluster.Producer)]))

(* ------------------------------------------------------------------ *)
(* cta_count / is_2sm                                                  *)
(* ------------------------------------------------------------------ *)

let test_cta_count_2sm () =
  Alcotest.(check int) "2" 2 (Cluster.cta_count (two_sm ()))

let test_cta_count_1sm () =
  Alcotest.(check int) "1" 1 (Cluster.cta_count (single_sm ()))

let test_is_2sm_true () =
  Alcotest.(check bool) "2sm" true (Cluster.is_2sm (two_sm ()))

let test_is_2sm_false () =
  Alcotest.(check bool) "1sm" false (Cluster.is_2sm (single_sm ()))

let test_producer_warp () =
  Alcotest.(check (option int)) "producer" (Some 0)
    (Cluster.producer_warp (two_sm ()))

let test_consumer_warp () =
  Alcotest.(check (option int)) "consumer" (Some 1)
    (Cluster.consumer_warp (two_sm ()))

let test_epilogue_warps () =
  Alcotest.(check (list int)) "epilogue" [2; 3; 4]
    (Cluster.epilogue_warps (two_sm ()))

let test_scheduler_warp () =
  Alcotest.(check (option int)) "scheduler" (Some 5)
    (Cluster.scheduler_warp (two_sm ()))

let test_scheduler_none () =
  Alcotest.(check (option int)) "no scheduler" None
    (Cluster.scheduler_warp (single_sm ()))

let test_thread_count_2sm () =
  Alcotest.(check int) "192" 192 (Cluster.thread_count (two_sm ()))

let test_thread_count_1sm () =
  Alcotest.(check int) "128" 128 (Cluster.thread_count (single_sm ()))

let test_cluster_arrive () =
  Alcotest.(check string) "arrive"
    "barrier.cluster.arrive.release.aligned;"
    (Cluster.cluster_arrive_ptx ())

let test_cluster_wait () =
  Alcotest.(check string) "wait"
    "barrier.cluster.wait.acquire.aligned;"
    (Cluster.cluster_wait_ptx ())

let test_cluster_ctaid () =
  Alcotest.(check string) "ctaid"
    "mov.u32 rank, %cluster_ctaid.x;"
    (Cluster.cluster_ctaid_ptx "rank")

let test_mapa () =
  Alcotest.(check string) "mapa"
    "mapa.shared::cluster.u32 dst, src, 0;"
    (Cluster.mapa_ptx "dst" "src" "0")

let test_mbarrier_cluster () =
  Alcotest.(check string) "mbar cluster"
    "mbarrier.arrive.expect_tx.shared::cluster.b64 [mbar], 256;"
    (Cluster.mbarrier_arrive_expect_cluster_ptx "mbar" 256)

let test_emit_cluster_attr () =
  Alcotest.(check string) "attr"
    "__cluster_dims__(2, 1, 1)"
    (Cluster.emit_cluster_attr (two_sm ()))

let test_emit_smem_mbar () =
  Alcotest.(check string) "smem mbar"
    "__shared__ __align__(8) uint64_t full_mbar[4];"
    (Cluster.emit_smem_mbar "full_mbar" 4)

let () =
  Alcotest.run "Cluster" [
    "make",      [ Alcotest.test_case "valid"    `Quick test_make_valid
                 ; Alcotest.test_case "cta-lim"  `Quick test_make_invalid_cta_count
                 ; Alcotest.test_case "warp-id"  `Quick test_make_invalid_warp_id ];
    "cta",       [ Alcotest.test_case "count-2"  `Quick test_cta_count_2sm
                 ; Alcotest.test_case "count-1"  `Quick test_cta_count_1sm
                 ; Alcotest.test_case "is-2sm"   `Quick test_is_2sm_true
                 ; Alcotest.test_case "is-1sm"   `Quick test_is_2sm_false ];
    "roles",     [ Alcotest.test_case "producer" `Quick test_producer_warp
                 ; Alcotest.test_case "consumer" `Quick test_consumer_warp
                 ; Alcotest.test_case "epilogue" `Quick test_epilogue_warps
                 ; Alcotest.test_case "sched"    `Quick test_scheduler_warp
                 ; Alcotest.test_case "no-sched" `Quick test_scheduler_none ];
    "threads",   [ Alcotest.test_case "2sm"  `Quick test_thread_count_2sm
                 ; Alcotest.test_case "1sm"  `Quick test_thread_count_1sm ];
    "ptx",       [ Alcotest.test_case "arrive"  `Quick test_cluster_arrive
                 ; Alcotest.test_case "wait"    `Quick test_cluster_wait
                 ; Alcotest.test_case "ctaid"   `Quick test_cluster_ctaid
                 ; Alcotest.test_case "mapa"    `Quick test_mapa
                 ; Alcotest.test_case "mbar"    `Quick test_mbarrier_cluster
                 ; Alcotest.test_case "attr"    `Quick test_emit_cluster_attr
                 ; Alcotest.test_case "smem"    `Quick test_emit_smem_mbar ];
  ]
