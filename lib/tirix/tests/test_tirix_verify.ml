open Tesserae_core
open Tesserae_atoms
open Tesserae_pipeline
open Tesserae_kernel
open Tesserae_tirix
open Tirix

(* ------------------------------------------------------------------ *)
(* helpers                                                             *)
(* ------------------------------------------------------------------ *)

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

let lower_ampere () =
  let (Lower.Pack desc) = Lower.lower_exn (ampere_kernel ()) in
  Kernel_desc_to_tir.lower desc

let mk_var name ty =
  { var_name    = name
  ; var_id      = 0
  ; var_type    = Scalar ty
  ; var_mutable = false
  }

let mk_tir_minimal () =
  { name       = "test_kernel"
  ; family     = Kernel_desc.Ampere
  ; params     = []
  ; tensors    = []
  ; smem_bytes = 0
  ; cluster    = Cluster.make { Cluster.x=1; y=1; z=1 } 4
      [ (0, Cluster.Producer); (1, Cluster.Consumer)
      ; (2, Cluster.Epilogue); (3, Cluster.Epilogue) ]
  ; body       = []
  ; helpers    = []
  }

(* ------------------------------------------------------------------ *)
(* valid IR — should pass                                              *)
(* ------------------------------------------------------------------ *)

let test_valid_lowered () =
  let tir = lower_ampere () in
  Alcotest.(check bool) "valid ampere" true
    (Result.is_ok (Tir_verify.verify tir))

let test_valid_empty_body () =
  let tir = mk_tir_minimal () in
  Alcotest.(check bool) "empty body ok" true
    (Result.is_ok (Tir_verify.verify tir))

let test_valid_slet_then_use () =
  let v = mk_var "x" S32 in
  let tir = { (mk_tir_minimal ()) with body = [
    SLet (v, Expr (Const (S32, 0l)));
    SAssign (v, Expr (Var v));
  ]} in
  Alcotest.(check bool) "let then use ok" true
    (Result.is_ok (Tir_verify.verify tir))

let test_valid_pipeline_stages () =
  let tir = { (mk_tir_minimal ()) with body = [
    SPipeline {
      stages   = 4;
      prologue = [];
      mainloop = [];
      epilogue = [];
    }
  ]} in
  Alcotest.(check bool) "pipeline 4 stages ok" true
    (Result.is_ok (Tir_verify.verify tir))

let test_valid_warp_group_roles () =
  let tir = lower_ampere () in
  Alcotest.(check bool) "warp roles consistent" true
    (Result.is_ok (Tir_verify.verify tir))

(* ------------------------------------------------------------------ *)
(* undefined variable — should fail                                    *)
(* ------------------------------------------------------------------ *)

let test_undefined_var_in_assign () =
  let v = mk_var "ghost" S32 in
  let tir = { (mk_tir_minimal ()) with body = [
    SAssign (v, Expr (Const (S32, 0l)));
  ]} in
  Alcotest.(check bool) "undefined var caught" true
    (Result.is_error (Tir_verify.verify tir))

let test_undefined_var_in_expr () =
  let v   = mk_var "declared" S32 in
  let bad = mk_var "undeclared" S32 in
  let tir = { (mk_tir_minimal ()) with body = [
    SLet (v, Expr (Var bad));
  ]} in
  Alcotest.(check bool) "undefined in expr caught" true
    (Result.is_error (Tir_verify.verify tir))

let test_use_before_declare () =
  let v = mk_var "late" S32 in
  let tir = { (mk_tir_minimal ()) with body = [
    SAssign (v, Expr (Const (S32, 1l)));
    SLet (v, Expr (Const (S32, 0l)));
  ]} in
  Alcotest.(check bool) "use before declare caught" true
    (Result.is_error (Tir_verify.verify tir))

(* ------------------------------------------------------------------ *)
(* pipeline structural errors — should fail                            *)
(* ------------------------------------------------------------------ *)

let test_pipeline_zero_stages () =
  let tir = { (mk_tir_minimal ()) with body = [
    SPipeline { stages = 0; prologue = []; mainloop = []; epilogue = [] }
  ]} in
  Alcotest.(check bool) "zero stages caught" true
    (Result.is_error (Tir_verify.verify tir))

let test_pipeline_negative_stages () =
  let tir = { (mk_tir_minimal ()) with body = [
    SPipeline { stages = -1; prologue = []; mainloop = []; epilogue = [] }
  ]} in
  Alcotest.(check bool) "negative stages caught" true
    (Result.is_error (Tir_verify.verify tir))

(* ------------------------------------------------------------------ *)
(* for loop structural errors — should fail                            *)
(* ------------------------------------------------------------------ *)

let test_for_zero_step () =
  let v = mk_var "i" S32 in
  let tir = { (mk_tir_minimal ()) with body = [
    SLet (v, Expr (Const (S32, 0l)));
    SFor {
      var    = v;
      start  = Const (S32, 0l);
      stop   = Const (S32, 10l);
      step   = Const (S32, 0l);
      dir    = Upto;
      unroll = false;
      body   = [];
    }
  ]} in
  Alcotest.(check bool) "zero step caught" true
    (Result.is_error (Tir_verify.verify tir))

(* ------------------------------------------------------------------ *)
(* tensor reference errors — should fail                               *)
(* ------------------------------------------------------------------ *)

let test_copy_unknown_src_tensor () =
  let ghost = Tensor {
    tensor_name      = "ghost_tensor";
    tensor_id        = Type_id.create ();
    tensor_elem_type = Elemtype.Float16;
    tensor_memspace  = Memspace.Global;
    tensor_layout    = Layout.make (Modes.Int 1) (Modes.Int 1);
    tensor_swizzle   = Swizzle.make 0 0 0;
  } in
  let dst = Tensor {
    tensor_name      = "smem_A";
    tensor_id        = Type_id.create ();
    tensor_elem_type = Elemtype.Float16;
    tensor_memspace  = Memspace.Shared;
    tensor_layout    = Layout.make (Modes.Int 1) (Modes.Int 1);
    tensor_swizzle   = Swizzle.make 0 0 0;
  } in
  let tir = { (mk_tir_minimal ()) with body = [
    SOp (Copy {
      copy_kind  = CpAsync;
      src_tensor = ghost;
      dst_tensor = dst;
      pred_expr  = None;
      mbar_var   = None;
    })
  ]} in
  Alcotest.(check bool) "unknown src tensor caught" true
    (Result.is_error (Tir_verify.verify tir))

(* ------------------------------------------------------------------ *)
(* warp role errors — should fail                                      *)
(* ------------------------------------------------------------------ *)

let test_scheduler_warp_no_scheduler_role () =
  let tir = { (mk_tir_minimal ()) with body = [
    SWarpGroup (Cluster.Scheduler, [SEmpty])
  ]} in
  Alcotest.(check bool) "scheduler warp without role caught" true
    (Result.is_error (Tir_verify.verify tir))

(* ------------------------------------------------------------------ *)
(* error accumulation — all errors reported                            *)
(* ------------------------------------------------------------------ *)

let test_multiple_errors_reported () =
  let v1 = mk_var "ghost1" S32 in
  let v2 = mk_var "ghost2" S32 in
  let tir = { (mk_tir_minimal ()) with body = [
    SAssign (v1, Expr (Const (S32, 0l)));
    SAssign (v2, Expr (Const (S32, 1l)));
    SPipeline { stages = 0; prologue = []; mainloop = []; epilogue = [] };
  ]} in
  match Tir_verify.verify tir with
  | Ok _ -> Alcotest.fail "expected errors"
  | Error errs ->
    Alcotest.(check bool) "multiple errors" true
      (List.length errs >= 2)

(* ------------------------------------------------------------------ *)
(* runner                                                              *)
(* ------------------------------------------------------------------ *)

let () =
  Alcotest.run "Tir_verify" [
    "valid",    [ Alcotest.test_case "lowered"        `Quick test_valid_lowered
                ; Alcotest.test_case "empty"          `Quick test_valid_empty_body
                ; Alcotest.test_case "let-use"        `Quick test_valid_slet_then_use
                ; Alcotest.test_case "pipeline"       `Quick test_valid_pipeline_stages
                ; Alcotest.test_case "warp-roles"     `Quick test_valid_warp_group_roles ];
    "undef",    [ Alcotest.test_case "assign"         `Quick test_undefined_var_in_assign
                ; Alcotest.test_case "expr"           `Quick test_undefined_var_in_expr
                ; Alcotest.test_case "use-before"     `Quick test_use_before_declare ];
    "pipeline", [ Alcotest.test_case "zero-stages"   `Quick test_pipeline_zero_stages
                ; Alcotest.test_case "neg-stages"     `Quick test_pipeline_negative_stages ];
    "for",      [ Alcotest.test_case "zero-step"      `Quick test_for_zero_step ];
    "tensor",   [ Alcotest.test_case "unknown-src"    `Quick test_copy_unknown_src_tensor ];
    "warp",     [ Alcotest.test_case "no-scheduler"   `Quick test_scheduler_warp_no_scheduler_role ];
    "errors",   [ Alcotest.test_case "accumulate"     `Quick test_multiple_errors_reported ];
  ]
