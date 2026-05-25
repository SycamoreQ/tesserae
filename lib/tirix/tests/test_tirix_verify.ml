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
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])



let hopper_kernel () =
  Kernel_ast.make
    ~name:"gemm_hopper"
    ~arch:Kernel_ast.SM90
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 32 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])


let blackwell_kernel () =
  Kernel_ast.make
    ~name:"gemm_blackwell"
    ~arch:Kernel_ast.SM100
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 128; n = 128; k = 64 }
    ~stages:4
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("B", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(Kernel_ast.Seq [])


let mk_var name ty =
  { var_name = name
  ; var_id = 0
  ; var_type = Scalar ty
  ; var_mutable = false
  }

let mk_mut_var name ty =
  { var_name = name
  ; var_id = 0
  ; var_type = Scalar ty
  ; var_mutable = true
  }

let mk_tir_minimal (f:  Kernel_desc.family) =
  { name = "test_kernel"
  ; family = f
  ; params = []
  ; tensors = []
  ; bm = 128
  ; bn = 128
  ; bk = 32
  ; smem_bytes = 0
  ; pipeline_depth = 4
  ; cluster    = Cluster.make { Cluster.x=1; y=1; z=1 } 4
      [ (0, Cluster.Producer); (1, Cluster.Consumer)
      ; (2, Cluster.Epilogue); (3, Cluster.Epilogue) ]
  ; body  = []
  ; helpers = []
  }

let test_lowered_arch arch name =
  let kernel = match arch with
    | Kernel_desc.Ampere -> ampere_kernel ()
    | Kernel_desc.Hopper -> hopper_kernel ()
    | Kernel_desc.Blackwell -> blackwell_kernel ()
  in
  let (Lower.Pack desc) = Lower.lower_exn kernel in
  let tir = Desc_to_tirix.lower desc in
  Alcotest.(check bool) name true (Result.is_ok (Tirix_verify.verify tir))


let test_valid_lowered_ampere () = test_lowered_arch Kernel_desc.Ampere "valid ampere"
let test_valid_lowered_hopper () = test_lowered_arch Kernel_desc.Hopper "valid hopper"
let test_valid_lowered_blackwell () = test_lowered_arch Kernel_desc.Blackwell "valid blackwell"

let test_empty_body_for_arch arch name =
  let tir = mk_tir_minimal arch in
  Alcotest.(check bool) name true (Result.is_ok (Tirix_verify.verify tir))

let test_valid_empty_body () =
  test_empty_body_for_arch Kernel_desc.Ampere "empty body ok (Ampere)";
  test_empty_body_for_arch Kernel_desc.Hopper "empty body ok (Hopper)";
  test_empty_body_for_arch Kernel_desc.Blackwell "empty body ok (Blackwell)"

let test_slet_use_for_arch arch name =
  let v = mk_mut_var "x" S32 in
  let tir = { (mk_tir_minimal arch) with body = [
    SLet (v, Expr (Const (S32, 0l)));
    SAssign (v, Expr (Var v));
  ]} in
  Alcotest.(check bool) name true (Result.is_ok (Tirix_verify.verify tir))

let test_valid_slet_then_use () =
  test_slet_use_for_arch Kernel_desc.Ampere "let then use ok (Ampere)";
  test_slet_use_for_arch Kernel_desc.Hopper "let then use ok (Hopper)";
  test_slet_use_for_arch Kernel_desc.Blackwell "let then use ok (Blackwell)"

let test_pipeline_stages_for_arch arch name =
  let tir = { (mk_tir_minimal arch) with body = [
    SPipeline {
      stages   = 4;
      prologue = [];
      mainloop = [];
      epilogue = [];
    }
  ]} in
  Alcotest.(check bool) name true
    (Result.is_ok (Tirix_verify.verify tir))

let test_valid_pipeline_stages () =
  test_pipeline_stages_for_arch Kernel_desc.Ampere "pipeline 4 stages ok (Ampere)";
  test_pipeline_stages_for_arch Kernel_desc.Hopper "pipeline 4 stages ok (Hopper)";
  test_pipeline_stages_for_arch Kernel_desc.Blackwell "pipeline 4 stages ok (Blackwell)"

let test_warp_group_roles_for_arch arch name =
  let kernel = match arch with
    | Kernel_desc.Ampere -> ampere_kernel ()
    | Kernel_desc.Hopper -> hopper_kernel ()
    | Kernel_desc.Blackwell -> blackwell_kernel ()
  in
  let (Lower.Pack desc) = Lower.lower_exn kernel in
  let tir = Desc_to_tirix.lower desc in
  Alcotest.(check bool) name true
    (Result.is_ok (Tirix_verify.verify tir))

let test_valid_warp_group_roles () =
  test_warp_group_roles_for_arch Kernel_desc.Ampere    "warp roles consistent (Ampere)";
  test_warp_group_roles_for_arch Kernel_desc.Hopper    "warp roles consistent (Hopper)";
  test_warp_group_roles_for_arch Kernel_desc.Blackwell "warp roles consistent (Blackwell)"

let test_undefined_var_assign_for_arch arch name =
  let v = mk_var "ghost" S32 in
  let tir = { (mk_tir_minimal arch) with body = [
    SAssign (v, Expr (Const (S32, 0l)));
  ]} in
  Alcotest.(check bool) name true
    (Result.is_error (Tirix_verify.verify tir))

let test_undefined_var_in_assign () =
  test_undefined_var_assign_for_arch Kernel_desc.Ampere    "undefined var caught (Ampere)";
  test_undefined_var_assign_for_arch Kernel_desc.Hopper    "undefined var caught (Hopper)";
  test_undefined_var_assign_for_arch Kernel_desc.Blackwell "undefined var caught (Blackwell)"

let test_undefined_var_expr_for_arch arch name =
  let v   = mk_var "declared" S32 in
  let bad = mk_var "undeclared" S32 in
  let tir = { (mk_tir_minimal arch) with body = [
    SLet (v, Expr (Var bad));
  ]} in
  Alcotest.(check bool) name true
    (Result.is_error (Tirix_verify.verify tir))

let test_undefined_var_in_expr () =
  test_undefined_var_expr_for_arch Kernel_desc.Ampere    "undefined in expr caught (Ampere)";
  test_undefined_var_expr_for_arch Kernel_desc.Hopper    "undefined in expr caught (Hopper)";
  test_undefined_var_expr_for_arch Kernel_desc.Blackwell "undefined in expr caught (Blackwell)"

let test_use_before_declare_for_arch arch name =
  let v = mk_var "late" S32 in
  let tir = { (mk_tir_minimal arch) with body = [
    SAssign (v, Expr (Const (S32, 1l)));
    SLet (v, Expr (Const (S32, 0l)));
  ]} in
  Alcotest.(check bool) name true
    (Result.is_error (Tirix_verify.verify tir))

let test_use_before_declare () =
  test_use_before_declare_for_arch Kernel_desc.Ampere    "use before declare caught (Ampere)";
  test_use_before_declare_for_arch Kernel_desc.Hopper    "use before declare caught (Hopper)";
  test_use_before_declare_for_arch Kernel_desc.Blackwell "use before declare caught (Blackwell)"

let test_pipeline_zero_stages_for_arch arch name =
  let tir = { (mk_tir_minimal arch) with body = [
    SPipeline { stages = 0; prologue = []; mainloop = []; epilogue = [] }
  ]} in
  Alcotest.(check bool) name true
    (Result.is_error (Tirix_verify.verify tir))

let test_pipeline_zero_stages () =
  test_pipeline_zero_stages_for_arch Kernel_desc.Ampere    "zero stages caught (Ampere)";
  test_pipeline_zero_stages_for_arch Kernel_desc.Hopper    "zero stages caught (Hopper)";
  test_pipeline_zero_stages_for_arch Kernel_desc.Blackwell "zero stages caught (Blackwell)"

let test_pipeline_negative_stages_for_arch arch name =
  let tir = { (mk_tir_minimal arch) with body = [
    SPipeline { stages = -1; prologue = []; mainloop = []; epilogue = [] }
  ]} in
  Alcotest.(check bool) name true
    (Result.is_error (Tirix_verify.verify tir))

let test_pipeline_negative_stages () =
  test_pipeline_negative_stages_for_arch Kernel_desc.Ampere    "negative stages caught (Ampere)";
  test_pipeline_negative_stages_for_arch Kernel_desc.Hopper    "negative stages caught (Hopper)";
  test_pipeline_negative_stages_for_arch Kernel_desc.Blackwell "negative stages caught (Blackwell)"

let test_for_zero_step_for_arch arch name =
  let v = mk_var "i" S32 in
  let tir = { (mk_tir_minimal arch) with body = [
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
  Alcotest.(check bool) name true
    (Result.is_error (Tirix_verify.verify tir))

let test_for_zero_step () =
  test_for_zero_step_for_arch Kernel_desc.Ampere    "zero step caught (Ampere)";
  test_for_zero_step_for_arch Kernel_desc.Hopper    "zero step caught (Hopper)";
  test_for_zero_step_for_arch Kernel_desc.Blackwell "zero step caught (Blackwell)"

let test_copy_unknown_src_for_arch arch name =
  let ghost = Tensor {
    tensor_name      = "ghost_tensor";
    tensor_id        = Type_id.create ();
    tensor_elem_type = Elemtype.Float16;
    tensor_memspace  = Memspace.Global;
    tensor_layout    = Layout.make (Modes.Int 1) (Modes.Int 1);
    tensor_swizzle   = Swizzle.make 0 0 0;
  } in
  let dst = Tensor {
    tensor_name = "smem_A";
    tensor_id = Type_id.create ();
    tensor_elem_type = Elemtype.Float16;
    tensor_memspace  = Memspace.Shared;
    tensor_layout    = Layout.make (Modes.Int 1) (Modes.Int 1);
    tensor_swizzle   = Swizzle.make 0 0 0;
  } in
  let tir = { (mk_tir_minimal arch) with body = [
    SOp (Copy {
      copy_kind  = CpAsync;
      src_tensor = ghost;
      dst_tensor = dst;
      pred_expr  = None;
      mbar_var   = None;
    })
  ]} in
  Alcotest.(check bool) name true
    (Result.is_error (Tirix_verify.verify tir))

let test_copy_unknown_src_tensor () =
  test_copy_unknown_src_for_arch Kernel_desc.Ampere    "unknown src tensor caught (Ampere)";
  test_copy_unknown_src_for_arch Kernel_desc.Hopper    "unknown src tensor caught (Hopper)";
  test_copy_unknown_src_for_arch Kernel_desc.Blackwell "unknown src tensor caught (Blackwell)"

let test_scheduler_warp_for_arch arch name =
  let tir = { (mk_tir_minimal arch) with body = [
    SWarpGroup (Cluster.Scheduler, [SEmpty])
  ]} in
  Alcotest.(check bool) name true
    (Result.is_error (Tirix_verify.verify tir))

let test_scheduler_warp_no_scheduler_role () =
  test_scheduler_warp_for_arch Kernel_desc.Ampere    "scheduler warp without role caught (Ampere)";
  test_scheduler_warp_for_arch Kernel_desc.Hopper    "scheduler warp without role caught (Hopper)";
  test_scheduler_warp_for_arch Kernel_desc.Blackwell "scheduler warp without role caught (Blackwell)"

let test_multiple_errors_for_arch arch name =
  let v1 = mk_var "ghost1" S32 in
  let v2 = mk_var "ghost2" S32 in
  let tir = { (mk_tir_minimal arch) with body = [
    SAssign (v1, Expr (Const (S32, 0l)));
    SAssign (v2, Expr (Const (S32, 1l)));
    SPipeline { stages = 0; prologue = []; mainloop = []; epilogue = [] };
  ]} in
  match Tirix_verify.verify tir with
  | Ok _ -> Alcotest.fail ("expected errors for " ^ name)
  | Error errs ->
    Alcotest.(check bool) name true
      (List.length errs >= 2)

let test_multiple_errors_reported () =
  test_multiple_errors_for_arch Kernel_desc.Ampere    "multiple errors (Ampere)";
  test_multiple_errors_for_arch Kernel_desc.Hopper    "multiple errors (Hopper)";
  test_multiple_errors_for_arch Kernel_desc.Blackwell "multiple errors (Blackwell)"


let () =
  Alcotest.run "Tirix_verify" [
    "valid",    [ Alcotest.test_case "lowered-ampere"    `Quick test_valid_lowered_ampere
                ; Alcotest.test_case "lowered-hopper"    `Quick test_valid_lowered_hopper
                ; Alcotest.test_case "lowered-blackwell" `Quick test_valid_lowered_blackwell
                ; Alcotest.test_case "empty"             `Quick test_valid_empty_body
                ; Alcotest.test_case "let-use"           `Quick test_valid_slet_then_use
                ; Alcotest.test_case "pipeline"          `Quick test_valid_pipeline_stages
                ; Alcotest.test_case "warp-roles"        `Quick test_valid_warp_group_roles ];
    "undef",    [ Alcotest.test_case "assign"            `Quick test_undefined_var_in_assign
                ; Alcotest.test_case "expr"              `Quick test_undefined_var_in_expr
                ; Alcotest.test_case "use-before"        `Quick test_use_before_declare ];
    "pipeline", [ Alcotest.test_case "zero-stages"       `Quick test_pipeline_zero_stages
                ; Alcotest.test_case "neg-stages"        `Quick test_pipeline_negative_stages ];
    "for",      [ Alcotest.test_case "zero-step"         `Quick test_for_zero_step ];
    "tensor",   [ Alcotest.test_case "unknown-src"       `Quick test_copy_unknown_src_tensor ];
    "warp",     [ Alcotest.test_case "no-scheduler"      `Quick test_scheduler_warp_no_scheduler_role ];
    "errors",   [ Alcotest.test_case "accumulate"        `Quick test_multiple_errors_reported ];
  ]
