open Base
open Tesserae_tirix
open Tirix
open Tesserae_pipeline
open Tesserae_core
open Tesserae_kernel
open Tesserae_backend

let tirix_is_tma (k : tirix) =
  List.exists k.params ~f:(fun p -> p.param_is_tma)

let emit_scalar_ty : type a. a scalar_ty -> string = function
  | U8 ->  "uint8_t"
  | U32 -> "uint32_t"
  | S32 -> "int32_t"
  | U64 -> "uint64_t"
  | F16 -> "__half"
  | F32 ->  "float"
  | BF16 -> "__nv_bfloat16"
  | Bool -> "bool"
  | Ptr ->  "uint64_t"

let emit_packed_scalar (Scalar s) = emit_scalar_ty s

let rec emit_expr : type a. a expr -> string = function
  | Const (U8,   v) -> string_of_int v
  | Const (U32,  v) -> Printf.sprintf "%luU" v
  | Const (S32,  v) -> Printf.sprintf "%ld" v
  | Const (U64,  v) -> Printf.sprintf "%LuULL" v
  | Const (F16,  v) -> Printf.sprintf "__float2half(%ff)" v
  | Const (F32,  v) -> Printf.sprintf "%ff" v
  | Const (BF16, v) -> Printf.sprintf "__float2bfloat16(%ff)" v
  | Const (Bool, v) -> if v then "true" else "false"
  | Const (Ptr,  v) -> Printf.sprintf "0x%LxULL" v
  | Cast (ty, e) ->
    Printf.sprintf "((%s)(%s))" (emit_scalar_ty ty) (emit_expr e)
  | Var v -> v.var_name
  | Builtin b -> emit_builtin b
  | Binop (op, l, r) ->
    Printf.sprintf "(%s %s %s)" (emit_expr l) (emit_binop op) (emit_expr r)
  | Unop (op, e) ->
    Printf.sprintf "(%s%s)" (emit_unop op) (emit_expr e)
  | AddrConv (kind, e) ->
    Printf.sprintf "%s(%s)" (emit_addr_conv kind) (emit_expr e)

and emit_builtin = function
  | ThreadIdx X -> "threadIdx.x"
  | ThreadIdx Y -> "threadIdx.y"
  | ThreadIdx Z -> "threadIdx.z"
  | BlockIdx  X -> "blockIdx.x"
  | BlockIdx  Y -> "blockIdx.y"
  | BlockIdx  Z -> "blockIdx.z"
  | ClusterCtaId ->
    "([](){ uint32_t id; \
     asm volatile(\"mov.u32 %0, %%cluster_ctaid.x;\" : \"=r\"(id)); \
     return id; }())"
  | WarpId -> "(threadIdx.x / 32)"
  | LaneId ->
    "([](){ uint32_t id; \
     asm volatile(\"mov.u32 %0, %%laneid;\" : \"=r\"(id)); \
     return id; }())"

and emit_binop = function
  | Add ->  "+"  | Sub ->  "-"  | Mul -> "*" | Div -> "/" | Mod -> "%"
  | Eq ->   "==" | Ne ->   "!=" | Lt ->  "<" | Le ->  "<="
  | Gt ->   ">"  | Ge ->   ">="
  | And ->  "&&" | Or ->   "||"
  | Shl ->  "<<" | Shr ->  ">>"
  | BitAnd -> "&"  | BitOr ->  "|"  | BitXor -> "^"

and emit_unop = function
  | Neg -> "-" | Not -> "!" | BitNot -> "~"

and emit_addr_conv = function
  | GenericToShared -> "__cvta_generic_to_shared"
  | GenericToGlobal -> "__cvta_generic_to_global"
  | GenericToLocal -> "__cvta_generic_to_local"
  | SharedToGeneric -> "__cvta_shared_to_generic"
  | GlobalToGeneric -> "__cvta_global_to_generic"
  | LocalToGeneric ->  "__cvta_local_to_generic"
  | ToSharedCluster -> "__cvta_generic_to_shared"
  | ClusterToShared -> "__cvta_cluster_to_shared"

let emit_packed_expr (Expr e) = emit_expr e

let emit_barrier = function
  | CtaSync ->  "__syncthreads();"
  | WarpSync -> "__syncwarp();"
  | MemFence -> "__threadfence();"
  | MbarInit { mbar; count } ->
    Printf.sprintf
      "asm volatile(\"mbarrier.init.shared.b64 [%%0], %d;\" \
       :: \"r\"(&%s) : \"memory\");"
      count mbar.var_name
  | MbarArriveExpect { mbar; bytes } ->
    Printf.sprintf
      "asm volatile(\"mbarrier.arrive.expect_tx.shared.b64 [%%0], %%1;\" \
       :: \"r\"(&%s), \"r\"(%s) : \"memory\");"
      mbar.var_name (emit_expr bytes)
  | MbarWaitParity { mbar; phase } ->
    Printf.sprintf
      "asm volatile(\"mbarrier.try_wait.parity.shared.b64 [%%0], %%1, %%2;\" \
       :: \"r\"(&%s), \"r\"(%s), \"r\"(0x989680U) : \"memory\");"
      mbar.var_name (emit_expr phase)
  | MbarArrive { mbar } ->
    Printf.sprintf
      "asm volatile(\"mbarrier.arrive.shared::cta.b64 [%%0];\" \
       :: \"r\"(&%s) : \"memory\");"
      mbar.var_name
  | ClusterArrive ->
    "asm volatile(\"barrier.cluster.arrive.relaxed.aligned;\");"
  | ClusterWait ->
    "asm volatile(\"barrier.cluster.wait.acquire.aligned;\");"
  | CpAsyncCommitGroup ->
    "asm volatile(\"cp.async.commit_group;\");"
  | CpAsyncWaitAll ->
    "asm volatile(\"cp.async.wait_all;\");"
  | Tcgen05Wait ->
    "asm volatile(\"tcgen05.wait::ld.sync.aligned;\");"
  | Tcgen05Fence ->
    "asm volatile(\"tcgen05.fence::after_thread_sync;\");"

let emit_copy (c : copy) : string =
  let (Tensor src) = c.src_tensor in
  let (Tensor dst) = c.dst_tensor in
  let pred = match c.pred_expr with
    | None ->   ""
    | Some p -> Printf.sprintf "if (%s) " (emit_expr p)
  in
  match c.copy_kind with
  | CpAsync ->
    Printf.sprintf
      "%sasm volatile(\"cp.async.ca.shared.global [%%0], [%%1], 16;\" \
       :: \"r\"(__cvta_generic_to_shared(%s)), \"l\"(%s) : \"memory\");"
      pred dst.tensor_name src.tensor_name
  | TmaLoad ->
    let mbar_s = match c.mbar_var with
      | None ->   "nullptr"
      | Some v -> Printf.sprintf "&%s" v.var_name
    in
    Printf.sprintf
      "%stma_2d_gmem2smem(%s, &%s_tmap, coord_k, coord_m, %s);"
      pred dst.tensor_name src.tensor_name mbar_s
  | TmaMulticast ->
    let mbar_s = match c.mbar_var with
      | None ->   "nullptr"
      | Some v -> Printf.sprintf "&%s" v.var_name
    in
    Printf.sprintf
      "%stma_2d_gmem2smem_multicast(%s, &%s_tmap, coord_k, \
       coord_n + cta_rank * (BN / 2), %s, 0b11);"
      pred dst.tensor_name src.tensor_name mbar_s
  | RegToSmem ->
    Printf.sprintf "cute::copy(%s, %s);"
      src.tensor_name dst.tensor_name
  | SmemToReg ->
    Printf.sprintf "cute::copy(%s, %s);"
      src.tensor_name dst.tensor_name

let emit_mma (m : mma_desc) : string =
  let (Tensor a) = m.tensor_a in
  let (Tensor b) = m.tensor_b in
  let (Tensor c) = m.tensor_c in
  let accum = if m.accum_flag then "1" else "0" in
  match m.mma_kind with
  | Sm80Mma ->
    Printf.sprintf
      "cute::gemm(tiled_mma, %s, %s, %s);"
      a.tensor_name b.tensor_name c.tensor_name
  | Sm90Wgmma ->
    let desc_a = match m.smem_desc_a with
      | None ->   a.tensor_name
      | Some d ->
        Smem_desc.emit_make_smem_desc
          a.tensor_name
          d.Smem_desc.leading_off
          d.Smem_desc.stride_off
          d.Smem_desc.swizzle_mode
    in
    let desc_b = match m.smem_desc_b with
      | None ->   b.tensor_name
      | Some d ->
        Smem_desc.emit_make_smem_desc
          b.tensor_name
          d.Smem_desc.leading_off
          d.Smem_desc.stride_off
          d.Smem_desc.swizzle_mode
    in
    Printf.sprintf
      "wgmma::wgmma_async(%s, %s, %s, %s);"
      c.tensor_name desc_a desc_b accum
  | Sm100Tcgen05 ->
    let desc_a = match m.smem_desc_a with
      | None ->   a.tensor_name
      | Some _ -> Printf.sprintf "make_smem_desc(%s)" a.tensor_name
    in
    let desc_b = match m.smem_desc_b with
      | None ->   b.tensor_name
      | Some _ -> Printf.sprintf "make_smem_desc(%s)" b.tensor_name
    in
    Printf.sprintf
      "asm volatile(\"tcgen05.mma.cta_group::1.kind::mxf16 \
       [%%0], %%1, %%2, %%3;\" \
       :: \"r\"(tmem_addr), \"r\"(%s), \"r\"(%s), \"n\"(%s) : \"memory\");"
      desc_a desc_b accum

let emit_op = function
  | Copy c -> emit_copy c
  | Mma  m -> emit_mma m
  | Barrier b -> emit_barrier b
  | TmemAlloc { addr_var; col_count } ->
    Printf.sprintf
      "asm volatile(\"tcgen05.alloc.cta_group::1.sync.aligned.\
       shared::cta.b32 [%%0], %d;\" \
       :: \"r\"(&%s) : \"memory\");"
      col_count addr_var.var_name
  | TmemDealloc { addr_var; col_count } ->
    Printf.sprintf
      "asm volatile(\"tcgen05.dealloc.cta_group::1.sync.aligned.b32 \
       %%0, %d;\" \
       :: \"r\"(%s) : \"memory\");"
      col_count addr_var.var_name
  | TmemLoad { dst_vars; src_addr; col_offset } ->
    let regs = String.concat ~sep:", "
      (List.map dst_vars ~f:(fun v -> v.var_name))
    in
    Printf.sprintf
      "asm volatile(\"tcgen05.ld.sync.aligned.32x32b.x%d.b32 \
       {%s}, [%%0 + %d];\" \
       :: \"r\"(%s) : \"memory\");"
      (List.length dst_vars)
      regs
      col_offset
      (emit_expr src_addr)
  | TmemCommit { mbar_var; cta_mask } ->
    let mask_s = match cta_mask with
      | None ->   ""
      | Some m -> Printf.sprintf ", %d" m
    in
    Printf.sprintf
      "asm volatile(\"tcgen05.commit.cta_group::1.\
       mbarrier::arrive::one.shared::cluster.b64 [%%0]%s;\" \
       :: \"r\"(&%s) : \"memory\");"
      mask_s mbar_var.var_name
  | SmemDescInit { desc_var; ptr_expr; leading_dim; stride; swizzle } ->
    let sw_bits = Smem_desc.swizzle_mode_bits
      (Smem_desc.swizzle_mode_of swizzle)
    in
    Printf.sprintf
      "uint64_t %s = make_smem_desc_raw(%s, %d, %d, %d);"
      desc_var.var_name (emit_expr ptr_expr)
      leading_dim stride sw_bits

let indent (depth : int) : string = String.make (depth * 2) ' '

let rec emit_stmt ?(depth = 0) (s : stmt) : string =
  let ind = indent depth in
  match s with
  | SEmpty -> ""
  | SLet (v, e) ->
    Printf.sprintf "%sconst %s %s = %s;"
      ind (emit_packed_scalar v.var_type) v.var_name (emit_packed_expr e)
  | SLetMut (v, e) ->
    Printf.sprintf "%s%s %s = %s;"
      ind (emit_packed_scalar v.var_type) v.var_name (emit_packed_expr e)
  | SAssign (v, e) ->
    Printf.sprintf "%s%s = %s;"
      ind v.var_name (emit_packed_expr e)
  | SOp op ->
    Printf.sprintf "%s%s" ind (emit_op op)
  | SIf (cond, thn, els) ->
    let thn_s = emit_stmts ~depth:(depth+1) thn in
    let els_s = match els with
      | [] -> ""
      | _ ->
        Printf.sprintf " else {\n%s\n%s}"
          (emit_stmts ~depth:(depth+1) els) ind
    in
    Printf.sprintf "%sif (%s) {\n%s\n%s}%s"
      ind (emit_expr cond) thn_s ind els_s
  | SFor { var; start; stop; step; dir = _; unroll; body } ->
    let pragma =
      if unroll then Printf.sprintf "%s#pragma unroll\n" ind else ""
    in
    Printf.sprintf
      "%s%sfor (%s %s = %s; %s < %s; %s += %s) {\n%s\n%s}"
      pragma ind
      (emit_packed_scalar var.var_type) var.var_name (emit_expr start)
      var.var_name (emit_expr stop)
      var.var_name (emit_expr step)
      (emit_stmts ~depth:(depth+1) body)
      ind
  | SPipeline { stages; prologue; mainloop; epilogue } ->
    String.concat ~sep:"\n" [
      Printf.sprintf "%s// pipeline prologue (depth=%d)" ind stages;
      emit_stmts ~depth prologue;
      Printf.sprintf "%s// pipeline mainloop" ind;
      emit_stmts ~depth mainloop;
      Printf.sprintf "%s// pipeline epilogue" ind;
      emit_stmts ~depth epilogue;
    ]
  | SWarpGroup (role, body) ->
    let role_s = match role with
      | Cluster.Producer ->  "producer"
      | Cluster.Consumer ->  "consumer"
      | Cluster.Epilogue ->  "epilogue"
      | Cluster.Scheduler -> "scheduler"
    in
    Printf.sprintf "%s// warp role: %s\n%s"
      ind role_s (emit_stmts ~depth body)
  | SPragma (pragma, body) ->
    Printf.sprintf "%s#pragma %s\n%s"
      ind pragma (emit_stmts ~depth body)
  | SSeq stmts ->
    emit_stmts ~depth stmts

and emit_stmts ?(depth = 0) (stmts : stmt list) : string =
  String.concat ~sep:"\n"
    (List.filter_map stmts ~f:(fun s ->
      let r = emit_stmt ~depth s in
      if String.is_empty r then None else Some r))

let emit_shared_storage (k : tirix) : string =
  let tensor_decls = List.filter_map k.tensors
    ~f:(fun (name, Tensor t) ->
      match t.tensor_memspace with
      | Memspace.Shared ->
        let elem_t = Elemtype.cpp_name t.tensor_elem_type in
        let size   = Layout.size t.tensor_layout in
        Some (Printf.sprintf "  %s %s[%d];" elem_t name size)
      | _ -> None)
  in
  let mbar_decls =
    if tirix_is_tma k then
      [ Printf.sprintf
          "  __align__(8) uint64_t full_mbar[%d];" k.pipeline_depth
      ; Printf.sprintf
          "  __align__(8) uint64_t empty_mbar[%d];" k.pipeline_depth ]
    else []
  in
  let tmem_decl =
    match k.family with
    | Kernel_desc.Blackwell -> ["  uint32_t tmem_addr[1];"]
    | _ -> []
  in
  let all = tensor_decls @ mbar_decls @ tmem_decl in
  Printf.sprintf "struct SharedStorage {\n%s\n};"
    (String.concat ~sep:"\n" all)

let emit_helper (h : helper_func) : string =
  let params = String.concat ~sep:", "
    (List.map h.hf_params ~f:(fun v ->
      Printf.sprintf "%s %s"
        (emit_packed_scalar v.var_type) v.var_name))
  in
  Printf.sprintf
    "__device__ __forceinline__ %s %s(%s) {\n%s\n}"
    (emit_packed_scalar h.hf_ret_type)
    h.hf_name
    params
    (emit_stmts ~depth:1 h.hf_body)

let emit_params (k : tirix) : string =
  String.concat ~sep:",\n  "
    (List.map k.params ~f:(fun p ->
      let (Tensor t) = p.param_tensor in
      let elem_t = Elemtype.cpp_name t.tensor_elem_type in
      match p.param_name with
      | "M" | "N" | "K" ->
        Printf.sprintf "int %s" p.param_name
      | _ ->
        if p.param_is_tma then
          Printf.sprintf
            "const __grid_constant__ CUtensorMap %s_tmap" p.param_name
        else
          Printf.sprintf "const %s* %s" elem_t p.param_name))

let emit_kernel_func (k : tirix) : string =
  let n_threads    = Cluster.thread_count k.cluster in
  let cluster_attr =
    if Cluster.is_2sm k.cluster then
      Printf.sprintf "__attribute__((%s))\n"
        (Cluster.emit_cluster_attr k.cluster)
    else ""
  in
  let pre_dispatch =
    List.filter k.body ~f:(function SWarpGroup _ -> false | _ -> true)
  in
  let pre_s = emit_stmts ~depth:1 pre_dispatch in
  let warp_cases = List.filter_map k.body ~f:(function
    | SWarpGroup (role, body) ->
      let warp_ids = List.filter_map k.cluster.Cluster.warp_roles
        ~f:(fun (id, r) -> if Poly.(r = role) then Some id else None)
      in
      let cond = match warp_ids with
        | [id] -> Printf.sprintf "warp_id == %d" id
        | ids ->
          String.concat ~sep:" || "
            (List.map ids ~f:(Printf.sprintf "warp_id == %d"))
      in
      Some (Printf.sprintf "  if (%s) {\n%s\n  }"
        cond (emit_stmts ~depth:2 body))
    | _ -> None)
  in
  let warp_dispatch = String.concat ~sep:" else " warp_cases in
  Printf.sprintf
    "%s__global__ __launch_bounds__(%d)\nvoid %s(\n  %s\n) {\n\
    \  extern __shared__ char smem_buf[];\n\
    \  SharedStorage& smem =\n\
    \    *reinterpret_cast<SharedStorage*>(smem_buf);\n\
    \  const int warp_id = threadIdx.x / 32;\n\
    \  const int lane_id = threadIdx.x %% 32;\n\
    \  (void)lane_id;\n\
     %s\n\
     %s\n\
     }\n"
    cluster_attr
    n_threads
    k.name
    (emit_params k)
    pre_s
    warp_dispatch

let emit_host_launcher (k : tirix) : string =
  let is_blackwell =
    match k.family with Kernel_desc.Blackwell -> true | _ -> false
  in
  let launch =
    if is_blackwell then
      Printf.sprintf
        "  cudaLaunchConfig_t cfg = {};\n\
        \  cfg.gridDim  = grid;\n\
        \  cfg.blockDim = block;\n\
        \  cfg.dynamicSmemBytes = %d;\n\
        \  cudaLaunchAttribute attrs[1];\n\
        \  attrs[0].id = cudaLaunchAttributeClusterDimension;\n\
        \  attrs[0].val.clusterDim = {%d, %d, %d};\n\
        \  cfg.attrs = attrs; cfg.numAttrs = 1;\n\
        \  cudaLaunchKernelEx(&cfg, %s, M, N, K);"
        k.smem_bytes
        k.cluster.Cluster.dims.Cluster.x
        k.cluster.Cluster.dims.Cluster.y
        k.cluster.Cluster.dims.Cluster.z
        k.name
    else
      Printf.sprintf
        "  cudaFuncSetAttribute(%s,\n\
        \    cudaFuncAttributeMaxDynamicSharedMemorySize, %d);\n\
        \  %s<<<grid, block, %d>>>(M, N, K);"
        k.name k.smem_bytes k.name k.smem_bytes
  in
  Printf.sprintf
    "void launch_%s(\n  %s,\n  int M, int N, int K\n) {\n\
    \  dim3 grid((M + %d - 1) / %d, (N + %d - 1) / %d, 1);\n\
    \  dim3 block(%d, 1, 1);\n\
     %s\n\
     }\n"
    k.name
    (emit_params k)
    k.bm k.bm k.bn k.bn
    (Cluster.thread_count k.cluster)
    launch

let emit_includes (k : tirix) : string =
  let arch_inc = match k.family with
    | Kernel_desc.Ampere ->    ""
    | Kernel_desc.Hopper ->    "#include <cuda_bf16.h>\n"
    | Kernel_desc.Blackwell -> "#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n"
  in
  String.concat ~sep:"\n" [
    "#pragma once"
  ; "#include <cute/tensor.hpp>"
  ; "#include <cute/atom/mma_atom.hpp>"
  ; "#include <cute/atom/copy_atom.hpp>"
  ; "#include <cute/algorithm/gemm.hpp>"
  ; "#include <cute/algorithm/copy.hpp>"
  ; "using namespace cute;"
  ; arch_inc
  ]

let emit (k : tirix) : Backend_cute.output =
  let includes       = emit_includes k in
  let helpers        = String.concat ~sep:"\n\n"
    (List.map k.helpers ~f:emit_helper) in
  let shared_storage = emit_shared_storage k in
  let kernel_func    = emit_kernel_func k in
  let host_launcher  = emit_host_launcher k in
  let full_source    = String.concat ~sep:"\n\n"
    [ includes; helpers; shared_storage; kernel_func; host_launcher ]
  in
  { Backend_cute.filename      = k.name ^ ".cuh"
  ; includes
  ; helpers
  ; shared_storage
  ; producer_body  = ""
  ; consumer_body  = ""
  ; epilogue_body  = ""
  ; kernel_func
  ; host_launcher
  ; full_source
  }
