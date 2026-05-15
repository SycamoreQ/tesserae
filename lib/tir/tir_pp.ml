open Base
open Tir

(* ------------------------------------------------------------------ *)
(* Helpers                                                             *)
(* ------------------------------------------------------------------ *)

let indent n = String.make (n * 2) ' '

let pp_axis = function X -> "x" | Y -> "y" | Z -> "z"

let pp_builtin = function
  | ThreadIdx a  -> Printf.sprintf "threadIdx.%s" (pp_axis a)
  | BlockIdx  a  -> Printf.sprintf "blockIdx.%s"  (pp_axis a)
  | ClusterCtaId -> "cluster_ctaid"
  | WarpId       -> "warp_id"
  | LaneId       -> "lane_id"

let pp_addr_conv = function
  | GenericToShared  -> "__cvta_generic_to_shared"
  | GenericToGlobal  -> "__cvta_generic_to_global"
  | GenericToLocal   -> "__cvta_generic_to_local"
  | SharedToGeneric  -> "__cvta_shared_to_generic"
  | GlobalToGeneric  -> "__cvta_global_to_generic"
  | LocalToGeneric   -> "__cvta_local_to_generic"
  | ToSharedCluster  -> "__cvta_to_shared_cluster"
  | ClusterToShared  -> "__cvta_cluster_to_shared"

let pp_binop = function
  | Add -> "+" | Sub -> "-" | Mul -> "*" | Div -> "/" | Mod -> "%"
  | Eq  -> "==" | Ne -> "!=" | Lt -> "<" | Le -> "<=" | Gt -> ">" | Ge -> ">="
  | And -> "&&" | Or -> "||"
  | Shl -> "<<" | Shr -> ">>"
  | BitAnd -> "&" | BitOr -> "|" | BitXor -> "^"

let pp_unop = function Neg -> "-" | Not -> "!" | BitNot -> "~"

let pp_scalar_ty : type a. a scalar_ty -> string = function
  | U8   -> "uint8_t"
  | U32  -> "uint32_t"
  | S32  -> "int32_t"
  | U64  -> "uint64_t"
  | F16  -> "__half"
  | F32  -> "float"
  | BF16 -> "__nv_bfloat16"
  | Bool -> "bool"
  | Ptr  -> "uint64_t*"

let pp_packed_scalar (Scalar s) = pp_scalar_ty s

(* ------------------------------------------------------------------ *)
(* Expressions                                                         *)
(* ------------------------------------------------------------------ *)

let rec pp_expr : type a. a expr -> string = function
  | Const (U8,  v) -> string_of_int v
  | Const (U32, v) -> Printf.sprintf "%lu" v
  | Const (S32, v) -> Printf.sprintf "%ld" v
  | Const (U64, v) -> Printf.sprintf "%Lu" v
  | Const (F16, v) -> Printf.sprintf "__float2half(%f)" v
  | Const (F32, v) -> Printf.sprintf "%ff" v
  | Const (BF16,v) -> Printf.sprintf "__float2bfloat16(%f)" v
  | Const (Bool,v) -> if v then "true" else "false"
  | Const (Ptr, v) -> Printf.sprintf "%Lu" v
  | Cast (ty, e)   -> Printf.sprintf "((%s)(%s))" (pp_scalar_ty ty) (pp_expr e)
  | Var v          -> v.var_name
  | Builtin b      -> pp_builtin b
  | Binop (op, l, r) ->
    Printf.sprintf "(%s %s %s)" (pp_expr l) (pp_binop op) (pp_expr r)
  | Unop (op, e)   -> Printf.sprintf "(%s%s)" (pp_unop op) (pp_expr e)
  | AddrConv (k, e)->
    Printf.sprintf "%s(%s)" (pp_addr_conv k) (pp_expr e)

let pp_packed_expr (Expr e) = pp_expr e

(* ------------------------------------------------------------------ *)
(* Barriers                                                            *)
(* ------------------------------------------------------------------ *)

let pp_barrier = function
  | CtaSync    -> "__syncthreads();"
  | WarpSync   -> "__syncwarp();"
  | MemFence   -> "__threadfence();"
  | MbarInit { mbar; count } ->
    Printf.sprintf
      "asm volatile(\"mbarrier.init.shared.b64 [%%0], %d;\" :: \"r\"(&%s));"
      count mbar.var_name
  | MbarArriveExpect { mbar; bytes } ->
    Printf.sprintf
      "asm volatile(\"mbarrier.arrive.expect_tx.shared.b64 [%%0], %%1;\" :: \"r\"(&%s), \"r\"(%s));"
      mbar.var_name (pp_expr bytes)
  | MbarWaitParity { mbar; phase } ->
    Printf.sprintf
      "asm volatile(\"mbarrier.wait.parity.shared.b64 [%%0], %%1;\" :: \"r\"(&%s), \"r\"(%s));"
      mbar.var_name (pp_expr phase)
  | MbarArrive { mbar } ->
    Printf.sprintf
      "asm volatile(\"mbarrier.arrive.shared.b64 [%%0];\" :: \"r\"(&%s));"
      mbar.var_name
  | ClusterArrive    -> "asm volatile(\"barrier.cluster.arrive.release.aligned;\");"
  | ClusterWait      -> "asm volatile(\"barrier.cluster.wait.acquire.aligned;\");"
  | CpAsyncWaitAll   -> "asm volatile(\"cp.async.wait_all;\");"
  | CpAsyncCommitGroup -> "asm volatile(\"cp.async.commit_group;\");"
  | Tcgen05Wait      -> "asm volatile(\"tcgen05.wait::ld.sync.aligned;\");"
  | Tcgen05Fence     -> "asm volatile(\"tcgen05.fence::after_thread_sync;\");"

(* ------------------------------------------------------------------ *)
(* Operations                                                          *)
(* ------------------------------------------------------------------ *)

let pp_copy_kind = function
  | CpAsync    -> "cp.async"
  | TmaLoad    -> "tma.load"
  | TmaMulticast -> "tma.multicast"
  | RegToSmem  -> "reg→smem"
  | SmemToReg  -> "smem→reg"

let pp_mma_kind = function
  | Sm80Mma      -> "mma.sync (SM80)"
  | Sm90Wgmma    -> "wgmma (SM90)"
  | Sm100Tcgen05 -> "tcgen05.mma (SM100)"

let pp_tensor_name (Tensor t) = t.tensor_name

let pp_op = function
  | Copy c ->
    Printf.sprintf "copy[%s] %s → %s%s"
      (pp_copy_kind c.copy_kind)
      (pp_tensor_name c.src_tensor)
      (pp_tensor_name c.dst_tensor)
      (Option.value_map c.pred_expr ~default:""
        ~f:(fun p -> Printf.sprintf " if (%s)" (pp_expr p)))
  | Mma m ->
    Printf.sprintf "%s(%s, %s) → %s"
      (pp_mma_kind m.mma_kind)
      (pp_tensor_name m.tensor_a)
      (pp_tensor_name m.tensor_b)
      (pp_tensor_name m.tensor_c)
  | Barrier b  -> pp_barrier b
  | TmemAlloc  { addr_var; col_count } ->
    Printf.sprintf "tmem.alloc %s cols=%d" addr_var.var_name col_count
  | TmemDealloc { addr_var; col_count } ->
    Printf.sprintf "tmem.dealloc %s cols=%d" addr_var.var_name col_count
  | TmemLoad { dst_vars; src_addr; col_offset } ->
    let dsts = List.map dst_vars ~f:(fun v -> v.var_name)
               |> String.concat ~sep:", " in
    Printf.sprintf "tmem.ld [%s+%d] → {%s}"
      (pp_expr src_addr) col_offset dsts
  | TmemCommit { mbar_var; cta_mask } ->
    Printf.sprintf "tmem.commit mbar=%s mask=%s"
      mbar_var.var_name
      (Option.value_map cta_mask ~default:"none"
        ~f:(fun m -> Printf.sprintf "0x%x" m))
  | SmemDescInit { desc_var; ptr_expr; leading_dim; stride; _ } ->
    Printf.sprintf "smem_desc %s = make_smem_desc(%s, ld=%d, st=%d)"
      desc_var.var_name (pp_expr ptr_expr) leading_dim stride

(* ------------------------------------------------------------------ *)
(* Statements                                                          *)
(* ------------------------------------------------------------------ *)

let rec pp_stmt ?(depth=0) stmt =
  let ind = indent depth in
  match stmt with
  | SLet (v, e) ->
    Printf.sprintf "%slet %s : %s = %s"
      ind v.var_name (pp_packed_scalar v.var_type) (pp_packed_expr e)
  | SLetMut (v, e) ->
    Printf.sprintf "%slet mut %s : %s = %s"
      ind v.var_name (pp_packed_scalar v.var_type) (pp_packed_expr e)
  | SAssign (v, e) ->
    Printf.sprintf "%s%s = %s" ind v.var_name (pp_packed_expr e)
  | SOp op ->
    Printf.sprintf "%s%s" ind (pp_op op)
  | SIf (cond, thn, els) ->
    let thn_s = List.map thn ~f:(pp_stmt ~depth:(depth+1))
                |> String.concat ~sep:"\n" in
    let els_s = match els with
      | [] -> ""
      | _  -> Printf.sprintf "\n%selse {\n%s\n%s}"
                ind
                (List.map els ~f:(pp_stmt ~depth:(depth+1))
                 |> String.concat ~sep:"\n")
                ind
    in
    Printf.sprintf "%sif (%s) {\n%s\n%s}%s"
      ind (pp_expr cond) thn_s ind els_s
  | SFor { var; start; stop; step; unroll; body; _ } ->
    let pragma = if unroll then Printf.sprintf "%s#pragma unroll\n" ind else "" in
    let body_s = List.map body ~f:(pp_stmt ~depth:(depth+1))
                 |> String.concat ~sep:"\n" in
    Printf.sprintf "%s%sfor (int %s = %s; %s < %s; %s += %s) {\n%s\n%s}"
      pragma ind
      var.var_name (pp_expr start)
      var.var_name (pp_expr stop)
      var.var_name (pp_expr step)
      body_s ind
  | SPipeline { stages; prologue; mainloop; epilogue } ->
    let pp_block label stmts =
      let s = List.map stmts ~f:(pp_stmt ~depth:(depth+1))
              |> String.concat ~sep:"\n" in
      Printf.sprintf "%s// %s\n%s" ind label s
    in
    Printf.sprintf "%s// pipeline stages=%d\n%s\n%s\n%s"
      ind stages
      (pp_block "prologue" prologue)
      (pp_block "mainloop" mainloop)
      (pp_block "epilogue" epilogue)
  | SWarpGroup (role, body) ->
    let role_s = match role with
      | Cluster.Producer  -> "Producer"
      | Cluster.Consumer  -> "Consumer"
      | Cluster.Epilogue  -> "Epilogue"
      | Cluster.Scheduler -> "Scheduler"
    in
    let body_s = List.map body ~f:(pp_stmt ~depth:(depth+1))
                 |> String.concat ~sep:"\n" in
    Printf.sprintf "%s// warp_group [%s]\n%s" ind role_s body_s
  | SPragma (pragma, body) ->
    let body_s = List.map body ~f:(pp_stmt ~depth:(depth+1))
                 |> String.concat ~sep:"\n" in
    Printf.sprintf "%s#pragma %s\n%s" ind pragma body_s
  | SSeq stmts ->
    List.map stmts ~f:(pp_stmt ~depth)
    |> String.concat ~sep:"\n"
  | SEmpty -> Printf.sprintf "%s// (empty)" ind

(* ------------------------------------------------------------------ *)
(* Helper functions                                                    *)
(* ------------------------------------------------------------------ *)

let pp_helper (h : helper_func) : string =
  let params = List.map h.hf_params ~f:(fun v ->
    Printf.sprintf "%s %s" (pp_packed_scalar v.var_type) v.var_name)
    |> String.concat ~sep:", "
  in
  let body = List.map h.hf_body ~f:(pp_stmt ~depth:1)
             |> String.concat ~sep:"\n" in
  Printf.sprintf "__device__ __forceinline__ %s %s(%s) {\n%s\n}"
    (pp_packed_scalar h.hf_ret_type)
    h.hf_name
    params
    body

(* ------------------------------------------------------------------ *)
(* Top-level kernel IR                                                 *)
(* ------------------------------------------------------------------ *)

let pp_param (p : param) : string =
  let (Tensor t) = p.param_tensor in
  let elem = Elemtype.cpp_name t.tensor_elem_type in
  if p.param_is_tma then
    Printf.sprintf "const __grid_constant__ CUtensorMap %s" p.param_name
  else
    Printf.sprintf "const %s* %s" elem p.param_name

let pp_tir (k : tir) : string =
  let params = List.map k.params ~f:pp_param
               |> String.concat ~sep:",\n    " in
  let helpers = List.map k.helpers ~f:pp_helper
                |> String.concat ~sep:"\n\n" in
  let body = List.map k.body ~f:(pp_stmt ~depth:1)
             |> String.concat ~sep:"\n" in
  Printf.sprintf
    "%s\n\n__global__ void %s(\n    %s)\n{\n%s\n}"
    helpers k.name params body
