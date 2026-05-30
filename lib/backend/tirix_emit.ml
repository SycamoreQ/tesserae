open Base
open Tesserae_tirix
open Tesserae_pipeline
open Tesserae_core
open Tirix
open Tesserae_kernel
open Tesserae_atoms

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
  | Ptr ->  "void*"

let emit_packed_scalar (Scalar s) = emit_scalar_ty s

let emit_arith_op = function
  | Arith.Add -> "+" | Arith.Sub -> "-" | Arith.Mul -> "*" | Arith.Div -> "/" | Arith.Mod -> "%"

let emit_cmp_op = function
  | Cmp.Eq  -> "==" | Cmp.Ne -> "!=" | Cmp.Lt -> "<" | Cmp.Le -> "<=" | Cmp.Gt -> ">" | Cmp.Ge -> ">="

let emit_logic_op = function
  | Logic.And -> "&&" | Logic.Or -> "||"

let emit_bitwise_op = function
  | Bitwise.BitAnd -> "&" | Bitwise.BitOr -> "|" | Bitwise.BitXor -> "^"
  | Bitwise.Shl    -> "<<" | Bitwise.Shr -> ">>"

let emit_unop = function
  | Unop.Neg -> "-" | Unop.Not -> "!" | Unop.BitNot -> "~"

let rec emit_expr : type a. a expr -> string = function
  | Const (U8, v) -> Int.to_string v
  | Const (U32, v) -> Printf.sprintf "%luU" v
  | Const (S32, v) -> Printf.sprintf "%ld" v
  | Const (U64, v) -> Printf.sprintf "%LuULL" v
  | Const (F16, v) -> Printf.sprintf "__float2half(%ff)" v
  | Const (F32, v) -> Printf.sprintf "%ff" v
  | Const (BF16, v) -> Printf.sprintf "__float2bfloat16(%ff)" v
  | Const (Bool, v) -> if v then "true" else "false"
  | Const (Ptr, v) -> Printf.sprintf "(void*)0x%LxULL" v
  | Cast (ty, e) ->
    Printf.sprintf "((%s)(%s))" (emit_scalar_ty ty) (emit_expr e)
  | Var v -> v.var_name
  | Builtin b -> emit_builtin b
  | Arith (op, l, r) ->
    Printf.sprintf "(%s %s %s)" (emit_expr l) (emit_arith_op op) (emit_expr r)
  | Cmp (op, l, r) ->
    Printf.sprintf "(%s %s %s)" (emit_expr l) (emit_cmp_op op) (emit_expr r)
  | Logic (op, l, r) ->
    Printf.sprintf "(%s %s %s)" (emit_expr l) (emit_logic_op op) (emit_expr r)
  | Bitwise (op, l, r) ->
    Printf.sprintf "(%s %s %s)" (emit_expr l) (emit_bitwise_op op) (emit_expr r)
  | Unop (op, e) ->
    Printf.sprintf "(%s%s)" (emit_unop op) (emit_expr e)
  | AddrConv (kind, e) ->
    let conv = emit_addr_conv kind in
    let arg = emit_expr e in
    Printf.sprintf "((uint32_t)%s(%s))" conv arg


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


(* All mbarrier address computations use __smem_base + offsetof to avoid
   NVRTC folding away the cvta.to.shared conversion on smem symbols. *)
let emit_barrier = function
  | CtaSync ->  "__syncthreads();"
  | WarpSync -> "__syncwarp();"
  | MemFence -> "__threadfence();"

  | MbarInit { mbar; count } ->
      Printf.sprintf
        "asm volatile(\"mbarrier.init.shared.b64 [%%0], %%1;\" \
         :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))), \
            \"n\"(%d) : \"memory\");"
        mbar.var_name count

  | MbarArriveExpect { mbar; bytes } ->
        Printf.sprintf
          "if (lane_id == 0) { \
             asm volatile(\"mbarrier.arrive.expect_tx.shared.b64 _ , [%%0], %%1;\" \
             :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))), \
                \"r\"((uint32_t)(%s)) : \"memory\"); \
           }"
          mbar.var_name (emit_expr bytes)

  | MbarWaitParity { mbar; phase } ->
      Printf.sprintf
        {ptx|asm volatile(
    "{\n\t"
    "  .reg .pred p;\n\t"
    "  LOOP_START:\n\t"
    "  mbarrier.try_wait.parity.shared.b64 p, [%%0], %%1;\n\t"
    "  @!p bra LOOP_START;\n\t"
    "}"
    :: "r"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))),
       "r"(%s) : "memory"
  );|ptx}
        mbar.var_name (emit_expr phase)

  | MbarArrive { mbar } ->
        Printf.sprintf
          "if (lane_id == 0) { \
             asm volatile(\"mbarrier.arrive.shared.b64 _ , [%%0];\" \
             :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))) \
                : \"memory\"); \
           }"
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
    (* Use __smem_base + offsetof so NVRTC emits a real cvta.to.shared *)
    Printf.sprintf
      "%sasm volatile(\"cp.async.ca.shared.global [%%0], [%%1], 16;\" \
       :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s))), \
          \"l\"((uint64_t)(uintptr_t)%s) : \"memory\");"
      pred dst.tensor_name src.tensor_name

  | TmaLoad ->
      let mbar_s = match c.mbar_var with
        | None -> "nullptr"
        | Some v ->
          Printf.sprintf
            "(uint64_t*)(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))"
            v.var_name
      in
      Printf.sprintf
        "%stma_2d_gmem2smem(__smem_base + offsetof(SharedStorage, %s) + (stage * (sizeof(smem.%s) / 4)), %s_tmap, coord_k, coord_m, %s);"
        pred dst.tensor_name dst.tensor_name src.tensor_name mbar_s

  | TmaMulticast ->
    let mbar_s = match c.mbar_var with
      | None -> "nullptr"
      | Some v ->
        Printf.sprintf
          "(uint64_t*)(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))"
          v.var_name
    in
    Printf.sprintf
      "%stma_2d_gmem2smem_multicast(__smem_base + offsetof(SharedStorage, %s), %s_tmap, coord_k, coord_m, %s, 0b11);"
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
  let (Tensor _c) = m.tensor_c in
  let accum = if m.accum_flag then "1" else "0" in
  match m.mma_kind with

  | Sm80Mma ->
    let _atom_str = match m.mma_atom with
      | Atom80 a -> Mma_atom.emit_cpp a
      | _ -> failwith "emit_mma: Sm80Mma requires Atom80"
    in
    let a_modes = Layout.modes a.tensor_layout in
    let b_modes = Layout.modes b.tensor_layout in
    let a_m, a_k = match a_modes with
      | [m; k] -> (m, k)
      | [_; m; k] -> (m, k)
      | _ -> (16, 16)
    in
    let b_n, b_k = match b_modes with
      | [n; k] -> (n, k)
      | [_; n; k] -> (n, k)
      | _ -> (8, 16)
    in
    Printf.sprintf
      "{\n\
      \  auto sA = make_tensor(make_smem_ptr(&smem.%s[0]),\n\
      \    make_layout(make_shape(Int<%d>{}, Int<%d>{})));\n\
      \  auto sB = make_tensor(make_smem_ptr(&smem.%s[0]),\n\
      \    make_layout(make_shape(Int<%d>{}, Int<%d>{})));\n\
      \  auto tAsA = thr_mma.partition_A(sA);\n\
      \  auto tBsB = thr_mma.partition_B(sB);\n\
      \  cute::gemm(tiled_mma, acc_frag, tAsA, tBsB, acc_frag);\n\
       }"
      a.tensor_name a_m a_k
      b.tensor_name b_n b_k

  | Sm90Wgmma ->
      let (Tensor a) = m.tensor_a in
      let (Tensor b) = m.tensor_b in
      let (mn_str, at, bt, num_regs, k_val) = match m.mma_atom with
        | Atom90 atom ->
          let (_, n, k) = Mma_atom.shape atom in
          let num_regs = n / 2 in
          let at = String.lowercase (Mma_atom.elem_string atom.Mma_atom.a_type) in
          let bt = String.lowercase (Mma_atom.elem_string atom.Mma_atom.b_type) in
          (Printf.sprintf "m64n%dk%d" n k, at, bt, num_regs, k)
        | _ -> failwith "emit_mma: Sm90Wgmma requires Atom90"
      in
      let accum = if m.accum_flag then 1 else 0 in
      (* lead = number of 16-byte units in the K dimension *)
      let lead = (k_val * 2) / 16 in
      let reg_tokens =
        List.init num_regs ~f:(fun i -> Printf.sprintf "%%%d" i)
        |> String.concat ~sep:", "
      in
      let acc_outputs =
        List.init num_regs ~f:(fun i -> Printf.sprintf "\"+f\"(acc_frag[%d])" i)
        |> String.concat ~sep:", "
      in
      let sw_lead_bits =
        Int64.bit_or
          (Int64.shift_left 2L 52)
          (Int64.of_int (lead lsl 16))
      in
      let or_const = Int64.to_string sw_lead_bits in
      (* Use __smem_base + offsetof so make_wgmma_desc receives a plain
         char* generic pointer, forcing NVRTC to emit cvta.to.shared.u64
         rather than folding away the conversion on a .shared symbol. *)
         let desc_a_expr =
                 Printf.sprintf
                   "make_wgmma_desc((uint32_t)(offsetof(SharedStorage, %s) + (stage * (sizeof(smem.%s) / 4))), %sULL)"
                   a.tensor_name a.tensor_name or_const
               in
        let desc_b_expr =
          Printf.sprintf
            "make_wgmma_desc((uint32_t)(offsetof(SharedStorage, %s) + (stage * (sizeof(smem.%s) / 4))), %sULL)"
            b.tensor_name b.tensor_name or_const
        in
      Printf.sprintf
        "{\n\
                 \  uint64_t desc_a = %s;\n\
                 \  uint64_t desc_b = %s;\n\
                 \  (void)desc_a; (void)desc_b; // Prevent unused variable warnings\n\
                 \  /* --- WGMMA ISOLATION TEST ---\n\
                 \  asm volatile(\"wgmma.fence.sync.aligned;\");\n\
                 \  asm volatile(\n\
                 \    \"wgmma.mma_async.sync.aligned.%s.f32.%s.%s {%s}, %%%d, %%%d, %d, 1, 1, 0, 0;\"\n\
                 \    : %s\n\
                 \    : \"l\"(desc_a), \"l\"(desc_b));\n\
                 \  asm volatile(\"wgmma.commit_group.sync.aligned;\");\n\
                 \  asm volatile(\"wgmma.wait_group.sync.aligned 0;\");\n\
                 \  ----------------------------- */\n\
                 }"
        desc_a_expr desc_b_expr
        mn_str at bt
        reg_tokens
        num_regs
        (num_regs + 1)
        accum
        acc_outputs

  | Sm100Tcgen05 ->
    let desc_a = match m.smem_desc_a with
      | None -> a.tensor_name
      | Some _ -> Printf.sprintf "make_smem_desc(%s)" a.tensor_name
    in
    let desc_b = match m.smem_desc_b with
      | None -> b.tensor_name
      | Some _ -> Printf.sprintf "make_smem_desc(%s)" b.tensor_name
    in
    Printf.sprintf
      "asm volatile(\"tcgen05.mma.cta_group::1.kind::mxf16 \
       [%%0], %%1, %%2, %%3;\" \
       :: \"r\"(tmem_addr), \"r\"(%s), \"r\"(%s), \"n\"(%s) : \"memory\");"
      desc_a desc_b accum


let emit_tiled_mma_decl (k : tirix) : string =
  let rec collect = function
    | [] -> []
    | SOp (Mma m) :: rest -> m.mma_atom :: collect rest
    | SWarpGroup (_, body) :: rest -> collect body @ collect rest
    | SIf (_, thn, els) :: rest -> collect thn @ collect els @ collect rest
    | SFor { body; _ } :: rest -> collect body @ collect rest
    | SPipeline { prologue; mainloop; epilogue; _ } :: rest ->
      collect prologue @ collect mainloop @ collect epilogue @ collect rest
    | SSeq ss :: rest -> collect ss @ collect rest
    | _ :: rest -> collect rest
  in
  let atoms = collect k.body in

  let has_sm90 = List.exists atoms ~f:(function Atom90 _ -> true | _ -> false) in
  if has_sm90 then
    Printf.sprintf
      "  float acc_frag[%d];\n\
      \  for (int i = 0; i < %d; i++) acc_frag[i] = 0.0f;\n"
      ((k.bm * k.bn) / 32) ((k.bm * k.bn) / 32)
  else
    let first_sm80_str =
      List.find_map atoms ~f:(function
        | Atom80 a -> Some (Mma_atom.emit_cpp a)
        | _ -> None)
    in
    match first_sm80_str with
    | None -> ""
    | Some atom_str ->
      Printf.sprintf
        "  auto tiled_mma = make_tiled_mma(MMA_Atom<%s>{});\n\
        \  auto thr_mma = tiled_mma.get_slice(threadIdx.x);\n\
        \  auto acc_frag = partition_fragment_C(thr_mma,\n\
        \    make_layout(make_shape(Int<%d>{}, Int<%d>{})));\n\
        \  clear(acc_frag);\n\
        \  float* acc_raw = reinterpret_cast<float*>(acc_frag.data());\n"
        atom_str k.bm k.bn


let emit_op = function
  | Copy c -> emit_copy c
  | Mma  m -> emit_mma m
  | Barrier b -> emit_barrier b
  | TmemAlloc { addr_var; col_count } ->
    Printf.sprintf
      "asm volatile(\"tcgen05.alloc.cta_group::1.sync.aligned.\
       shared::cta.b32 [%%0], %d;\" \
       :: \"r\"(&smem.%s) : \"memory\");"
      col_count addr_var.var_name

  | TmemDealloc { addr_var; col_count } ->
    Printf.sprintf
      "asm volatile(\"tcgen05.dealloc.cta_group::1.sync.aligned.b32 \
       %%0, %d;\" \
       :: \"r\"(smem.%s) : \"memory\");"
      col_count addr_var.var_name

  | TmemLoad { dst_vars; src_addr; col_offset } ->
    let regs = String.concat ~sep:", "
      (List.map dst_vars ~f:(fun v -> v.var_name))
    in
    Printf.sprintf
      "asm volatile(\"tcgen05.ld.sync.aligned.32x32b.x%d.b32 \
       {%s}, [%%0 + %d];\" \
       :: \"r\"(%s) : \"memory\");"
      (List.length dst_vars) regs col_offset (emit_expr src_addr)

  | TmemCommit { mbar_var; cta_mask } ->
    let mask_s = match cta_mask with
      | None -> ""
      | Some m -> Printf.sprintf ", %d" m
    in
    Printf.sprintf
      "asm volatile(\"tcgen05.commit.cta_group::1.\
       mbarrier::arrive::one.shared::cluster.b64 [%%0]%s;\" \
       :: \"r\"(&smem.%s) : \"memory\");"
      mask_s mbar_var.var_name

  | SmemDescInit { desc_var; ptr_expr; leading_dim; stride; swizzle } ->
    let sw_bits = Smem_desc.swizzle_mode_bits (Smem_desc.swizzle_mode_of swizzle) in
    Printf.sprintf
      "uint64_t %s = make_smem_desc_raw(%s, %d, %d, %d);"
      desc_var.var_name (emit_expr ptr_expr) leading_dim stride sw_bits


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
  let mbar_names = ["full_mbar"; "empty_mbar"] in
  let tensor_decls = List.filter_map k.tensors
    ~f:(fun (name, Tensor t) ->
      if List.mem mbar_names name ~equal:String.equal then None
      else match t.tensor_memspace with
      | Memspace.Shared ->
        let elem_t = Elemtype.cpp_name t.tensor_elem_type in
        let size   = Layout.size t.tensor_layout in
        Some (Printf.sprintf "  %s %s[%d];" elem_t name size)
      | _ -> None)
  in
  let mbar_decls =
    if tirix_is_tma k then
      [ Printf.sprintf "  __align__(8) uint64_t full_mbar[%d];" k.pipeline_depth
      ; Printf.sprintf "  __align__(8) uint64_t empty_mbar[%d];" k.pipeline_depth ]
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
  let builtin_helpers = ["tma_2d_gmem2smem"; "tma_2d_gmem2smem_multicast"; "make_smem_desc_raw"; "make_smem_desc"] in
  if List.mem builtin_helpers h.hf_name ~equal:String.equal then ""
  else
  let params = String.concat ~sep:", "
    (List.map h.hf_params ~f:(fun v ->
      Printf.sprintf "%s %s"
        (emit_packed_scalar v.var_type) v.var_name))
  in
  Printf.sprintf
    "__device__ inline %s %s(%s) {\n%s\n}"
    (emit_packed_scalar h.hf_ret_type)
    h.hf_name params
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
          Printf.sprintf "const CUtensorMap* %s_tmap" p.param_name
        else
          Printf.sprintf "const %s* %s" elem_t p.param_name))

let rec collect_warp_groups (stmts : stmt list) : (Cluster.warp_role * stmt list) list =
  List.concat_map stmts ~f:(function
    | SWarpGroup (role, body) -> [(role, body)]
    | SSeq ss -> collect_warp_groups ss
    | SIf (_, thn, els) -> collect_warp_groups thn @ collect_warp_groups els
    | SFor { body; _ } -> collect_warp_groups body
    | SPipeline { prologue; mainloop; epilogue; _ } ->
      collect_warp_groups prologue @ collect_warp_groups mainloop @ collect_warp_groups epilogue
    | _ -> [])

let rec filter_builtin_vars (stmts : stmt list) : stmt list =
  List.filter_map stmts ~f:(function
    | SLet (v, _) | SLetMut (v, _) when
        String.equal v.var_name "warp_id" ||
        String.equal v.var_name "lane_id" ||
        String.equal v.var_name "full_mbar" ||
        String.equal v.var_name "empty_mbar" ||
        String.equal v.var_name "block_m" ||
        String.equal v.var_name "block_n" ||
        String.equal v.var_name "row" ||
        String.equal v.var_name "col" ||
        String.equal v.var_name "coord_n"   -> None
    | SSeq ss ->
      let filtered = filter_builtin_vars ss in
      if List.is_empty filtered then None else Some (SSeq filtered)
    | s -> Some s)


let rec filter_non_warp (stmts : stmt list) : stmt list =
  List.filter_map stmts ~f:(function
    | SWarpGroup _ -> None
    | SIf (cond, thn, els) ->
      let thn' = filter_non_warp thn in
      let els' = filter_non_warp els in
      if List.is_empty thn' && List.is_empty els' then None
      else Some (SIf (cond, thn', els'))
    | SSeq ss ->
      let filtered = filter_non_warp ss in
      if List.is_empty filtered then None else Some (SSeq filtered)
    | s -> Some s)


(* Emit mbarrier init loop using __smem_base + offsetof for both
   full_mbar and empty_mbar so addresses are stable. *)
let emit_mbar_init_loop (k : tirix) : string =
  if not (tirix_is_tma k) then ""
  else
    let depth = k.pipeline_depth in
    (* Count how many consumer warps exist in the cluster configuration *)
    let consumer_count = List.count k.cluster.Cluster.warp_roles ~f:(fun (_, role) ->
      Poly.(role = Cluster.Consumer))
    in
    let empty_count = if consumer_count = 0 then 1 else consumer_count in
    Printf.sprintf
      "  if (warp_id == 0 && lane_id == 0) {\n\
      \    #pragma unroll\n\
      \    for (int s = 0; s < %d; s++) {\n\
      \      asm volatile(\"mbarrier.init.shared.b64 [%%0], %%1;\"\n\
      \        :: \"r\"((uint32_t)__cvta_generic_to_shared(\n\
      \            __smem_base + offsetof(SharedStorage, full_mbar) + s * sizeof(uint64_t))),\n\
      \         \"n\"(1) : \"memory\");\n\
      \      asm volatile(\"mbarrier.init.shared.b64 [%%0], %%1;\"\n\
      \        :: \"r\"((uint32_t)__cvta_generic_to_shared(\n\
      \            __smem_base + offsetof(SharedStorage, empty_mbar) + s * sizeof(uint64_t))),\n\
      \         \"n\"(%d) : \"memory\");\n\
      \    }\n\
      \  }\n\
      \  __syncthreads();"
      depth empty_count


let emit_kernel_func (k : tirix) : string =
  let n_threads = Cluster.thread_count k.cluster in
  let cluster_attr =
    if Cluster.is_2sm k.cluster then
    Printf.sprintf
      "__cluster_dims__(%d, %d, %d) "
      k.cluster.Cluster.dims.Cluster.x
      k.cluster.Cluster.dims.Cluster.y
      k.cluster.Cluster.dims.Cluster.z
    else ""
  in
  let warp_groups = collect_warp_groups k.body in
  let pre_dispatch = filter_non_warp k.body |> filter_builtin_vars in
  let pre_s = emit_stmts ~depth:1 pre_dispatch in
  let warp_cases = List.map warp_groups ~f:(fun (role, body) ->
    let warp_ids = List.filter_map k.cluster.Cluster.warp_roles
      ~f:(fun (id, r) -> if Poly.(r = role) then Some id else None)
    in
    let cond = match warp_ids with
      | [] -> "false"
      | [id] -> Printf.sprintf "warp_id == %d" id
      | ids ->
        String.concat ~sep:" || "
          (List.map ids ~f:(Printf.sprintf "warp_id == %d"))
    in
    Printf.sprintf "  if (%s) {\n%s\n  }"
      cond (emit_stmts ~depth:2 body))
  in
  let warp_dispatch =
    if List.is_empty warp_cases then ""
    else
      String.concat ~sep:" else " warp_cases
  in
  let tiled_mma_s = emit_tiled_mma_decl k in
  let mbar_init_s = emit_mbar_init_loop k in

  Printf.sprintf
    "extern \"C\" %s__global__ __launch_bounds__(%d)\nvoid %s(\n  %s\n) {\n\
    \  extern __shared__ char smem_buf[];\n\
    \  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);\n\
    \  char* __smem_base = smem_buf;\n\
    \  (void)smem;\n\
    \  const int warp_id = threadIdx.x / 32;\n\
    \  const int lane_id = threadIdx.x %% 32;\n\
    \  (void)lane_id;\n\
     %s\n%s\n%s\n%s\n}\n"
    cluster_attr n_threads k.name (emit_params k)
    tiled_mma_s mbar_init_s pre_s warp_dispatch


let emit_host_launcher_params (k : tirix) : string =
  String.concat ~sep:",\n  "
    (List.map k.params ~f:(fun p ->
      let (Tensor t) = p.param_tensor in
      let elem_t = Elemtype.cpp_name t.tensor_elem_type in
      match p.param_name with
      | "M" | "N" | "K" ->
        Printf.sprintf "int %s" p.param_name
      | _ ->
        if p.param_is_tma then
          Printf.sprintf "CUtensorMap* %s_tmap" p.param_name
        else
          Printf.sprintf "const %s* %s" elem_t p.param_name))


let emit_host_launcher (k : tirix) : string =
  let is_blackwell =
    match k.family with Kernel_desc.Blackwell -> true | _ -> false
  in
  let kernel_args = String.concat ~sep:", "
    (List.map k.params ~f:(fun p ->
      if p.param_is_tma then Printf.sprintf "%s_tmap" p.param_name
      else p.param_name))
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
        \  cudaLaunchKernelEx(&cfg, %s, %s);"
        k.smem_bytes
        k.cluster.Cluster.dims.Cluster.x
        k.cluster.Cluster.dims.Cluster.y
        k.cluster.Cluster.dims.Cluster.z
        k.name kernel_args
    else
      Printf.sprintf
        "#ifndef __CUDACC_RTC__\n\
        \  %s<<<grid, block, %d>>>(%s);\n\
         #endif"
        k.name k.smem_bytes kernel_args
  in
  Printf.sprintf
    "void launch_%s(\n  %s\n) {\n\
    \  dim3 grid((M + %d - 1) / %d, (N + %d - 1) / %d, 1);\n\
    \  dim3 block(%d, 1, 1);\n\
    \  (void)block;\n\
     %s\n\
     }\n"
    k.name (emit_host_launcher_params k)
    k.bm k.bm k.bn k.bn
    (Cluster.thread_count k.cluster) launch


let emit_includes (k : tirix) : string =
  let arch_inc = match k.family with
    | Kernel_desc.Ampere -> "#include <cuda_fp16.h>\n"
    | Kernel_desc.Hopper -> "#include <cuda_bf16.h>\n"
    | Kernel_desc.Blackwell -> "#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n"
  in
  let lines = [
    "#pragma once";
    "#include <cute/tensor.hpp>";
    "#include <cute/atom/mma_atom.hpp>";
    "#include <cute/atom/copy_atom.hpp>";
    "#include <cute/algorithm/gemm.hpp>";
    "#include <cute/algorithm/copy.hpp>";
    "using namespace cute;";
    arch_inc;
    "struct CUtensorMap { alignas(64) uint64_t opaque[16]; };";
    "";
    "// SMEM descriptor helper";
    "__device__ inline uint64_t make_smem_desc_raw(";
    "    void* smem_ptr, uint32_t lead, uint32_t stride, uint32_t sw) {";
    "  uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);";
    "  return ((uint64_t)sw << 52) | ((uint64_t)stride << 36) |";
    "         ((uint64_t)lead << 16) | (addr >> 4);";
    "}";
    "";
    "// WGMMA matrix descriptor builder.";
    "// Receives a plain char* generic pointer (never a .shared symbol";
    "// reference directly) so NVRTC is forced to emit cvta.to.shared.u64";
    "// rather than folding away the conversion.";
    "__device__ __forceinline__ uint64_t make_wgmma_desc(uint32_t smem_offset, uint64_t or_bits) {";
    "  uint64_t desc;";
    "  asm volatile(";
    "    \"{ .reg .u64 saddr;\\n\\t\"";
    "    \".reg .u32 lo;\\n\\t\"";
    "    \"mov.u64 saddr, smem_buf;\\n\\t\"";
    "    \"cvt.u32.u64 lo, saddr;\\n\\t\"";
    "    \"add.u32 lo, lo, %1;\\n\\t\"";
    "    \"shr.u32 lo, lo, 4;\\n\\t\"";
    "    \"cvt.u64.u32 %0, lo;\\n\\t\"";
    "    \"or.b64 %0, %0, %2; }\\n\\t\"";
    "    : \"=l\"(desc) : \"r\"(smem_offset), \"l\"(or_bits));";
    "  return desc;";
    "}";
    "";
    "// TMA load helper";
    "__device__ inline void tma_2d_gmem2smem(";
    "    void* smem, const CUtensorMap* tmap, int x, int y, uint64_t* mbar) {";
    "     if (threadIdx.x % 32 == 0) {";
    "       asm volatile(";
    "         \"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes\"";
    "         \" [%0], [%1, {%2, %3}], [%4];\"";
    "         :: \"r\"((uint32_t)__cvta_generic_to_shared(smem)), \"l\"(tmap),";
    "         \"r\"(x), \"r\"(y), \"r\"((uint32_t)__cvta_generic_to_shared(mbar))";
    "          : \"memory\");";
    "     };";
    "}";
    "";
    "// TMA multicast helper";
    "__device__ inline void tma_2d_gmem2smem_multicast(";
    "    void* smem, const CUtensorMap* tmap, int x, int y,";
    "    uint64_t* mbar, uint16_t cta_mask) {";
    "    if (threadIdx.x % 32 == 0) {";
    "       asm volatile(";
    "       \"cp.async.bulk.tensor.2d.shared::cluster.global\"";
    "       \".mbarrier::complete_tx::bytes.multicast::cluster\"";
    "       \" [%0], [%1, {%2, %3}], [%4], %5;\"";
    "       :: \"r\"((uint32_t)__cvta_generic_to_shared(smem)), \"l\"(tmap),";
    "       \"r\"(x), \"r\"(y), \"l\"(mbar), \"h\"(cta_mask)";
    "       : \"memory\");";
    "     };";
    "}";
  ] in
  String.concat ~sep:"\n" lines


let emit (k : tirix) : Backend_cute.output =
  let includes = emit_includes k in
  let helpers = String.concat ~sep:"\n\n"
    (List.map k.helpers ~f:emit_helper) in
  let shared_storage = emit_shared_storage k in
  let kernel_func = emit_kernel_func k in
  let host_launcher  = emit_host_launcher k in
  let kernel_source = String.concat ~sep:"\n\n"
    [ includes; helpers; shared_storage; kernel_func ]
  in
  let full_source = String.concat ~sep:"\n\n"
    [ kernel_source; host_launcher ]
  in
  { Backend_cute.filename = k.name ^ ".cuh"
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
