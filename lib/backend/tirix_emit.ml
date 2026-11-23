open Base
open Tesserae_tirix
open Tesserae_pipeline
open Tesserae_core
open Tirix
open Tesserae_kernel
open Tesserae_atoms

let tirix_is_tma (k : tirix) =
  List.exists k.params ~f:(fun p -> p.param_is_tma)

let debug_consumer_delay_us = 0

let debug_pipeline_stats =
  match Sys.getenv "TESSERAE_PIPELINE_STATS" with
  | Some _ -> true
  | None -> false

let stats_cta_cond = "(blockIdx.x % 16 == 0 && blockIdx.y % 16 == 0)"

let stats_print_group (role_s : string) : string =
  String.concat ~sep:"\n" [
    "    if (lane_id == 0 && " ^ stats_cta_cond ^ ") {";
    "      const long long _tot = clock64() - _grp_t0;";
    "      printf(\"[stats] cta=(%d,%d) " ^ role_s
      ^ " total=%lld cyc  wait_full=%lld (%.1f%%)  wait_empty=%lld (%.1f%%)\\n\",";
    "             (int)blockIdx.x, (int)blockIdx.y, _tot,";
    "             _wait_full_mbar, 100.0 * (double)_wait_full_mbar / (double)_tot,";
    "             _wait_empty_mbar, 100.0 * (double)_wait_empty_mbar / (double)_tot);";
    "    }";
  ]

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
  | GridDim   X -> "gridDim.x"
  | GridDim   Y -> "gridDim.y"
  | GridDim   Z -> "gridDim.z"
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


let rec stmts_use_multicast (stmts : stmt list) : bool =
  List.exists stmts ~f:(function
    | SOp (Copy c) -> (match c.copy_kind with TmaMulticast -> true | _ -> false)
    | SSeq ss -> stmts_use_multicast ss
    | SFor { body; _ } -> stmts_use_multicast body
    | SIf (_, thn, els) -> stmts_use_multicast thn || stmts_use_multicast els
    | SWarpGroup (_, body) -> stmts_use_multicast body
    | SPragma (_, body) -> stmts_use_multicast body
    | SPipeline { prologue; mainloop; epilogue; _ } ->
      stmts_use_multicast prologue || stmts_use_multicast mainloop
      || stmts_use_multicast epilogue
    | _ -> false)

let uses_multicast (k : tirix) : bool = stmts_use_multicast k.body

(* True only in the even CTA of a 2-CTA cluster: the tcgen05.mma cta_group::2
   leader that issues the pair's MMAs and commits (PTX issue granularity:
   a single thread from the CTA pair initiates the base operation — issuing
   it from both CTAs runs every MMA twice). *)
let leader_cta_check =
  "([](){ uint32_t id; \
   asm volatile(\"mov.u32 %0, %%cluster_ctaid.x;\" : \"=r\"(id)); \
   return id; }() == 0)"

let emit_barrier (k : tirix) = function
  | CtaSync ->  "__syncthreads();"
  | WarpSync -> "__syncwarp();"
  | MemFence -> "__threadfence();"

  | MbarInit { mbar; count } ->
      Printf.sprintf {|
        asm volatile("mbarrier.init.shared::cta.b64 [%%0], %%1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))), "n"(%d) : "memory");
        asm volatile("fence.mbarrier_init.release.cta;");
      |} mbar.var_name count

  | MbarArriveExpect { mbar; bytes } ->
    if Cluster.is_2sm k.cluster && String.equal mbar.var_name "full_mbar" then
      (* 2-CTA pair: the MMA runs on the leader CTA, so every CTA's
         expect_tx (like its TMA completions) must land on the leader's
         full_mbar. mapa routes the arrive there; .shared::cluster is the
         address space of the mapped operand. *)
      Printf.sprintf {|
        if (threadIdx.x == 0) {
            uint32_t _mb = mapa_to_cta0((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t)));
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%%0], %%1;"
                :: "r"(_mb), "r"(%s) : "memory");
        }
      |} mbar.var_name (emit_expr bytes)
    else
      let scope = "cta" in
      Printf.sprintf {|
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.release.%s.shared::cta.b64 _, [%%0], %%1;"
                :: "r"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))), "r"(%s) : "memory");
        }
      |} scope mbar.var_name (emit_expr bytes)

  | MbarWaitParity { mbar; phase } ->
      let scope = "cta" in
      let wait_asm = Printf.sprintf {|
        asm volatile(
          "{"
          "  .reg .pred p;"
          "  LOOP_START:"
          "  mbarrier.try_wait.parity.acquire.%s.shared::cta.b64 p, [%%0], %%1, 0x989680;"
          "  @!p bra LOOP_START;"
          "}"
          :: "r"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))),
             "r"((uint32_t)(%s))
          : "memory");
      |} scope mbar.var_name (emit_expr phase) in
      if debug_pipeline_stats
         && (String.equal mbar.var_name "full_mbar"
             || String.equal mbar.var_name "empty_mbar")
      then
        Printf.sprintf "{ const long long _tw0 = clock64();\n%s\n_wait_%s += clock64() - _tw0; }"
          wait_asm mbar.var_name
      else wait_asm

  | MbarArrive { mbar } ->
        Printf.sprintf
          "if (threadIdx.x == 0) { \
             asm volatile(\"mbarrier.arrive.shared.b64 _ , [%%0];\" \
             :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))) \
                : \"memory\"); \
           }"
          mbar.var_name

  | ClusterArrive ->
    "asm volatile(\"barrier.cluster.arrive.relaxed.aligned;\");"

  | ClusterWait ->
    "asm volatile(\"barrier.cluster.wait.acquire.aligned;\");"

  | CpAsyncCommitGroup -> "asm volatile(\"cp.async.commit_group;\");"
  | CpAsyncWaitAll -> "asm volatile(\"cp.async.wait_all;\");"
  | Tcgen05Wait -> "asm volatile(\"tcgen05.wait::ld.sync.aligned;\");"
  | Tcgen05Fence -> "asm volatile(\"tcgen05.fence::after_thread_sync;\");"


let emit_copy (k: tirix) (c : copy) : string =
  let (Tensor src) = c.src_tensor in
  let (Tensor dst) = c.dst_tensor in

  let pred = match c.pred_expr with
    | None -> ""
    | Some p -> Printf.sprintf "if (%s) " (emit_expr p)
  in
  match c.copy_kind with
  | CpAsync ->
    Printf.sprintf
      "%sasm volatile(\"cp.async.ca.shared.global [%%0], [%%1], 16;\" \
       :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s))), \
          \"l\"((uint64_t)(uintptr_t)%s) : \"memory\");"
      pred dst.tensor_name src.tensor_name

  | TmaLoad ->
    let coord_k = match c.tma_coord_k with Some e -> emit_expr e | None -> "0" in
    let coord_m = match c.tma_coord_m with Some e -> emit_expr e | None -> "0" in
    let coord_n = match c.tma_coord_n with Some e -> emit_expr e | None -> "0" in
    let stage_off = match c.stage_var with
      | Some v -> Printf.sprintf " + (%s * (sizeof(smem.%s[0])))" v.var_name dst.tensor_name
      | None -> ""
    in
    let mbar_s = match c.mbar_var with
      | None -> "nullptr"
      | Some v ->
        Printf.sprintf "(uint64_t*)(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))" v.var_name
    in
    let is_b = String.is_substring dst.tensor_name ~substring:"_B" in
    let x_coord, y_coord =
      if is_b then (coord_k, coord_n) else (coord_k, coord_m)
    in
    (* 2-CTA pair: the load completion must signal the leader CTA's
       full_mbar (the barrier the pair's MMA warp waits on), so use the
       cta_group::2 TMA whose mbarrier may live in the peer CTA. *)
    let helper =
      if Cluster.is_2sm k.cluster then "tma_2d_gmem2smem_cg2"
      else "tma_2d_gmem2smem" in
    (* SW128 K-major smem sub-tiles are 64 elements (128B) wide — the same
       width as the sw128 TMA box. A stage wider than that along K is
       consecutive 64-wide sub-tiles, each filled by its own TMA op at
       coord + _tc*64, landing _tc sub-tiles further into the stage. *)
    let tile_k = match Layout.modes dst.tensor_layout with
      | [_; k] -> k | [_; _; k] -> k | _ -> 64 in
    let n_chunks = Int.max 1 (tile_k / 64) in
    if n_chunks = 1 then
      Printf.sprintf
        "%s%s(__smem_base + offsetof(SharedStorage, %s)%s, \
         %s_tmap, %s, %s, %s);"
        pred helper dst.tensor_name stage_off src.tensor_name x_coord y_coord mbar_s
    else
      Printf.sprintf
        "%sfor (int _tc = 0; _tc < %d; _tc++) { \
         %s(__smem_base + offsetof(SharedStorage, %s)%s + (_tc * (sizeof(smem.%s[0]) / %d)), \
         %s_tmap, (%s) + (_tc * 64), %s, %s); }"
        pred n_chunks helper dst.tensor_name stage_off dst.tensor_name n_chunks
        src.tensor_name x_coord y_coord mbar_s

  | TmaMulticast ->
    let coord_k = match c.tma_coord_k with Some e -> emit_expr e | None -> "0" in
    let coord_m = match c.tma_coord_m with Some e -> emit_expr e | None -> "0" in
    let coord_n = match c.tma_coord_n with Some e -> emit_expr e | None -> "0" in
    let is_b = String.is_substring dst.tensor_name ~substring:"_B" in
    let x_coord, y_coord =
      if is_b then (coord_k, coord_n) else (coord_k, coord_m)
    in
    let stage_off = match c.stage_var with
      | Some v -> Printf.sprintf " + (%s * (sizeof(smem.%s[0])))" v.var_name dst.tensor_name
      | None -> ""
    in
    let mbar_s = match c.mbar_var with
      | None -> "nullptr"
      | Some v ->
        Printf.sprintf "(uint64_t*)(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))" v.var_name
    in
    if Poly.(k.cluster.Cluster.dims.Cluster.x <> 2) then
      failwith (Printf.sprintf
        "TmaMulticast emit currently assumes a 2-CTA cluster along x; got (%d,%d,%d)."
        k.cluster.Cluster.dims.Cluster.x
        k.cluster.Cluster.dims.Cluster.y
        k.cluster.Cluster.dims.Cluster.z);
    let leader_check = leader_cta_check in
    let guard = match c.pred_expr with
      | None -> Printf.sprintf "(%s)" leader_check
      | Some p -> Printf.sprintf "((%s) && (%s))" (emit_expr p) leader_check
    in
    let mask = 0b11 in
    Printf.sprintf
      "if (%s) tma_2d_gmem2smem_multicast(__smem_base + offsetof(SharedStorage, %s)%s, \
       %s_tmap, %s, %s, %s, %d);"
      guard dst.tensor_name stage_off src.tensor_name x_coord y_coord mbar_s mask

  | RegToSmem ->
      let src_name =
        if String.equal src.tensor_name "acc" then "acc_frag"
        else src.tensor_name
      in
      (match dst.tensor_memspace with
        | Memspace.Global when Poly.(k.family = Kernel_desc.Blackwell)
                              && String.equal src.tensor_name "acc" ->
          let n_regs = k.bm * k.bn / 128 in
          let chunk = Int.min 32 n_regs in
          let n_chunks = (n_regs + chunk - 1) / chunk in
          if n_regs < 1 || n_regs > 256 || Int.bit_and n_regs (n_regs - 1) <> 0
              || (n_chunks > 1 && Int.bit_and chunk (chunk - 1) <> 0) then
            failwith (Printf.sprintf
              "emit_copy: tcgen05.ld 32x32b needs a power-of-two reg count <= 256, got %d" n_regs);
          let outs = String.concat ~sep:", "
            (List.init chunk ~f:(fun i -> Printf.sprintf "\"=r\"(_t[%d])" i)) in
          let regs = String.concat ~sep:", "
            (List.init chunk ~f:(fun i -> Printf.sprintf "%%%d" i)) in
          let row_base = match c.tma_coord_m with
            | Some e -> Printf.sprintf "((int)(%s))" (emit_expr e)
            | None -> Printf.sprintf "blockIdx.x * %d" k.bm
          in
          let col_base = match c.tma_coord_n with
            | Some e -> Printf.sprintf "((int)(%s))" (emit_expr e)
            | None -> Printf.sprintf "blockIdx.y * %d" k.bn
          in
          let epi_done_signal =
            "  if ((threadIdx.x % 32) == 0) {\n\
            \    asm volatile(\"mbarrier.arrive.shared.b64 _, [%0];\" :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, epilogue_done_mbar) + (tile % 2) * sizeof(uint64_t))) : \"memory\");\n\
            \  }"
          in
          (* C is passed as a grid_constant CUtensorMap param, named like
              the A/B maps: <tensor_name>_tmap (see emit_params). *)
          let c_tma_param = dst.tensor_name ^ "_tmap" in
          let chunk_body = String.concat ~sep:"\n" [
            Printf.sprintf "  uint32_t _t[%d];" chunk;
            Printf.sprintf
              "  asm volatile(\"tcgen05.ld.sync.aligned.32x32b.x%d.b32 {%s}, [%%%d];\" : %s : \"r\"(smem.tmem_addr[tile %% 2] + _c * %d) : \"memory\");"
              chunk regs chunk outs chunk;
            "  asm volatile(\"tcgen05.wait::ld.sync.aligned;\" ::: \"memory\");";
            "  const uint32_t _srow = (uint32_t)(32 * (warp_id % 4) + (int)(threadIdx.x % 32));";
            "  const uint32_t _epi_smem = (uint32_t)__cvta_generic_to_shared(smem.epilogue_smem);";
            "  #pragma unroll";
            Printf.sprintf "  for (int _v = 0; _v < %d; _v++) {" (chunk / 4);
            "    const uint32_t _swiz_v = (uint32_t)_v ^ (_srow & 0x7u);";
            "    const uint32_t _saddr = _epi_smem + _srow * 128u + _swiz_v * 16u;";
            "    asm volatile(\"st.shared.v4.b32 [%0], {%1, %2, %3, %4};\" :: \"r\"(_saddr), \"r\"(_t[_v * 4 + 0]), \"r\"(_t[_v * 4 + 1]), \"r\"(_t[_v * 4 + 2]), \"r\"(_t[_v * 4 + 3]) : \"memory\");";
            "  }";
            "  asm volatile(\"fence.proxy.async.shared::cta;\" ::: \"memory\");";
            "  asm volatile(\"bar.sync 1, 128;\" ::: \"memory\");";
            "  if ((warp_id % 4) == 0 && (threadIdx.x % 32) == 0) {";
            Printf.sprintf "    const int32_t _tma_x = %s + _c * %d;" col_base chunk;
            Printf.sprintf "    const int32_t _tma_y = %s;" row_base;
            Printf.sprintf
              "    asm volatile(\"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%%0, {%%1, %%2}], [%%3];\" :: \"l\"(%s), \"r\"(_tma_x), \"r\"(_tma_y), \"r\"(_epi_smem) : \"memory\");"
              c_tma_param;
            "    asm volatile(\"cp.async.bulk.commit_group;\" ::: \"memory\");";
            "    asm volatile(\"cp.async.bulk.wait_group.read 0;\" ::: \"memory\");";
            "  }";
            "  asm volatile(\"bar.sync 1, 128;\" ::: \"memory\");";
          ] in
          let tma_drain =
            "  if ((warp_id % 4) == 0 && (threadIdx.x % 32) == 0) {\n\
            \    asm volatile(\"cp.async.bulk.wait_group 0;\" ::: \"memory\");\n\
            \  }"
          in
          String.concat ~sep:"\n" [
            "{";
            Printf.sprintf "  const uint32_t _mbar = (uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, mma_done_mbar) + (tile %% 2) * sizeof(uint64_t));";
            (if debug_pipeline_stats then "  const long long _e0 = clock64();" else "");
            "  asm volatile(";
            "    \"{\\n\\t\"";
            "    \".reg .pred p;\\n\\t\"";
            "    \"MMA_DONE_WAIT:\\n\\t\"";
            "    \"mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, 0x989680;\\n\\t\"";
            "    \"@!p bra MMA_DONE_WAIT;\\n\\t\"";
            "    \"}\"";
            "    :: \"r\"(_mbar), \"r\"(0u) : \"memory\");";
            (if debug_pipeline_stats then "  const long long _e1 = clock64();" else "");
            "  asm volatile(\"tcgen05.fence::after_thread_sync;\" ::: \"memory\");";
            Printf.sprintf "  #pragma unroll\n  for (int _c = 0; _c < %d; _c++) {" n_chunks;
            chunk_body;
            "  }";
            tma_drain;
            epi_done_signal;
            (if debug_pipeline_stats then
                "  if (lane_id == 0 && warp_id == 2 && " ^ stats_cta_cond ^ ")\n"
                ^ "    printf(\"[stats] cta=(%d,%d) epilogue wait_mma_done=%lld cyc  tcgen05.ld+store=%lld cyc\\n\", (int)blockIdx.x, (int)blockIdx.y, _e1 - _e0, clock64() - _e1);"
              else "");
            "}"
          ]
       | Memspace.Global ->
         let n_regs = k.bm * k.bn / 128 in
         Printf.sprintf
           "{\n\
           \  const int _n = %d;\n\
           \  const int _local_tid = (int)threadIdx.x %% 128;\n\
           \  for (int _i = 0; _i < _n; _i++) {\n\
           \    %s[_local_tid * _n + _i] = %s[_i];\n\
           \  }\n\
            }"
           n_regs dst.tensor_name src_name
       | _ -> Printf.sprintf "cute::copy(%s, %s);" src_name dst.tensor_name)

  | SmemToReg ->
    Printf.sprintf "cute::copy(%s, %s);" src.tensor_name dst.tensor_name

  | SmemToGlobal ->
    let pred_guard = match c.pred_expr with
      | None -> "" | Some p -> Printf.sprintf "if (%s) " (emit_expr p)
    in
    Printf.sprintf
      "%s{\n\
      \  const int _tid = (int)threadIdx.x;\n\
      \  const int _nthreads = (int)blockDim.x;\n\
      \  const int _n_half2  = (int)(sizeof(smem.%s) / sizeof(__half2));\n\
      \  __half2* _smem_ptr = reinterpret_cast<__half2*>(&smem.%s[0]);\n\
      \  __half2* _dst_ptr = reinterpret_cast<__half2*>(const_cast<__half*>(%s)\n\
      \                        + (blockIdx.x * %d + blockIdx.y * %d));\n\
      \  for (int _i = _tid; _i < _n_half2; _i += _nthreads) {\n\
      \    _dst_ptr[_i] = _smem_ptr[_i];\n\
      \  }\n\
      }"
      pred_guard
      src.tensor_name
      src.tensor_name
      dst.tensor_name
      (Layout.size src.tensor_layout)
      (Layout.size src.tensor_layout)


let sm100_idesc ~(m : int) ~(n : int) ~(a_fmt : int) ~(b_fmt : int) : int =
  let c_fmt = 1 in
  Int.bit_or (Int.shift_left c_fmt 4)
    (Int.bit_or (Int.shift_left a_fmt 7)
       (Int.bit_or (Int.shift_left b_fmt 10)
          (Int.bit_or (Int.shift_left (n / 8) 17)
             (Int.shift_left (m / 16) 24))))

let emit_mma (m : mma_desc) : string =
  let (Tensor a) = m.tensor_a in
  let (Tensor b) = m.tensor_b in
  let (Tensor _c) = m.tensor_c in
  let accum = if m.accum_flag then 1 else 0 in
  match m.mma_kind with
  | Sm80Mma ->
    let _atom_str = match m.mma_atom with
      | Atom80 a -> Mma_atom.emit_cpp a
      | _ -> failwith "emit_mma: Sm80Mma requires Atom80"
    in
    let a_modes = Layout.modes a.tensor_layout in
    let b_modes = Layout.modes b.tensor_layout in
    let a_m, a_k = match a_modes with
      | [m; k] -> (m, k) | [_; m; k] -> (m, k) | _ -> (16, 16)
    in
    let b_n, b_k = match b_modes with
      | [n; k] -> (n, k) | [_; n; k] -> (n, k) | _ -> (8, 16)
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
      a.tensor_name a_m a_k b.tensor_name b_n b_k

  | Sm90Wgmma ->
        let (Tensor a) = m.tensor_a in
        let (Tensor b) = m.tensor_b in
        let (mn_str, at, bt, num_regs, atom_k) =
          match m.mma_atom with
          | Atom90 atom ->
              let (_, n, k) = Mma_atom.shape atom in
              let num_regs = n / 2 in
              let at = String.lowercase (Mma_atom.elem_string atom.Mma_atom.a_type) in
              let bt = String.lowercase (Mma_atom.elem_string atom.Mma_atom.b_type) in
              (Printf.sprintf "m64n%dk%d" n k, at, bt, num_regs, k)
          | _ -> failwith "emit_mma: Sm90Wgmma requires Atom90"
        in
        let a_modes = Layout.modes a.tensor_layout in
        let a_m = match a_modes with [m; _] -> m | [_; m; _] -> m | _ -> 64 in
        let a_k = match a_modes with [_; k] -> k | [_; _; k] -> k | _ -> 16 in
        let b_modes = Layout.modes b.tensor_layout in
        let _b_n = match b_modes with [_; n] -> n | [_; _; n] -> n | _ -> 128 in
        let k_fragments = a_k / atom_k in
        let elem_bytes = 2 in
        let n_tiles_m = a_m / 64 in
        let sub_tile_bytes = 64 * a_k * 2 in
        let a_lbo = a_k * elem_bytes in
        let b_lbo = a_k * elem_bytes in
        let a_kfrag_bytes = atom_k * elem_bytes in
        let b_kfrag_bytes = atom_k * elem_bytes in
        let reg_tokens =
          List.init num_regs ~f:(fun i -> Printf.sprintf "%%%d" i)
          |> String.concat ~sep:", "
        in
        let ref_decls =
          let total = n_tiles_m * num_regs in
          List.init total ~f:(fun i -> Printf.sprintf "  float& _f%d = acc_frag[%d];" i i)
          |> String.concat ~sep:"\n"
        in
        let sub_tile_strs =
          List.concat_map (List.init n_tiles_m ~f:Fn.id) ~f:(fun tile_m ->
              List.init k_fragments ~f:(fun kfrag ->
                let a_offset = tile_m * sub_tile_bytes + kfrag * a_kfrag_bytes in
                let b_offset = kfrag * b_kfrag_bytes in
                let desc_a_expr =
                  Printf.sprintf
                    "make_wgmma_desc(__smem_base_u32, (uint32_t)(offsetof(SharedStorage,%s) + stage*sizeof(smem.%s[0]) + %d), %d, 0, 0)"
                    a.tensor_name a.tensor_name a_offset a_lbo
                in
                let desc_b_expr =
                  Printf.sprintf
                    "make_wgmma_desc(__smem_base_u32, (uint32_t)(offsetof(SharedStorage,%s) + stage*sizeof(smem.%s[0]) + %d), %d, 0, 3)"
                    b.tensor_name b.tensor_name b_offset b_lbo
                in
                let reg_base = tile_m * num_regs in
                let acc_outputs =
                  List.init num_regs ~f:(fun i ->
                    Printf.sprintf "\"+f\"(_f%d)" (reg_base + i))
                  |> String.concat ~sep:", "
                in
                Printf.sprintf
                  "    {\n\
                  \      uint64_t desc_a = %s;\n\
                  \      uint64_t desc_b = %s;\n\
                  \      asm volatile(\"wgmma.fence.sync.aligned;\");\n\
                  \      asm volatile(\n\
                  \        \"wgmma.mma_async.sync.aligned.%s.f32.%s.%s {%s}, %%%d, %%%d, %d, 1, 1, 0, 0;\"\n\
                  \        : %s : \"l\"(desc_a), \"l\"(desc_b));\n\
                  \      asm volatile(\"wgmma.commit_group.sync.aligned;\");\n\
                  \      asm volatile(\"wgmma.wait_group.sync.aligned 0;\");\n\
                  \    }"
                  desc_a_expr desc_b_expr mn_str at bt reg_tokens
                  num_regs (num_regs + 1) accum acc_outputs))
        in
        Printf.sprintf "{\n%s\n%s\n}" ref_decls (String.concat ~sep:"\n" sub_tile_strs)

  | Sm100Tcgen05 ->
    let (atom_m, atom_n, atom_k, kind, a_fmt, b_fmt) =
      match m.mma_atom with
      | Atom100 atom ->
        let (am, an, ak) = Mma_atom.shape atom in
        let fmt_of t =
          match String.lowercase (Mma_atom.elem_string t) with
          | "f16"  -> ("f16", 0)
          | "bf16" -> ("f16", 1)
          | "tf32" -> ("tf32", 2)
          | other  -> failwith (Printf.sprintf "emit_mma: unsupported operand type %s" other)
        in
        let (ka, fa) = fmt_of atom.Mma_atom.a_type in
        let (kb, fb) = fmt_of atom.Mma_atom.b_type in
        if not (String.equal ka kb) then
          failwith "emit_mma: Sm100Tcgen05 A and B must use the same .kind";
        (am, an, ak, ka, fa, fb)
      | _ -> failwith "emit_mma: Sm100Tcgen05 requires Atom100"
    in
    if atom_m <> 128 && atom_m <> 256 then
      failwith (Printf.sprintf "emit_mma: tcgen05 cta_group::2 needs M=128|256, got %d" atom_m);
    (* idesc M is the PAIR M: a cta_group::2 instruction covers two per-CTA
       halves along M (even CTA = rows 0..M/2-1, odd = M/2..M-1), so an
       atom_m=128 half tile encodes M=256 in the instruction descriptor. *)
    let idesc_m = if atom_m = 128 then 256 else atom_m in
    let idesc = sm100_idesc ~m:idesc_m ~n:atom_n ~a_fmt ~b_fmt in
    let mode_rows t = match Layout.modes t.tensor_layout with
      | [r; _] -> r | [r; _; _] -> r | _ -> 64 in
    let tile_k = match Layout.modes a.tensor_layout with
      | [_; k] -> k | [_; _; k] -> k
      | _ -> failwith "emit_mma: cannot determine tile K"
    in
    let elem_bytes = if String.equal kind "tf32" then 4 else 2 in
    let k_steps = tile_k / atom_k in
    let desc_step = atom_k * elem_bytes / 16 in
    (* SW128 K-major sub-tiles are 64 elements (128B) wide — the TMA box
       width. A stage wider than that along K is consecutive 64-wide
       sub-tiles (one TMA op each), so the descriptor hops rows*64*elem
       bytes between sub-tiles and desc_step within them. *)
    let subtile_k = 64 in
    let per_subtile = subtile_k / atom_k in
    let subtile_desc = mode_rows a * subtile_k * elem_bytes / 16 in
    let subtile_desc_b = mode_rows b * subtile_k * elem_bytes / 16 in
    let desc_a = Printf.sprintf "make_sm100_smem_desc(&smem.%s[stage][0])" a.tensor_name in
    let desc_b = Printf.sprintf "make_sm100_smem_desc(&smem.%s[stage][0])" b.tensor_name in
    String.concat ~sep:"\n" [
      "{";
      "  uint32_t tmem_addr = smem.tmem_addr[tile % 2];";
      Printf.sprintf "  uint64_t desc_a = %s;" desc_a;
      Printf.sprintf "  uint64_t desc_b = %s;" desc_b;
      Printf.sprintf "  uint32_t idesc  = 0x%08xu;" idesc;
      "  if ((threadIdx.x % 32) == 0 && cooperative_groups::cluster_group::block_rank() == 0) {";
      (if debug_consumer_delay_us > 0
       then Printf.sprintf "    for (int _w = 0; _w < %d; _w++) __nanosleep(1000);" debug_consumer_delay_us
       else "");
      "    #pragma unroll";
      Printf.sprintf "    for (int _kk = 0; _kk < %d; _kk++) {" k_steps;
      Printf.sprintf "      uint64_t _da = desc_a + (uint64_t)(((_kk / %d) * %d) + ((_kk %% %d) * %d));"
        per_subtile subtile_desc per_subtile desc_step;
      Printf.sprintf "      uint64_t _db = desc_b + (uint64_t)(((_kk / %d) * %d) + ((_kk %% %d) * %d));"
        per_subtile subtile_desc_b per_subtile desc_step;
      "      asm volatile(";
      "        \"{\\n\\t\"";
      "        \".reg .pred p;\\n\\t\"";
      "        \"setp.ne.b32 p, %4, 0;\\n\\t\"";
      Printf.sprintf "        \"tcgen05.mma.cta_group::2.kind::%s [%%0], %%1, %%2, %%3, p;\\n\\t\"" kind;
      "        \"}\"";
      "        :: \"r\"(tmem_addr), \"l\"(_da), \"l\"(_db), \"r\"(idesc), \"r\"(_mma_acc)";
      "        : \"memory\");";
      "      _mma_acc = 1u;";
      "    }";
      "  }";
      "}";
    ]


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

let emit_tiled_mma_decl (k : tirix) : string =
  let atoms = collect k.body in
  let has_sm90_or_100 = List.exists atoms ~f:(function Atom90 _ | Atom100 _ -> true | _ -> false) in
  if has_sm90_or_100 then ""
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
        \  clear(acc_frag);\n"
        atom_str k.bm k.bn


let emit_op (k:tirix) = function
  | Copy c -> emit_copy k c
  | Mma  m -> emit_mma m
  | Barrier b -> emit_barrier k b
  | TmemAlloc { addr_var; col_count } ->
    Printf.sprintf
      "asm volatile(\"tcgen05.alloc.cta_group::2.sync.aligned.\
       shared::cta.b32 [%%0], %d;\" :: \"r\"((uint32_t)__cvta_generic_to_shared(&smem.%s)) : \"memory\");"
      col_count addr_var.var_name

  | TmemDealloc { addr_var; col_count } ->
    Printf.sprintf
      "asm volatile(\"tcgen05.dealloc.cta_group::2.sync.aligned.b32 %%0, %d;\" :: \"r\"(smem.%s[0]) : \"memory\");"
      col_count addr_var.var_name

  | TmemLoad { dst_vars; src_addr; col_offset } ->
    let n = List.length dst_vars in
    let outs = String.concat ~sep:", " (List.init n ~f:(fun i -> Printf.sprintf "\"=r\"(__t[%d])" i)) in
    let regs = String.concat ~sep:", " (List.init n ~f:(fun i -> Printf.sprintf "%%%d" i)) in
    let copies = String.concat ~sep:" "
      (List.mapi dst_vars ~f:(fun i v -> Printf.sprintf "%s = __uint_as_float(__t[%d]);" v.var_name i)) in
    Printf.sprintf
      "{ uint32_t __t[%d]; asm volatile(\"tcgen05.ld.sync.aligned.32x32b.x%d.b32 {%s}, [%%%d];\" : %s : \"r\"((uint32_t)(%s) + %d) : \"memory\"); %s }"
      n n regs n outs (emit_expr src_addr) col_offset copies

  | TmemCommit { mbar_var; cta_mask } ->
    let is_stage_array =
      String.equal mbar_var.var_name "full_mbar"
      || String.equal mbar_var.var_name "empty_mbar"
    in
    let mbar =
      if is_stage_array then
        Printf.sprintf "(uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, %s) + stage * sizeof(uint64_t))" mbar_var.var_name
      else Printf.sprintf "(uint32_t)__cvta_generic_to_shared(&smem.%s)" mbar_var.var_name
    in
    let inst = match cta_mask with
      | None ->
        Printf.sprintf "asm volatile(\"tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%%0];\" :: \"r\"(%s) : \"memory\");" mbar
      | Some mask ->
        Printf.sprintf "asm volatile(\"tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%%0], %%1;\" :: \"r\"(%s), \"h\"((uint16_t)%d) : \"memory\");" mbar mask
    in
    Printf.sprintf "if ((threadIdx.x %% 32) == 0 && cooperative_groups::cluster_group::block_rank() == 0) { %s }" inst

  | SmemDescInit { desc_var; ptr_expr; leading_dim; stride; swizzle } ->
    let sw_bits = Smem_desc.swizzle_mode_bits (Smem_desc.swizzle_mode_of swizzle) in
    Printf.sprintf "uint64_t %s = make_smem_desc_raw(%s, %d, %d, %d);"
      desc_var.var_name (emit_expr ptr_expr) leading_dim stride sw_bits


let indent (depth : int) : string = String.make (depth * 2) ' '


let rec emit_stmt (k: tirix) ?(depth = 0) ?stage_depth  (s : stmt) : string =
  let ind = indent depth in
  match s with
  | SEmpty -> ""
  | SLet (v, e) ->
    Printf.sprintf "%sconst %s %s = %s;" ind (emit_packed_scalar v.var_type) v.var_name (emit_packed_expr e)
  | SLetMut (v, e) ->
    Printf.sprintf "%s%s %s = %s;" ind (emit_packed_scalar v.var_type) v.var_name (emit_packed_expr e)
  | SAssign (v, e) ->
    Printf.sprintf "%s%s = %s;" ind v.var_name (emit_packed_expr e)
  | SOp op -> Printf.sprintf "%s%s" ind (emit_op k op)

  | SFor { var; start; stop; step; dir = _; unroll; body } ->
    let pragma = if unroll then Printf.sprintf "%s#pragma unroll\n" ind else "" in
    let stage_inc = "" in
    let acc_reset =
      if String.equal var.var_name "tile" && Poly.(k.family = Kernel_desc.Blackwell)
      then Printf.sprintf "%s  _mma_acc = 0u;\n" ind
      else ""
    in
    Printf.sprintf "%s%sfor (%s %s = %s; %s < %s; %s += %s) {\n%s%s%s\n%s}"
      pragma ind (emit_packed_scalar var.var_type) var.var_name (emit_expr start)
      var.var_name (emit_expr stop) var.var_name (emit_expr step)
      acc_reset (emit_stmts k ~depth:(depth+1) ?stage_depth body) stage_inc ind

  | SIf (cond, thn, els) ->
    let thn_s = emit_stmts k ~depth:(depth+1) ?stage_depth thn in
    let els_s = match els with
      | [] -> ""
      | _  -> Printf.sprintf " else {\n%s\n%s}"
                 (emit_stmts k ~depth:(depth+1) ?stage_depth els) ind
    in
    Printf.sprintf "%sif (%s) {\n%s\n%s}%s" ind (emit_expr cond) thn_s ind els_s

  | SSeq stmts -> emit_stmts k ~depth ?stage_depth stmts

  | SPipeline { stages; prologue; mainloop; epilogue } ->
    String.concat ~sep:"\n" [
      Printf.sprintf "%s// pipeline prologue (depth=%d)" ind stages;
      emit_stmts k ~depth ?stage_depth prologue;
      Printf.sprintf "%s// pipeline mainloop" ind;
      emit_stmts k ~depth ?stage_depth mainloop;
      Printf.sprintf "%s// pipeline epilogue" ind;
      emit_stmts k ~depth ?stage_depth epilogue;
    ]
  | SWarpGroup (role, body) ->
    let role_s = match role with
      | Cluster.Producer -> "producer" | Cluster.Consumer -> "consumer"
      | Cluster.Epilogue -> "epilogue" | Cluster.Scheduler -> "scheduler"
    in
    Printf.sprintf "%s// warp role: %s\n%s" ind role_s (emit_stmts k ~depth body)

  | SPragma (pragma, body) ->
    Printf.sprintf "%s#pragma %s\n%s" ind pragma (emit_stmts k ~depth body)

and emit_stmts (k: tirix) ?(depth = 0) ?stage_depth (stmts : stmt list) : string =
  String.concat ~sep:"\n"
    (List.filter_map stmts ~f:(fun s ->
      let r = emit_stmt k ~depth ?stage_depth s in
      if String.is_empty r then None else Some r))


let emit_shared_storage (k : tirix) : string =
  let mbar_names = ["full_mbar"; "empty_mbar"] in
  let tensor_decls = List.filter_map k.tensors
    ~f:(fun (name, Tensor t) ->
      if List.mem mbar_names name ~equal:String.equal then None
      else match t.tensor_memspace with
      | Memspace.Shared ->
        let elem_t = Elemtype.cpp_name t.tensor_elem_type in
        let per_stage = Layout.size t.tensor_layout in
        if tirix_is_tma k then
          Some (Printf.sprintf "  __align__(1024) %s %s[%d][%d];" elem_t name k.pipeline_depth per_stage)
        else
          Some (Printf.sprintf "  %s %s[%d];" elem_t name per_stage)
      | _ -> None)
  in
  let mbar_decls =
    if tirix_is_tma k then
      let has_empty = List.exists k.tensors ~f:(fun (n, _) -> String.equal n "empty_mbar") in
      let full = [Printf.sprintf "  __align__(8) uint64_t full_mbar[%d];" k.pipeline_depth] in
      let empty = if has_empty then [Printf.sprintf "  __align__(8) uint64_t empty_mbar[%d];" k.pipeline_depth] else [] in
      full @ empty
    else []
  in
  let tmem_decl =
    match k.family with
    | Kernel_desc.Blackwell ->
      [ "  __align__(16) uint32_t tmem_addr[2];"
      ; "  __align__(8) uint64_t mma_done_mbar[2];"
      ; "  __align__(8) uint64_t epilogue_done_mbar[2];"
      (* 1024B-aligned: the SW128 swizzle pattern is anchored at absolute
         1024B boundaries (addr bits [9:7]), so a tile base sitting mid-atom
         shifts the pattern phase and the TMA store reads permuted chunks. *)
      ; "  __align__(1024) float epilogue_smem[128 * 32];"]
    | _ -> []
  in
  let all = tensor_decls @ mbar_decls @ tmem_decl in
  Printf.sprintf "struct __align__(128) SharedStorage {\n%s\n};" (String.concat ~sep:"\n" all)


let emit_helper (k: tirix) (h : helper_func) : string =
  let builtin_helpers = ["tma_2d_gmem2smem"; "tma_2d_gmem2smem_multicast"; "make_smem_desc_raw"; "make_smem_desc"] in
  if List.mem builtin_helpers h.hf_name ~equal:String.equal then ""
  else
  let params = String.concat ~sep:", "
    (List.map h.hf_params ~f:(fun v -> Printf.sprintf "%s %s" (emit_packed_scalar v.var_type) v.var_name))
  in
  Printf.sprintf "__device__ inline %s %s(%s) {\n%s\n}"
    (emit_packed_scalar h.hf_ret_type) h.hf_name params
    (emit_stmts k ~depth:1 h.hf_body)

let emit_params (k : tirix) : string =
  String.concat ~sep:",\n  "
    (List.map k.params ~f:(fun p ->
      let (Tensor t) = p.param_tensor in
      let elem_t = Elemtype.cpp_name t.tensor_elem_type in
      match p.param_name with
      | "M" | "N" | "K" -> Printf.sprintf "int %s" p.param_name
      | _ ->
        if p.param_is_tma then
          Printf.sprintf "const CUtensorMap* const __grid_constant__ %s_tmap" p.param_name
        else Printf.sprintf "%s* %s" elem_t p.param_name))

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
        String.equal v.var_name "warp_id" || String.equal v.var_name "lane_id" -> None
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

let rec count_arrives (stmts : stmt list) : int =
  List.sum (module Int) stmts ~f:(function
    | SOp (Barrier (MbarArriveExpect _)) -> 1
    | SSeq ss -> count_arrives ss
    | SFor { body; _ } -> count_arrives body
    | SIf (_, thn, els) -> max (count_arrives thn) (count_arrives els)
    | SWarpGroup (_, body) -> count_arrives body
    | SPipeline { prologue; mainloop; epilogue; _ } ->
        count_arrives prologue + count_arrives mainloop + count_arrives epilogue
    | _ -> 0)

let rec has_tmem_alloc = function
  | [] -> false
  | SOp (TmemAlloc _) :: _ -> true
  | SWarpGroup (_, body) :: rest -> has_tmem_alloc body || has_tmem_alloc rest
  | SIf (_, thn, els) :: rest -> has_tmem_alloc thn || has_tmem_alloc els || has_tmem_alloc rest
  | SFor { body; _ } :: rest -> has_tmem_alloc body || has_tmem_alloc rest
  | SPipeline { prologue; mainloop; epilogue; _ } :: rest ->
    has_tmem_alloc prologue || has_tmem_alloc mainloop || has_tmem_alloc epilogue || has_tmem_alloc rest
  | SSeq ss :: rest -> has_tmem_alloc ss || has_tmem_alloc rest
  | _ :: rest -> has_tmem_alloc rest

let emit_mbar_init_loop (k : tirix) : string =
  if not (tirix_is_tma k) then ""
  else
    let depth = k.pipeline_depth in
    let has_empty = List.exists k.tensors ~f:(fun (n, _) -> String.equal n "empty_mbar") in
    let consumer_count = List.count k.cluster.Cluster.warp_roles
      ~f:(fun (_, role) -> Poly.(role = Cluster.Consumer)) in
    let cta_count =
      k.cluster.Cluster.dims.Cluster.x
      * k.cluster.Cluster.dims.Cluster.y
      * k.cluster.Cluster.dims.Cluster.z in
    let empty_count =
      (if consumer_count = 0 then 1 else consumer_count)
      * (if uses_multicast k then max 1 cta_count else 1) in
    let empty_init = if has_empty then
      Printf.sprintf
        "      asm volatile(\"mbarrier.init.shared.b64 [%%0], %%1;\"\n\
        \        :: \"r\"((uint32_t)__cvta_generic_to_shared(\n\
        \            __smem_base + offsetof(SharedStorage, empty_mbar) + s * sizeof(uint64_t))),\n\
        \         \"n\"(%d) : \"memory\");\n"
        empty_count
      else "" in
    let cluster_sync =
      if Poly.(k.family = Kernel_desc.Blackwell) then
        String.concat ~sep:"" [
          "\n  asm volatile(\"barrier.cluster.arrive.release.aligned;\" ::: \"memory\");";
          "\n  asm volatile(\"barrier.cluster.wait.acquire.aligned;\" ::: \"memory\");" ]
      else "" in
    (* Init both mma_done_mbar (count 1: one commit per MMA warp) and
       epilogue_done_mbar (count 4: one arrive per epilogue warp). *)
    let done_init =
      if Poly.(k.family = Kernel_desc.Blackwell) then
        "    for(int i=0; i<2; i++) { asm volatile(\"mbarrier.init.shared.b64 [%0], 1;\" :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, mma_done_mbar) + i * sizeof(uint64_t))) : \"memory\"); }\n\
        \    for(int i=0; i<2; i++) { asm volatile(\"mbarrier.init.shared.b64 [%0], 4;\" :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, epilogue_done_mbar) + i * sizeof(uint64_t))) : \"memory\"); }\n"
      else "" in
    (* 2-CTA pair: every CTA's expect_tx arrives (like its TMA completions)
       are routed to the leader CTA's full_mbar, so the arrival count is the
       per-CTA arrive count times the CTA count in the cluster. *)
    let full_count =
      max 1 (count_arrives k.body)
      * (if Cluster.is_2sm k.cluster then cta_count else 1) in
    let tmem_alloc_s =
      if Poly.(k.family = Kernel_desc.Blackwell) && not (has_tmem_alloc k.body) then
        Printf.sprintf
          "  if (warp_id == 0) {\n\
          \    asm volatile(\"tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%%0], %d;\"\n\
          \      :: \"r\"((uint32_t)__cvta_generic_to_shared(&smem.tmem_addr[0])) : \"memory\");\n\
          \    asm volatile(\"tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%%0], %d;\"\n\
          \      :: \"r\"((uint32_t)__cvta_generic_to_shared(&smem.tmem_addr[1])) : \"memory\");\n\
          \    asm volatile(\"tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\" ::: \"memory\");\n\
          \  }\n"
          k.bn k.bn
      else "" in
    Printf.sprintf
      "  if (warp_id == 0 && lane_id == 0) {\n\
      \    #pragma unroll\n\
      \    for (int s = 0; s < %d; s++) {\n\
      \      asm volatile(\"mbarrier.init.shared.b64 [%%0], %%1;\"\n\
      \        :: \"r\"((uint32_t)__cvta_generic_to_shared(\n\
      \            __smem_base + offsetof(SharedStorage, full_mbar) + s * sizeof(uint64_t))),\n\
      \         \"n\"(%d) : \"memory\");\n\
      %s\
      \    }\n\
      %s\
      \    asm volatile(\"fence.mbarrier_init.release.cluster;\");\n\
      \  }\n\
      %s\
      \  __syncthreads();%s"
      depth full_count empty_init done_init tmem_alloc_s cluster_sync


let is_acc_global_store (s : stmt) : bool =
  match s with
  | SOp (Copy c) ->
    (match c.copy_kind with
     | RegToSmem ->
       let (Tensor src) = c.src_tensor in
       let (Tensor dst) = c.dst_tensor in
       String.equal src.tensor_name "acc"
       && String.equal (Memspace.name dst.tensor_memspace) "global"
     | _ -> false)
  | _ -> false

let rec extract_acc_stores (stmts : stmt list) : stmt list * stmt list =
  List.fold stmts ~init:([], []) ~f:(fun (rest, stores) s ->
    if is_acc_global_store s then (rest, stores @ [s])
    else
      match s with
      | SSeq ss ->
        let (r, st) = extract_acc_stores ss in
        ((if List.is_empty r then rest else rest @ [SSeq r]), stores @ st)
      | _ -> (rest @ [s], stores))

let epilogue_warp_ids (k : tirix) : int list =
  let ids = List.filter_map k.cluster.Cluster.warp_roles
    ~f:(fun (id, r) -> if Poly.(r = Cluster.Epilogue) then Some id else None) in
  let quads = List.dedup_and_sort ~compare:Int.compare (List.map ids ~f:(fun i -> i % 4)) in
  if not (List.equal Int.equal quads [0; 1; 2; 3]) then
    failwith (Printf.sprintf
      "Blackwell epilogue needs Epilogue warps covering all 4 TMEM lane quadrants; have warps [%s]."
      (String.concat ~sep:"," (List.map ids ~f:Int.to_string)));
  ids

let emit_mma_done_signal () : string =
  String.concat ~sep:"\n" [
    "  if ((threadIdx.x % 32) == 0) {";
    (* Multicast to both CTAs of the pair: only the leader CTA issues the
       commit, but each CTA's epilogue warps wait on their local
       mma_done_mbar (one arrival each). *)
    "    asm volatile(\"tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\"";
    "      :: \"r\"((uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, mma_done_mbar) + (tile % 2) * sizeof(uint64_t))), \"h\"((uint16_t)3) : \"memory\");";
    "  }";
  ]

let emit_kernel_func (k : tirix) : string =
  let n_threads = Cluster.thread_count k.cluster in
  let cluster_attr =
    if Cluster.is_2sm k.cluster then
      Printf.sprintf "__cluster_dims__(%d, %d, %d) "
        k.cluster.Cluster.dims.Cluster.x
        k.cluster.Cluster.dims.Cluster.y
        k.cluster.Cluster.dims.Cluster.z
    else "" in
  let warp_groups = collect_warp_groups k.body in
  let pre_dispatch = filter_non_warp k.body |> filter_builtin_vars in
  let sd = if tirix_is_tma k then Some k.pipeline_depth else None in
  let pre_s = emit_stmts k ~depth:1 ?stage_depth:sd pre_dispatch in
  let sm90_acc_decl =
    let per_thread = (k.bm * k.bn) / 128 in
    let atoms = List.concat_map warp_groups ~f:(fun (_, body) -> collect body) in
    let has_acc = List.exists atoms ~f:(function Atom90 _ | Atom100 _ -> true | _ -> false) in
    if has_acc then
      Printf.sprintf "    float acc_frag[%d];\n    for (int i = 0; i < %d; i++) acc_frag[i] = 0.0f;\n"
        per_thread per_thread
    else "" in
  let is_bw = Poly.(k.family = Kernel_desc.Blackwell) in
  let split = List.map warp_groups ~f:(fun (role, body) ->
    match role with
    | Cluster.Consumer when is_bw ->
      let (rest, stores) = extract_acc_stores body in
      (role, rest, stores)
    | _ -> (role, body, []))
  in
  let epi_stores = List.concat_map split ~f:(fun (_, _, st) -> st) in
  let has_epi_warps =
    List.exists k.cluster.Cluster.warp_roles ~f:(fun (_, r) -> Poly.(r = Cluster.Epilogue)) in
  let tile_var = { Tirix.var_name = "tile"; var_id = 999; var_type = Tirix.Scalar Tirix.S32; var_mutable = true } in
  let epi_stores =
    if is_bw && not (List.is_empty epi_stores) then
      [ Tirix.SFor {
        var = tile_var;
        start = Tirix.Const (Tirix.S32, 0l);
        stop = Tirix.Const (Tirix.S32, 2l);
        step = Tirix.Const (Tirix.S32, 1l);
        dir = Tirix.Upto;
        unroll = false;
        body = epi_stores;
      } ]
    else epi_stores in
  let need_epilogue = is_bw && (has_epi_warps || not (List.is_empty epi_stores)) in
  let warp_cases = List.map split ~f:(fun (role, body, _) ->
    let warp_ids = List.filter_map k.cluster.Cluster.warp_roles
      ~f:(fun (id, r) -> if Poly.(r = role) then Some id else None) in
    let cond = match warp_ids with
      | [] -> "false" | [id] -> Printf.sprintf "warp_id == %d" id
      | ids -> String.concat ~sep:" || " (List.map ids ~f:(Printf.sprintf "warp_id == %d"))
    in
    (* 2-CTA tcgen05.mma is issued by the leader CTA only; the MMA warp of
       the peer CTA goes fully idle (its A/B halves are loaded by its own
       producer and read by the leader's pair MMA, which its own epilogue
       drains). *)
    let cond =
      if Poly.(role = Cluster.Consumer) && Poly.(k.family = Kernel_desc.Blackwell)
      then Printf.sprintf "((%s) && (%s))" cond leader_cta_check
      else cond in
    let prefix = match role with
      | Cluster.Consumer -> if is_bw then "" else sm90_acc_decl
      | _ -> "" in
    let body_s = emit_stmts k ~depth:2 ?stage_depth:sd body in
    let body_s =
      if need_epilogue && Poly.(role = Cluster.Consumer) then
        match body with
        | [Tirix.SFor { var; start; stop; step; dir = _dir; unroll = _unroll; body = loop_body }]
          when String.equal var.var_name "tile" ->
          let ind = indent 2 in
          let loop_s = emit_stmts k ~depth:3 ?stage_depth:sd loop_body in
          (* Wait for the previous epilogue that used this same TMEM slot.
             For tile 0 and 1 the slot is fresh (no prior epilogue), so skip.
             For tile t >= 2 we wait on epilogue_done_mbar[t % 2] at
             phase ((t / 2) + 1) % 2, which matches tile (t - 2)'s completion. *)
          let wait_epi =
            Printf.sprintf
              "    if (%s >= 2) {\n\
              \      const uint32_t _em = (uint32_t)__cvta_generic_to_shared(__smem_base + offsetof(SharedStorage, epilogue_done_mbar) + (%s %% 2) * sizeof(uint64_t));\n\
              \      asm volatile(\"{\\n\\t\" \".reg .pred p;\\n\\t\" \"LOOP_EPI:\\n\\t\" \"mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%%0], %%1, 0x989680;\\n\\t\" \"@!p bra LOOP_EPI;\\n\\t\" \"}\" :: \"r\"(_em), \"r\"((uint32_t)(((%s / 2) + 1) %% 2)) : \"memory\");\n\
              \    }\n"
              var.var_name var.var_name var.var_name
          in
          Printf.sprintf "%sfor (%s %s = %s; %s < %s; %s += %s) {\n%s%s%s\n%s%s\n%s}"
            ind
            (emit_packed_scalar var.var_type) var.var_name (emit_expr start)
            var.var_name (emit_expr stop) var.var_name (emit_expr step)
            wait_epi "    _mma_acc = 0u;" loop_s ind (emit_mma_done_signal ()) ind
        | _ ->
          body_s ^ "\n" ^ emit_mma_done_signal ()
      else body_s in
    let body_s =
      if debug_pipeline_stats then
        (match role with
         | Cluster.Producer -> "    const long long _grp_t0 = clock64();\n" ^ body_s ^ "\n" ^ stats_print_group "producer"
         | Cluster.Consumer -> "    const long long _grp_t0 = clock64();\n" ^ body_s ^ "\n" ^ stats_print_group "consumer"
         | _ -> body_s)
      else body_s in
    Printf.sprintf "if (%s) {\n%s%s\n}" cond prefix body_s)
  in
  let epilogue_case =
    if not (List.is_empty epi_stores) then
      let ids = epilogue_warp_ids k in
      let cond = String.concat ~sep:" || " (List.map ids ~f:(Printf.sprintf "warp_id == %d")) in
      [ Printf.sprintf "if (%s) {\n%s\n}" cond (emit_stmts k ~depth:2 ?stage_depth:sd epi_stores) ]
    else [] in
  let warp_cases = warp_cases @ epilogue_case in
  let warp_dispatch =
    if List.is_empty warp_cases then "" else String.concat ~sep:" else " warp_cases in
  let tiled_mma_s = emit_tiled_mma_decl k in
  let stage_decl =
    (if debug_pipeline_stats
     then "long long _wait_full_mbar = 0, _wait_empty_mbar = 0;\n  (void)_wait_full_mbar; (void)_wait_empty_mbar;\n"
     else "")
    ^ (if tirix_is_tma k then "int stage = 0;\n  (void)stage;\n" else "")
    ^ (if Poly.(k.family = Kernel_desc.Blackwell) then "uint32_t _mma_acc = 0u;\n  (void)_mma_acc;\n" else "") in
  let mbar_init_s = emit_mbar_init_loop k in
  let tmem_dealloc_s =
    if Poly.(k.family = Kernel_desc.Blackwell) && not (has_tmem_alloc k.body) then
      Printf.sprintf
        "  __syncthreads();\n\
        \  if (warp_id == 0) {\n\
        \    asm volatile(\"tcgen05.dealloc.cta_group::2.sync.aligned.b32 %%0, %d;\"\n\
        \      :: \"r\"(smem.tmem_addr[0]) : \"memory\");\n\
        \    asm volatile(\"tcgen05.dealloc.cta_group::2.sync.aligned.b32 %%0, %d;\"\n\
        \      :: \"r\"(smem.tmem_addr[1]) : \"memory\");\n\
        \  }\n"
        k.bn k.bn
    else "" in
  Printf.sprintf
    "extern \"C\" __global__ %s__launch_bounds__(%d)\nvoid %s(\n  %s\n) {\n\
    \  extern __shared__ __align__(1024) char smem_buf[];\n\
    \  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);\n\
    \  char* __smem_base = smem_buf;\n\
    \  uint32_t __smem_base_u32 = (uint32_t)__cvta_generic_to_shared(smem_buf);\n\
    \  (void)smem;\n\
    \  const int warp_id = threadIdx.x / 32;\n\
    \  const int lane_id = threadIdx.x %% 32;\n\
    \  (void)lane_id;\n\
     %s%s\n%s\n%s\n%s\n%s}\n"
    cluster_attr n_threads k.name (emit_params k)
    tiled_mma_s stage_decl mbar_init_s pre_s warp_dispatch tmem_dealloc_s


let emit_host_launcher_params (k : tirix) : string =
  String.concat ~sep:",\n  "
    (List.map k.params ~f:(fun p ->
      let (Tensor t) = p.param_tensor in
      let elem_t = Elemtype.cpp_name t.tensor_elem_type in
      match p.param_name with
      | "M" | "N" | "K" -> Printf.sprintf "int %s" p.param_name
      | _ ->
        if p.param_is_tma then Printf.sprintf "const CUtensorMap* %s_tmap" p.param_name
        else Printf.sprintf "const %s* %s" elem_t p.param_name))

let emit_host_launcher (k : tirix) : string =
  let is_blackwell = match k.family with Kernel_desc.Blackwell -> true | _ -> false in
  let kernel_args = String.concat ~sep:", "
    (List.map k.params ~f:(fun p ->
      if p.param_is_tma then Printf.sprintf "%s_tmap" p.param_name
      else p.param_name)) in
  let has_m = List.exists k.params ~f:(fun p -> String.equal p.param_name "M") in
  let has_n = List.exists k.params ~f:(fun p -> String.equal p.param_name "N") in
  let grid_x = if has_m then Printf.sprintf "((M + %d - 1) / %d) / 2" k.bm k.bm else "1" in
  let grid_y = if has_n then Printf.sprintf "(N + %d - 1) / %d" k.bn k.bn else "1" in
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
        k.smem_bytes k.cluster.Cluster.dims.Cluster.x k.cluster.Cluster.dims.Cluster.y
        k.cluster.Cluster.dims.Cluster.z k.name kernel_args
    else
      Printf.sprintf "#ifndef __CUDACC_RTC__\n  %s<<<grid, block, %d>>>(%s);\n#endif"
        k.name k.smem_bytes kernel_args
  in
  Printf.sprintf
    "#ifndef __CUDACC_RTC__\nvoid launch_%s(\n  %s\n) {\n  dim3 grid(%s, %s, 1);\n  dim3 block(%d, 1, 1);\n  (void)block;\n %s\n }\n#endif\n"
    k.name (emit_host_launcher_params k) grid_x grid_y
    (Cluster.thread_count k.cluster) launch


let emit_includes (k : tirix) : string =
  let arch_inc = match k.family with
    | Kernel_desc.Ampere -> "#include <cuda_fp16.h>\n"
    | Kernel_desc.Hopper -> "#include <cuda_bf16.h>\n"
    | Kernel_desc.Blackwell -> "#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n" in
  String.concat ~sep:"\n" [
    "#pragma once";
    "#include <cute/tensor.hpp>";
    "#include <cute/atom/mma_atom.hpp>";
    "#include <cute/atom/copy_atom.hpp>";
    "#include <cute/algorithm/gemm.hpp>";
    "#include <cute/algorithm/copy.hpp>";
    "#include <cooperative_groups.h>";
    "using namespace cute;";
    arch_inc;
    "struct CUtensorMap { alignas(64) uint64_t opaque[16]; };";
    "";
    "__device__ inline uint64_t make_smem_desc_raw(";
    "    void* smem_ptr, uint32_t lead, uint32_t stride, uint32_t sw) {";
    "  uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);";
    "  return ((uint64_t)sw << 62) | ((uint64_t)stride << 36) | ((uint64_t)lead << 16) | (addr >> 4);";
    "}";
    "__device__ inline uint64_t make_smem_desc(const void* smem_ptr) {";
    "  uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);";
    "  return (uint64_t)(addr >> 4);";
    "}";
    "__device__ inline uint64_t make_smem_desc(";
    "    const void* smem_ptr, uint32_t lead, uint32_t stride, uint32_t sw) {";
    "  return make_smem_desc_raw(const_cast<void*>(smem_ptr), lead, stride, sw);";
    "}";
    "__device__ __forceinline__ uint64_t make_wgmma_desc(";
    "    uint32_t smem_base, uint32_t smem_offset,";
    "    uint32_t lbo, uint32_t sbo, uint32_t swizzle) {";
    "  uint32_t addr = smem_base + smem_offset;";
    "  uint64_t desc = (uint64_t)(addr >> 4);";
    "  desc |= ((uint64_t)(lbo >> 4)) << 16;";
    "  desc |= ((uint64_t)(sbo >> 4)) << 32;";
    "  desc |= ((uint64_t)swizzle)    << 62;";
    "  return desc;";
    "}";
    "__device__ __forceinline__ uint64_t make_sm100_smem_desc(const void* smem_ptr) {";
    "  uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);";
    "  uint64_t desc = 0;";
    "  desc |= (uint64_t)((addr & 0x3FFFFu) >> 4);";
    "  desc |= (uint64_t)1 << 16;";
    "  desc |= (uint64_t)(1024 >> 4) << 32;";
    "  desc |= (uint64_t)1 << 46;";
    "  desc |= (uint64_t)2 << 61;";
    "  return desc;";
    "}";
    "__device__ inline void tma_2d_gmem2smem(";
    "    void* smem, const CUtensorMap* tmap, int x, int y, uint64_t* mbar) {";
    "  if (threadIdx.x == 0) {";
    "    asm volatile(";
    "      \"cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes \"";
    "      \"[%0], [%1, {%2, %3}], [%4];\"";
    "      :: \"r\"((uint32_t)__cvta_generic_to_shared(smem)), \"l\"(tmap),";
    "      \"r\"(x), \"r\"(y), \"r\"((uint32_t)__cvta_generic_to_shared(mbar))";
    "      : \"memory\");";
    "  }";
    "}";
    "__device__ inline void tma_2d_gmem2smem_multicast(";
    "    void* smem, const CUtensorMap* tmap, int x, int y,";
    "    uint64_t* mbar, uint16_t cta_mask) {";
    "  if (threadIdx.x == 0) {";
    "    asm volatile(";
    "      \"cp.async.bulk.tensor.2d.shared::cluster.global\"";
    "      \".mbarrier::complete_tx::bytes.multicast::cluster\"";
    "      \" [%0], [%1, {%2, %3}], [%4], %5;\"";
    "      :: \"r\"((uint32_t)__cvta_generic_to_shared(smem)), \"l\"(tmap),";
    "      \"r\"(x), \"r\"(y), \"r\"((uint32_t)__cvta_generic_to_shared(mbar)), \"h\"(cta_mask)";
    "      : \"memory\");";
    "  }";
    "}";
    "__device__ inline uint32_t mapa_to_cta0(uint32_t a) {";
    "  uint32_t r;";
    "  asm volatile(\"mapa.shared::cluster.u32 %0, %1, 0;\" : \"=r\"(r) : \"r\"(a));";
    "  return r;";
    "}";
    "__device__ inline void tma_2d_gmem2smem_cg2(";
    "    void* smem, const CUtensorMap* tmap, int x, int y, uint64_t* mbar) {";
    "  if (threadIdx.x == 0) {";
    "    uint32_t mb = mapa_to_cta0((uint32_t)__cvta_generic_to_shared(mbar));";
    "    asm volatile(";
    "      \"cp.async.bulk.tensor.2d.shared::cluster.global\"";
    "      \".mbarrier::complete_tx::bytes.cta_group::2\"";
    "      \" [%0], [%1, {%2, %3}], [%4];\"";
    "      :: \"r\"((uint32_t)__cvta_generic_to_shared(smem)), \"l\"(tmap),";
    "      \"r\"(x), \"r\"(y), \"r\"(mb)";
    "      : \"memory\");";
    "  }";
    "}";
  ]


let emit (k : tirix) : Backend_cute.output =
  let includes = emit_includes k in
  let helpers = String.concat ~sep:"\n\n" (List.map k.helpers ~f:(emit_helper k)) in
  let shared_storage = emit_shared_storage k in
  let kernel_func = emit_kernel_func k in
  let host_launcher  = emit_host_launcher k in
  let kernel_source = String.concat ~sep:"\n\n" [ includes; helpers; shared_storage; kernel_func ] in
  let full_source = String.concat ~sep:"\n\n" [ kernel_source; host_launcher ] in
  Stdlib.Printf.printf "=== Generated kernel: %s ===\n%s\n=== End kernel ===\n%!" k.name full_source;
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
