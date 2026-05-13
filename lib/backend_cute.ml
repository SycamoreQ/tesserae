open Base
open Stdio

type output = {
  filename  : string;
  includes  : string;
  helpers  : string;
  shared_storage : string;
  producer_body : string;
  consumer_body : string;
  epilogue_body : string;
  kernel_func : string;
  host_launcher : string;
  full_source : string;
}

let elem_t (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  Elemtype.cpp_name
    desc.Kernel_desc.tile_io
      .Tile_io.tiled_copy_a
      .Tiled_copy.atom
      .Copy_atom.elem_type

let depth (desc : (_, _, _, _, _, _) Kernel_desc.t) : int =
  desc.Kernel_desc.pipeline.Pipeline.depth

let is_tma (desc : (_, _, _, _, _, _) Kernel_desc.t) : bool =
  Tile_io.is_tma desc.Kernel_desc.tile_io

let is_blackwell (desc : (_, _, _, _, _, _) Kernel_desc.t) : bool =
  match desc.Kernel_desc.family with
  | Kernel_desc.Blackwell -> true
  | _ -> false

let bn_smem (desc : (_, _, _, _, _, _) Kernel_desc.t) : int =
  match desc.Kernel_desc.family with
  | Kernel_desc.Blackwell -> desc.Kernel_desc.bn / 2
  | _ -> desc.Kernel_desc.bn

let emit_includes (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let arch_includes = match desc.Kernel_desc.family with
    | Kernel_desc.Ampere    -> ""
    | Kernel_desc.Hopper    -> "#include <cuda_bf16.h>\n"
    | Kernel_desc.Blackwell -> "#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n"
  in
  Printf.sprintf "%s\n%s\n%s"
    (Codegen.emit_include_guard ())
    (Codegen.emit_cute_includes ())
    arch_includes

let emit_helpers (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  if not (is_tma desc) then ""
  else
    let smem_desc_helper = Smem_desc.emit_cpp_helper () in
    let tma_load = String.concat ~sep:"\n" [
      "__device__ __forceinline__ void tma_2d_gmem2smem(";
      "    void* smem, CUtensorMap* tmap, int x, int y, uint64_t* mbar) {";
      "  asm volatile (";
      "    \"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes\"";
      "    \" [%0], [%1, {%2, %3}], [%4];\"";
      "    :: \"r\"(smem), \"l\"(tmap), \"r\"(x), \"r\"(y), \"r\"(mbar)";
      "    : \"memory\");";
      "}";
    ] in
    let tma_multicast = match desc.Kernel_desc.family with
      | Kernel_desc.Blackwell -> String.concat ~sep:"\n" [
          "__device__ __forceinline__ void tma_2d_gmem2smem_multicast(";
          "    void* smem, CUtensorMap* tmap, int x, int y,";
          "    uint64_t* mbar, uint16_t cta_mask) {";
          "  asm volatile (";
          "    \"cp.async.bulk.tensor.2d.shared::cluster.global\"";
          "    \".mbarrier::complete_tx::bytes.multicast::cluster\"";
          "    \" [%0], [%1, {%2, %3}], [%4], %5;\"";
          "    :: \"r\"(smem), \"l\"(tmap), \"r\"(x), \"r\"(y),";
          "       \"r\"(mbar), \"h\"(cta_mask)";
          "    : \"memory\");";
          "}";
        ]
      | _ -> ""
    in
    String.concat ~sep:"\n\n" [smem_desc_helper; tma_load; tma_multicast]

(* __shared__ must NOT appear inside a struct — only on the instance *)
let emit_shared_storage (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let et   = elem_t desc in
  let bm   = desc.Kernel_desc.bm in
  let bk   = desc.Kernel_desc.bk in
  let bn_s = bn_smem desc in
  let d    = depth desc in
  let mbar_decls = if not (is_tma desc) then ""
    else Printf.sprintf "  uint64_t full_mbar[%d];\n  uint64_t empty_mbar[%d];\n" d d
  in
  let tmem_decl = if is_blackwell desc
    then "  uint32_t tmem_addr[1];\n"
    else ""
  in
  Printf.sprintf
    "struct SharedStorage {\n\
    \  %s smem_A[%d][%d];\n\
    \  %s smem_B[%d][%d];\n\
    %s%s};"
    et d (bm * bk)
    et d (bn_s * bk)
    mbar_decls
    tmem_decl

let emit_producer_body (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let d    = depth desc in
  let bm   = desc.Kernel_desc.bm in
  let bk   = desc.Kernel_desc.bk in
  let bn_s = bn_smem desc in
  let bw   = Elemtype.byte_width
    desc.Kernel_desc.tile_io
      .Tile_io.tiled_copy_a
      .Tiled_copy.atom
      .Copy_atom.elem_type
  in
  let load_body = match desc.Kernel_desc.family with
    | Kernel_desc.Ampere ->
      (* use __cvta_generic_to_shared to get 32-bit smem addr for "r" constraint *)
      Printf.sprintf
        "    // cp.async load A\n\
        \    unsigned smem_a = __cvta_generic_to_shared(smem.smem_A[stage]);\n\
        \    asm volatile(\"cp.async.ca.shared.global [%%0], [%%1], 16;\"\n\
        \      :: \"r\"(smem_a), \"l\"(A + row * K + k * %d) : \"memory\");\n\
        \    // cp.async load B\n\
        \    unsigned smem_b = __cvta_generic_to_shared(smem.smem_B[stage]);\n\
        \    asm volatile(\"cp.async.ca.shared.global [%%0], [%%1], 16;\"\n\
        \      :: \"r\"(smem_b), \"l\"(B + k * %d + col) : \"memory\");\n\
        \    asm volatile(\"cp.async.commit_group;\");"
        bk bn_s
    | Kernel_desc.Hopper ->
      Printf.sprintf
        "    // TMA expect_tx\n\
        \    asm volatile(\"mbarrier.expect_tx.shared.b64 [%%0], %%1;\" ::\n\
        \      \"r\"(&smem.full_mbar[stage]), \"r\"(%d));\n\
        \    // TMA load A\n\
        \    tma_2d_gmem2smem(smem.smem_A[stage], &A_tmap, k, row, &smem.full_mbar[stage]);\n\
        \    // TMA load B\n\
        \    tma_2d_gmem2smem(smem.smem_B[stage], &B_tmap, col, k, &smem.full_mbar[stage]);"
        ((bm + bn_s) * bk * bw)
    | Kernel_desc.Blackwell ->
      Printf.sprintf
        "    // TMA multicast expect_tx\n\
        \    asm volatile(\"mbarrier.expect_tx.shared.b64 [%%0], %%1;\" ::\n\
        \      \"r\"(&smem.full_mbar[stage]), \"r\"(%d));\n\
        \    // TMA multicast load A\n\
        \    tma_2d_gmem2smem_multicast(smem.smem_A[stage], &A_tmap, k, row,\n\
        \      &smem.full_mbar[stage], 0b11);\n\
        \    // TMA multicast load B (each CTA loads half)\n\
        \    tma_2d_gmem2smem_multicast(smem.smem_B[stage], &B_tmap,\n\
        \      col + cta_rank * (BN / 2), k, &smem.full_mbar[stage], 0b11);"
        ((bm + bn_s) * bk * bw)
  in
  Printf.sprintf
    "// Producer warp body\n\
     for (int k = 0; k < K / %d; k++) {\n\
    \  int stage = k %% %d;\n\
     %s\n\
     }\n\
    \  asm volatile(\"cp.async.wait_group 0;\");"
    bk d load_body

let emit_consumer_body (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let d  = depth desc in
  let bk = desc.Kernel_desc.bk in
  let mma_instr = match desc.Kernel_desc.family with
    | Kernel_desc.Ampere ->
      "    // mma.sync via CuTe\n\
      \    cute::gemm(tiled_mma, acc, sA, sB, acc);"
    | Kernel_desc.Hopper ->
      "    // wgmma\n\
      \    asm volatile(\"wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 ...;\");"
    | Kernel_desc.Blackwell ->
      "    // tcgen05.mma\n\
      \    asm volatile(\"tcgen05.mma.cta_group::1.kind::mxf16 [%0], %1, %2, 1;\"\n\
      \      :: \"r\"(tmem_addr), \"r\"(make_smem_desc(smem.smem_A[stage])),\n\
      \         \"r\"(make_smem_desc(smem.smem_B[stage])));"
  in
  let commit = if is_blackwell desc
    then "\n  // tcgen05 commit\n  " ^
      (Tmem.commit_ptx
        (Tmem.make ~cta_group:Tmem.CTA1
          ~num_cols:desc.Kernel_desc.bn
          ~num_rows:desc.Kernel_desc.bm)
        "smem.full_mbar[0]")
    else ""
  in
  Printf.sprintf
    "// Consumer warp body\n\
     for (int k = 0; k < K / %d; k++) {\n\
    \  int stage = k %% %d;\n\
    \  int phase = (k / %d) %% 2;\n\
    \  // wait for data\n\
    \  asm volatile(\"mbarrier.wait.parity.shared.b64 [%%0], %%1;\" ::\n\
    \    \"r\"(&smem.full_mbar[stage]), \"r\"(phase));\n\
    \  auto sA = make_tensor(make_smem_ptr(smem.smem_A[stage]),\n\
    \    Layout<Shape<Int<%d>, Int<%d>>, Stride<Int<1>, Int<%d>>>{});\n\
    \  auto sB = make_tensor(make_smem_ptr(smem.smem_B[stage]),\n\
    \    Layout<Shape<Int<%d>, Int<%d>>, Stride<Int<1>, Int<%d>>>{});\n\
     %s\n\
    \  // signal empty\n\
    \  asm volatile(\"mbarrier.arrive.shared.b64 [%%0];\" ::\n\
    \    \"r\"(&smem.empty_mbar[stage]));\n\
     }\n%s"
    bk d d
    desc.Kernel_desc.bm desc.Kernel_desc.bk desc.Kernel_desc.bm
    (bn_smem desc) desc.Kernel_desc.bk (bn_smem desc)
    mma_instr
    commit

let emit_epilogue_body (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  match desc.Kernel_desc.family with
  | Kernel_desc.Blackwell ->
    Printf.sprintf
      "// Epilogue — tcgen05.ld + store\n\
       uint32_t taddr = *smem.tmem_addr;\n\
       for (int i = 0; i < %d; i++) {\n\
      \  float regs[8];\n\
      \  asm volatile(\n\
      \    \"tcgen05.ld.sync.aligned.32x32b.x8.b32 {%%0,%%1,%%2,%%3,%%4,%%5,%%6,%%7}, [%%8];\"\n\
      \    : \"=r\"(regs[0]),\"=r\"(regs[1]),\"=r\"(regs[2]),\"=r\"(regs[3]),\n\
      \      \"=r\"(regs[4]),\"=r\"(regs[5]),\"=r\"(regs[6]),\"=r\"(regs[7])\n\
      \    : \"r\"(taddr + i * 8));\n\
      \  int row = block_m * %d + (threadIdx.x / 32) * 8 + i;\n\
      \  int col = block_n * %d + (threadIdx.x %% 32);\n\
      \  if (row < M && col < N) C[row * N + col] = regs[0];\n\
       }"
      (desc.Kernel_desc.bn / 8)
      desc.Kernel_desc.bm
      desc.Kernel_desc.bn
  | _ ->
    (* standard register accumulator epilogue via CuTe *)
    Printf.sprintf
      "// Epilogue — store accumulators to global\n\
      \  auto C_gmem = make_tensor(\n\
      \    make_gmem_ptr(C + block_m * %d * N + block_n * %d),\n\
      \    make_layout(\n\
      \      make_shape(Int<%d>{}, Int<%d>{}),\n\
      \      make_stride(N, Int<1>{})));\n\
      \  auto thr_mma = tiled_mma.get_slice(threadIdx.x);\n\
      \  auto C_frag  = thr_mma.partition_C(C_gmem);\n\
      \  CUTE_UNROLL\n\
      \  for (int i = 0; i < size(C_frag); i++) {\n\
      \    int row = block_m * %d + get<0>(thr_mma.get_thread_slice_coord(i));\n\
      \    int col = block_n * %d + get<1>(thr_mma.get_thread_slice_coord(i));\n\
      \    if (row < M && col < N) C_frag(i) = acc(i);\n\
      \  }"
      desc.Kernel_desc.bm desc.Kernel_desc.bn
      desc.Kernel_desc.bm desc.Kernel_desc.bn
      desc.Kernel_desc.bm desc.Kernel_desc.bn

let emit_kernel_func (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let params   = Kernel_desc.emit_kernel_params desc in
  let n_warps  = desc.Kernel_desc.cluster.Cluster.num_warps in
  let smem_sz  = desc.Kernel_desc.pipeline.Pipeline.smem_bytes in
  let producer = Option.value ~default:0
    (Cluster.producer_warp desc.Kernel_desc.cluster) in
  let consumer = Option.value ~default:1
    (Cluster.consumer_warp desc.Kernel_desc.cluster) in
  let tmem_alloc = if is_blackwell desc
    then "\n  // TMEM alloc\n  " ^
      (Tmem.alloc_ptx
        (Tmem.make ~cta_group:Tmem.CTA1
          ~num_cols:desc.Kernel_desc.bn
          ~num_rows:desc.Kernel_desc.bm)
        "smem.tmem_addr[0]") ^ "\n"
    else ""
  in
  let tmem_dealloc = if is_blackwell desc
    then "\n  // TMEM dealloc\n  " ^
      (Tmem.dealloc_ptx
        (Tmem.make ~cta_group:Tmem.CTA1
          ~num_cols:desc.Kernel_desc.bn
          ~num_rows:desc.Kernel_desc.bm)
        "*smem.tmem_addr") ^ "\n"
    else ""
  in
  let cluster_attr = if is_blackwell desc
    then Printf.sprintf "__attribute__((%s))\n"
      (Cluster.emit_cluster_attr desc.Kernel_desc.cluster)
    else ""
  in
  (* accumulator and tiled_mma declarations — only for non-Blackwell *)
  let acc_decl = match desc.Kernel_desc.family with
    | Kernel_desc.Blackwell -> ""
    | _ ->
      Printf.sprintf
        "  // tiled MMA and accumulator\n\
        \  using TiledMMA = decltype(make_tiled_mma(\n\
        \    MMA_Atom<SM80_16x8x16_F32F16F16F32>{},\n\
        \    Layout<Shape<_2,_2,_1>>{},\n\
        \    Tile<Int<%d>,Int<%d>,_16>{}));\n\
        \  TiledMMA tiled_mma;\n\
        \  auto thr_mma = tiled_mma.get_slice(threadIdx.x);\n\
        \  auto acc = partition_fragment_C(\n\
        \    thr_mma,\n\
        \    make_layout(make_shape(Int<%d>{}, Int<%d>{})));\n\
        \  clear(acc);\n"
        desc.Kernel_desc.bm desc.Kernel_desc.bn
        desc.Kernel_desc.bm desc.Kernel_desc.bn
  in
  (* mbarrier init — uses asm volatile *)
  let mbar_init = if not (is_tma desc) then ""
    else Printf.sprintf
      "  // init mbarriers\n\
      \  if (warp_id == 0 && lane_id == 0) {\n\
      \    for (int i = 0; i < %d; i++) {\n\
      \      asm volatile(\"mbarrier.init.shared.b64 [%%0], 1;\" :: \"r\"(&smem.full_mbar[i]));\n\
      \      asm volatile(\"mbarrier.init.shared.b64 [%%0], 1;\" :: \"r\"(&smem.empty_mbar[i]));\n\
      \    }\n\
      \  }\n\
      \  __syncthreads();\n"
      (depth desc)
  in
  (* for Ampere cp.async, simpler sync barrier *)
  let cp_async_barrier = if is_tma desc then ""
    else
      "  // wait for all cp.async to complete\n\
      \  asm volatile(\"cp.async.wait_all;\");\n\
      \  __syncthreads();\n"
  in
  ignore smem_sz;
  Printf.sprintf
    "%s__global__ __launch_bounds__(%d)\n\
     void %s(%s) {\n\
    \  extern __shared__ char smem_buf[];\n\
    \  SharedStorage& smem =\n\
    \    *reinterpret_cast<SharedStorage*>(smem_buf);\n\
    \  int warp_id = threadIdx.x / 32;\n\
    \  int lane_id = threadIdx.x %% 32;\n\
    \  (void)lane_id;\n\
    \  // grid coords\n\
    \  int block_m = blockIdx.x;\n\
    \  int block_n = blockIdx.y;\n\
    \  int row = block_m * %d;\n\
    \  int col = block_n * %d;\n\
     %s%s%s\
    \  // warp dispatch\n\
    \  if (warp_id == %d) {\n\
    \    %s\n\
    \  } else if (warp_id == %d) {\n\
    \    %s\n\
    \  } else {\n\
    \    %s\n\
    \  }\n\
     %s}\n"
    cluster_attr
    (n_warps * 32)
    desc.Kernel_desc.name
    params
    desc.Kernel_desc.bm
    desc.Kernel_desc.bn
    tmem_alloc
    mbar_init
    (acc_decl ^ cp_async_barrier)
    producer
    (emit_producer_body desc)
    consumer
    (emit_consumer_body desc)
    (emit_epilogue_body desc)
    tmem_dealloc

let emit_host_launcher (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let et      = elem_t desc in
  let n_warps = desc.Kernel_desc.cluster.Cluster.num_warps in
  let smem_sz = desc.Kernel_desc.pipeline.Pipeline.smem_bytes in
  let bm      = desc.Kernel_desc.bm in
  let bn      = desc.Kernel_desc.bn in
  let tmap_setup = if is_tma desc then
    Printf.sprintf
      "  // TMA descriptor setup\n\
      \  CUtensorMap A_tmap, B_tmap;\n\
      \  cuTensorMapEncode2D(&A_tmap, %s_type, A_ptr, M, K, K, %d, %d);\n\
      \  cuTensorMapEncode2D(&B_tmap, %s_type, B_ptr, K, N, N, %d, %d);\n"
      et bm desc.Kernel_desc.bk et bn desc.Kernel_desc.bk
    else ""
  in
  let cluster_launch = match desc.Kernel_desc.family with
    | Kernel_desc.Blackwell ->
      Printf.sprintf
        "  // Blackwell cluster launch\n\
        \  cudaLaunchConfig_t cfg = {};\n\
        \  cfg.gridDim  = grid;\n\
        \  cfg.blockDim = block;\n\
        \  cfg.dynamicSmemBytes = %d;\n\
        \  cudaLaunchAttribute attrs[1];\n\
        \  attrs[0].id = cudaLaunchAttributeClusterDimension;\n\
        \  attrs[0].val.clusterDim = {%d, %d, %d};\n\
        \  cfg.attrs    = attrs;\n\
        \  cfg.numAttrs = 1;\n\
        \  cudaLaunchKernelEx(&cfg, %s, A_tmap, B_tmap, C_ptr, M, N, K);"
        smem_sz
        desc.Kernel_desc.cluster.Cluster.dims.Cluster.x
        desc.Kernel_desc.cluster.Cluster.dims.Cluster.y
        desc.Kernel_desc.cluster.Cluster.dims.Cluster.z
        desc.Kernel_desc.name
    | _ ->
      Printf.sprintf
        "  cudaFuncSetAttribute(%s,\n\
        \    cudaFuncAttributeMaxDynamicSharedMemorySize, %d);\n\
        \  %s<<<grid, block, %d>>>(%s);"
        desc.Kernel_desc.name
        smem_sz
        desc.Kernel_desc.name
        smem_sz
        (if is_tma desc
         then "A_tmap, B_tmap, C_ptr, M, N, K"
         else "A_ptr, B_ptr, C_ptr, M, N, K")
  in
  Printf.sprintf
    "void launch_%s(\n\
    \  const %s* A_ptr, const %s* B_ptr, float* C_ptr,\n\
    \  int M, int N, int K)\n\
     {\n\
    \  dim3 grid((M + %d - 1) / %d, (N + %d - 1) / %d, 1);\n\
    \  dim3 block(%d, 1, 1);\n\
     %s%s\n\
     }\n"
    desc.Kernel_desc.name
    et et
    bm bm bn bn
    (n_warps * 32)
    tmap_setup
    cluster_launch

let emit (desc : (_, _, _, _, _, _) Kernel_desc.t) : output =
  let includes       = emit_includes       desc in
  let helpers        = emit_helpers        desc in
  let shared_storage = emit_shared_storage desc in
  let producer_body  = emit_producer_body  desc in
  let consumer_body  = emit_consumer_body  desc in
  let epilogue_body  = emit_epilogue_body  desc in
  let kernel_func    = emit_kernel_func    desc in
  let host_launcher  = emit_host_launcher  desc in
  let full_source    = String.concat ~sep:"\n\n"
    [ includes; helpers; shared_storage
    ; kernel_func; host_launcher ] in
  { filename       = desc.Kernel_desc.name ^ ".cuh"
  ; includes
  ; helpers
  ; shared_storage
  ; producer_body
  ; consumer_body
  ; epilogue_body
  ; kernel_func
  ; host_launcher
  ; full_source
  }

let write (desc : (_, _, _, _, _, _) Kernel_desc.t) (path : string) : unit =
  let out = emit desc in
  Out_channel.write_all path ~data:out.full_source
