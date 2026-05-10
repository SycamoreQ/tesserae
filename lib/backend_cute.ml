open Base
open Stdio

type output = {
  filename : string;
  includes : string;
  helpers : string;
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


let emit_shared_storage (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let et    = elem_t desc in
  let bm    = desc.Kernel_desc.bm in
  let bk    = desc.Kernel_desc.bk in
  let bn_s  = bn_smem desc in
  let d     = depth desc in
  let mbar_decls = if not (is_tma desc) then ""
    else Printf.sprintf "  %s\n  %s\n"
      (Cluster.emit_smem_mbar "full_mbar"  d)
      (Cluster.emit_smem_mbar "empty_mbar" d)
  in
  let tmem_decl = if is_blackwell desc
    then "  __shared__ uint32_t tmem_addr[1];\n"
    else ""
  in
  Printf.sprintf
    "struct SharedStorage {\n\
    \  __shared__ %s smem_A[%d][%d];\n\
    \  __shared__ %s smem_B[%d][%d];\n\
    %s%s};"
    et d (bm * bk)
    et d (bn_s * bk)
    mbar_decls
    tmem_decl

let emit_producer_body (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let d  = depth desc in
  let bm = desc.Kernel_desc.bm in
  let bk = desc.Kernel_desc.bk in
  let bn_s = bn_smem desc in
  let bw = Elemtype.byte_width
    desc.Kernel_desc.tile_io
      .Tile_io.tiled_copy_a
      .Tiled_copy.atom
      .Copy_atom.elem_type
  in
  let load_body = match desc.Kernel_desc.family with
    | Kernel_desc.Ampere ->
      Printf.sprintf
        "    // cp.async load A\n\
        \    asm volatile(\"cp.async.ca.shared.global [%%0], [%%1], 16;\"\n\
        \      :: \"r\"(smem.smem_A[stage]), \"l\"(A + row * K + k * %d) : \"memory\");\n\
        \    // cp.async load B\n\
        \    asm volatile(\"cp.async.ca.shared.global [%%0], [%%1], 16;\"\n\
        \      :: \"r\"(smem.smem_B[stage]), \"l\"(B + k * %d + col) : \"memory\");\n\
        \    asm volatile(\"cp.async.commit_group;\");"
        bk bn_s
    | Kernel_desc.Hopper ->
      Printf.sprintf
        "    // TMA expect_tx\n\
        \    mbarrier.expect_tx.shared.b64 [full_mbar[stage]], %d;\n\
        \    // TMA load A\n\
        \    tma_2d_gmem2smem(smem.smem_A[stage], &A_tmap, k, row, &smem.full_mbar[stage]);\n\
        \    // TMA load B\n\
        \    tma_2d_gmem2smem(smem.smem_B[stage], &B_tmap, col, k, &smem.full_mbar[stage]);"
        ((bm + bn_s) * bk * bw)
    | Kernel_desc.Blackwell ->
      Printf.sprintf
        "    // TMA multicast expect_tx\n\
        \    mbarrier.expect_tx.shared.b64 [full_mbar[stage]], %d;\n\
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
    \  // signal done\n\
    \  asm volatile(\"cp.async.wait_group 0;\");"
    bk d load_body


let emit_consumer_body (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let d  = depth desc in
  let bk = desc.Kernel_desc.bk in
  let mma_instr = match desc.Kernel_desc.family with
    | Kernel_desc.Ampere ->
      Printf.sprintf "    // mma.sync\n\
      \    cute::gemm(tiled_mma, acc, smem.smem_A[stage], smem.smem_B[stage], acc);"
    | Kernel_desc.Hopper ->
      Printf.sprintf "    // wgmma\n\
      \    wgmma.mma_async.sync.aligned smem.smem_A[stage], smem.smem_B[stage];"
    | Kernel_desc.Blackwell ->
      Printf.sprintf "    // tcgen05.mma\n\
      \    tcgen05.mma.cta_group::1.kind::mxf16 [tmem_addr],\n\
      \      make_smem_desc(smem.smem_A[stage]),\n\
      \      make_smem_desc(smem.smem_B[stage]), 1;"
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
    \  // wait full mbar\n\
    \  mbarrier.wait.parity.shared.b64 [full_mbar[stage]], phase;\n\
     %s\n\
    \  // arrive empty mbar\n\
    \  mbarrier.arrive.shared.b64 [empty_mbar[stage]];\n\
     }\n%s"
    bk d mma_instr commit


let emit_epilogue_body (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let _et = elem_t desc in
  match desc.Kernel_desc.family with
  | Kernel_desc.Blackwell ->
    Printf.sprintf
      "// Epilogue — tcgen05.ld + store\n\
       uint32_t taddr = *smem.tmem_addr;\n\
       for (int i = 0; i < %d; i++) {\n\
      \  float regs[8];\n\
      \  // tcgen05.ld\n\
      \  tcgen05.ld.sync.aligned.32x32b.x8.b32\n\
      \    {regs[0],regs[1],regs[2],regs[3],regs[4],regs[5],regs[6],regs[7]},\n\
      \    [taddr + i * 8];\n\
      \  // store to global via predicate\n\
      \  if (predicate) store(C + offset + i, regs);\n\
       }"
      (desc.Kernel_desc.bn / 8)
  | _ ->
    Printf.sprintf
      "// Epilogue — predicated store of register accumulators\n\
       // predicate: check C tile bounds\n\
       auto predicate = %s;\n\
       if (predicate) {\n\
      \  cute::copy(acc, smem_C);\n\
      \  __syncthreads();\n\
      \  cute::copy(smem_C, C_tile);\n\
       }\n\
       // store result\n\
       cute::axpby(1.0f, acc, 0.0f, C_tile);"
      (Predicate.emit_predicate_check
        desc.Kernel_desc.pred_c "coord")

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
     %s\
    \  // init mbarriers\n\
    \  if (warp_id == 0 && lane_id == 0) {\n\
    \    for (int i = 0; i < %d; i++) {\n\
    \      mbarrier.init.shared.b64 [smem.full_mbar[i]], 1;\n\
    \      mbarrier.init.shared.b64 [smem.empty_mbar[i]], 1;\n\
    \    }\n\
    \  }\n\
    \  __syncthreads();\n\
    \  // warp dispatch\n\
    \  if (warp_id == %d) {\n\
    \    // producer\n\
    \    %s\n\
    \  } else if (warp_id == %d) {\n\
    \    // consumer\n\
    \    %s\n\
    \  } else {\n\
    \    // epilogue\n\
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
    (depth desc)
    producer
    (emit_producer_body desc)
    consumer
    (emit_consumer_body desc)
    (emit_epilogue_body desc)
    tmem_dealloc
    |> fun s -> ignore smem_sz; s


let emit_host_launcher (desc : (_, _, _, _, _, _) Kernel_desc.t) : string =
  let et = elem_t desc in
  let n_warps = desc.Kernel_desc.cluster.Cluster.num_warps in
  let smem_sz = desc.Kernel_desc.pipeline.Pipeline.smem_bytes in
  let bm = desc.Kernel_desc.bm in
  let bn = desc.Kernel_desc.bn in
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
  let full_source = String.concat ~sep:"\n\n"
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
