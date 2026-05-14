open Base
open Tesserae_core
open Tesserae_atoms
open Tesserae_pipeline


type family =
  | Ampere
  | Hopper
  | Blackwell

type ('arch, 'a, 'b, 'c, 'd, 'elem) t = {
  family : family;
  name : string;
  bm : int;
  bn : int;
  bk : int;
  tile_io : 'elem Tile_io.t;
  pipeline : Pipeline.t;
  tile_op : ('arch, 'a, 'b, 'c, 'd) Tile_op.t;
  pred_a : Predicate.t;
  pred_b : Predicate.t;
  pred_c : Predicate.t;
  cluster : Cluster.t;
  sm_count : int;
}

let make_ampere
    ~(name : string)
    ~(bm : int) ~(bn : int) ~(bk : int)
    ~(elem : 'elem Elemtype.t)
    ~(m : int) ~(n : int) ~(k : int) =
  let i n = Modes.Int n in
  let tup ts = Modes.Tuple ts in
  let lay s d = Layout.make s d in
  let swizzle  = Swizzle.smem_selector elem bm bk in
  let cluster  = Cluster.make
    { Cluster.x = 1; y = 1; z = 1 } 4
    [ (0, Cluster.Producer); (1, Cluster.Consumer)
    ; (2, Cluster.Epilogue);  (3, Cluster.Epilogue) ] in
  let pipeline = Pipeline.make 4
    ((bm + bn) * bk * Elemtype.byte_width elem) in
  let tile_io = Tile_io.make
    Tile_io.CpAsync elem cluster pipeline swizzle bm bn bk in
  let atom = Mma_atom.sm80_16x8x16_f32f16f16f32
    Mma_atom.ColMajor Mma_atom.RowMajor in
  let tiled_mma = Tiled_mma.make atom
    (lay (i 32) (i 1))
    (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
    (tup [i bm; i bn]) in
  let tile_op  = Tile_op.make tiled_mma Tile_op.Registers () in
  let pred_a   = Predicate.make
    (lay (tup [i bm; i bk]) (tup [i 1; i bm])) [m; k] in
  let pred_b   = Predicate.make
    (lay (tup [i bk; i bn]) (tup [i 1; i bk])) [k; n] in
  let pred_c   = Predicate.make
    (lay (tup [i bm; i bn]) (tup [i 1; i bm])) [m; n] in
  { family   = Ampere
  ; name
  ; bm; bn; bk
  ; tile_io; pipeline; tile_op
  ; pred_a; pred_b; pred_c
  ; cluster
  ; sm_count = 108  (* A100 SM count *)
  }


let make_hopper
    ~(name : string)
    ~(bm : int) ~(bn : int) ~(bk : int)
    ~(elem : 'elem Elemtype.t)
    ~(m : int) ~(n : int) ~(k : int)
  =
  let i n = Modes.Int n in
  let tup ts = Modes.Tuple ts in
  let lay s d = Layout.make s d in
  let swizzle  = Swizzle.smem_selector elem bm bk in
  let cluster = Cluster.make
  { Cluster.x = 2; y = 1; z = 1 }
      8
      [ (0, Cluster.Producer)
      ; (1, Cluster.Consumer)
      ; (2, Cluster.Consumer)
      ; (3, Cluster.Consumer)
      ; (4, Cluster.Consumer)
      ; (5, Cluster.Epilogue)
      ; (6, Cluster.Epilogue)
      ; (7, Cluster.Epilogue) ] in
  let pipeline = Pipeline.make 4
    ((bm + bn) * bk * Elemtype.byte_width elem) in
  let tile_io  = Tile_io.make
    Tile_io.TmaLoad elem cluster pipeline swizzle bm bn bk in
  let atom = Mma_atom.sm90_64x64x16_f32bf16bf16f32
    Mma_atom.ColMajor Mma_atom.RowMajor in
  let tiled_mma = Tiled_mma.make atom
    (lay (i 128) (i 1))
    (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
    (tup [i bm; i bn]) in
  let tile_op  = Tile_op.make tiled_mma Tile_op.Registers () in
  let pred_a   = Predicate.make
    (lay (tup [i bm; i bk]) (tup [i 1; i bm])) [m; k] in
  let pred_b   = Predicate.make
    (lay (tup [i bk; i bn]) (tup [i 1; i bk])) [k; n] in
  let pred_c   = Predicate.make
    (lay (tup [i bm; i bn]) (tup [i 1; i bm])) [m; n] in

  { family = Hopper
  ; name
  ; bm; bn; bk
  ; tile_io; pipeline; tile_op
  ; pred_a; pred_b; pred_c
  ; cluster
  ; sm_count = 132
  }



let make_blackwell
    ~(name : string)
    ~(bm : int) ~(bn : int) ~(bk : int)
    ~(elem : 'elem Elemtype.t)
    ~(m : int) ~(n : int) ~(k : int) =

    let i n = Modes.Int n in
    let tup ts = Modes.Tuple ts in
    let lay s d = Layout.make s d in
    let swizzle  = Swizzle.smem_selector elem bm bk in
    let cluster = Cluster.make
      { Cluster.x = 2; y = 1; z = 1 } 6
      [ (0, Cluster.Producer)
      ; (1, Cluster.Consumer)
      ; (2, Cluster.Epilogue)
      ; (3, Cluster.Epilogue)
      ; (4, Cluster.Epilogue)
      ; (5, Cluster.Scheduler) ] in
    let pipeline = Pipeline.make 4
      ((bm + bn) * bk * Elemtype.byte_width elem) in
    let tile_io  = Tile_io.make
      Tile_io.TmaMulticast elem cluster pipeline swizzle bm bn bk in
    let atom = Mma_atom.sm100_128x128x16_f32f16f16f32
      Mma_atom.ColMajor Mma_atom.RowMajor in
    let tiled_mma = Tiled_mma.make atom
      (lay (i 128) (i 1))
      (lay (tup [i 1; i 1]) (tup [i 1; i 0]))
      (tup [i bm; i bn]) in
    let tmem = Tmem.double_buf_make ~cta_group:Tmem.CTA2 ~num_rows:bm in
    let tile_op = Tile_op.make tiled_mma Tile_op.TensorMem ~tmem ~double_buf:true () in
    let pred_a   = Predicate.make
      (lay (tup [i bm; i bk]) (tup [i 1; i bm])) [m; k] in
    let pred_b   = Predicate.make
      (lay (tup [i bk; i bn]) (tup [i 1; i bk])) [k; n] in
    let pred_c   = Predicate.make
      (lay (tup [i bm; i bn]) (tup [i 1; i bm])) [m; n] in

    { family = Blackwell
    ; name
    ; bm; bn; bk
    ; tile_io; pipeline; tile_op
    ; pred_a; pred_b; pred_c
    ; cluster
    ; sm_count = 148
    }

let smem_bytes (t : (_, _, _, _, _, _) t) : int =
  t.pipeline.Pipeline.smem_bytes

let validate (t : (_, _, _, _, _, _) t) : (unit, string) Result.t =
  let is_pow2 x = x > 0 && (x land (x - 1) = 0) in
  let (atom_m, atom_n, atom_k) = Mma_atom.shape t.tile_op.Tile_op.tiled_mma.Tiled_mma.atom in
  let smem = smem_bytes t in
  let is_blackwell = match t.family with Blackwell -> true | _ -> false in

  if not (is_pow2 t.bm && is_pow2 t.bn && is_pow2 t.bk) then
    Error "Matrix dimensions (bm, bn, bk) must be positive powers of 2"

  else if (t.bm % atom_m <> 0) ||
          (t.bn % atom_n <> 0) ||
          (t.bk % atom_k <> 0) then
    Error (Printf.sprintf
      "Tile dimensions must be divisible by atoms: M=%d, N=%d, K=%d"
      atom_m atom_n atom_k)

  else if smem > (t.sm_count * 227 * 1024) then
    Error "Total pipeline shared memory exceeds the limit of sm_count * 227KB"

    else if is_blackwell && not (Cluster.is_2sm t.cluster) then
        Error "Blackwell requires 2SM cluster"

  else
    Ok ()


let arithmetic_intensity (t : (_, _, _, _, _, _) t) : float =
  let bw = Elemtype.byte_width
    t.tile_io.Tile_io.tiled_copy_a.Tiled_copy.atom.Copy_atom.elem_type in
  let effective_bn = match t.family with
    | Blackwell -> t.bn / 2
    | _ -> t.bn
  in
  (* compute is always full bm*bn, data movement uses effective_bn *)
  2.0 *. Float.of_int t.bm *. Float.of_int t.bn *. Float.of_int t.bk
  /. Float.of_int ((t.bm + effective_bn) * t.bk * bw)



let num_warps (t : (_, _, _, _, _, _) t) : int =
  t.cluster.Cluster.num_warps

let emit_kernel_params (t : (_, _, _, _, _, _) t) : string =
  let dtype = Elemtype.cpp_name
    t.tile_io.Tile_io.tiled_copy_a.Tiled_copy.atom.Copy_atom.elem_type in
  let tmap_params = match t.family with
    | Ampere -> ""
    | Hopper | Blackwell ->
      ",\n    const __grid_constant__ CUtensorMap A_tmap,\
       \n    const __grid_constant__ CUtensorMap B_tmap"
  in
  Printf.sprintf
    "const %s* A, const %s* B, float* C,\n    int M, int N, int K%s"
    dtype dtype tmap_params

let emit_launch_config (t : (_, _, _, _, _, _) t) (m : int) (n : int) : string =
  let grid_x = (m + t.bm - 1) / t.bm in
  let grid_y = (n + t.bn - 1) / t.bn in
  let threads = num_warps t * 32 in
  let smem = smem_bytes t in
  let cluster_part = match t.family with
    | Blackwell ->
      Printf.sprintf "\ncudaLaunchConfig_t cfg = {};\ncfg.gridDim = grid;\ncfg.blockDim = block;\n%s"
        (Cluster.emit_cluster_attr t.cluster)
    | _ -> "" in

  Printf.sprintf
    "\ncudaLaunchConfig_t cfg = {};
     dim3 grid(%d, %d, 1);\n\
     dim3 block(%d, 1, 1);\n\
     int smem_size = %d; // %d bytes per SM
     cluster = %s"
    grid_x grid_y threads smem smem cluster_part



let pp fmt t =
  let fam = match t.family with
    | Ampere -> "Ampere" | Hopper -> "Hopper" | Blackwell -> "Blackwell"
  in
  Stdlib.Format.fprintf fmt "Kernel(%s bm=%d bn=%d bk=%d warps=%d smem=%d)"
    fam t.bm t.bn t.bk (num_warps t) (smem_bytes t)
