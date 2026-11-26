(* Timing for the Blackwell warp-specialised GEMM.

   -l2 none,128,256 runs an in-process L2-promotion ablation.

   -pairs 0,60,64,68,72,74 runs an in-process persistent-cluster sweep. The
   kernel is compiled once per (k_total, stages, multicast, bn); the pairs
   list only changes the launch grid. This is the only way to compare cluster
   counts without cross-session noise.

   0 = full tile-shaped grid (one cluster per output tile; multi-wave on GPU,
   i.e. "no persistence" baseline).
   N > 0 = at most N resident clusters, each processes multiple tiles
   (single wave if N <= max resident on the GPU).

   dune exec lib/runtime/bench/bench_gemm.exe -- \
     -k 4104 -stages 3 -bn 256 -pairs 0,60,64,68,72,74
*)

open Tesserae_runtime
open Tesserae_kernel
open Base

module Blackwell_gemm = Tesserae_blackwell_kernels.Gemm_warpspec

let ceil_div a b = (a + b - 1) / b
let now () = Unix.gettimeofday ()

(* 4th arg is the TMA box height (tile rows) for the operand. *)
external create_tma_descriptor_sw128_l2 :
  nativeint -> int -> int -> int -> int -> nativeint
  = "caml_create_tma_descriptor_sw128_l2"

(* f32 C-output map: box 128 rows x 32 cols (32*4 = 128B, required by SW128). *)
external create_tma_descriptor_f32_sw128_l2 :
  nativeint -> int -> int -> int -> int -> nativeint
  = "caml_create_tma_descriptor_f32_sw128_l2"

let l2_name = function
  | 0 -> "L2_NONE"
  | 1 -> "L2_128B"
  | 2 -> "L2_256B"
  | m -> Stdlib.Format.asprintf "L2_%d" m

(* Returns a list of (pairs, clusters, l2_mode, best_secs, median_secs, bad). *)
let bench_shape ?(l2_modes = [1]) ~m ~n ~k_total ~stages ~bk ~warmup ~iters ~repeats
    ~multicast ~pairs_list ~bn () =
  if k_total % 8 <> 0 then failwith "k must be a multiple of 8";
  let epi_smem = 16384 + 1024 in
  let per_stage = 128 * bk * 2 + (bn / 2) * bk * 2 in
  let max_stages = (232448 - 320 - epi_smem) / per_stage in
  let stages =
    if stages > max_stages then begin
      Stdio.printf "note: clamping stages %d -> %d (smem limit with bk=%d)\n%!"
        stages max_stages bk;
      max_stages
    end else stages in
  let gx = let g = ceil_div m 128 in if g % 2 = 0 then g else g + 1 in
  let total_tiles = ceil_div m 256 * ceil_div n bn in

  let host_a = Array.init (m * k_total) ~f:(fun i ->
    let r = i / k_total and kk = i % k_total in
    Float.of_int (((r * 3 + kk * 5) % 7) - 3)) in
  let host_b = Array.init (n * k_total) ~f:(fun i ->
    let c = i / k_total and kk = i % k_total in
    Float.of_int (((c * 2 + kk * 3) % 7) - 3)) in
  let c_len = gx * 128 * n + 128 in

  let buf_a = Gpu_buffer.alloc_f16 (m * k_total) in
  let buf_b = Gpu_buffer.alloc_f16 (n * k_total) in
  let buf_c = Gpu_buffer.alloc c_len in
  Gpu_buffer.copy_from_host_f16 buf_a host_a;
  Gpu_buffer.copy_from_host_f16 buf_b host_b;
  Gpu_buffer.copy_from_host buf_c (Array.create ~len:c_len Float.nan);

  let result =
    match Compile.to_ptx (Blackwell_gemm.gemm_blackwell_ws ~k_total ~stages ~multicast ~bn ~bk ()) with
    | Error e ->
      failwith (Stdlib.Format.asprintf "Compilation failed: %a" Compile.pp_error e)
    | Ok result ->
      let ptx = match result.Compile.ptx with
        | Some p -> p
        | None -> failwith "No PTX generated"
      in
      let module_ = Kernel_launch.load_ptx ptx in
      let func = Kernel_launch.get_function module_ "gemm_blackwell_ws" in
      let smem_bytes = stages * per_stage + (2 * stages * 8 + 256) + epi_smem in

      Stdio.printf
        "shape M=%d N=%d K=%d stages=%d bn=%d smem=%d B total_tiles=%d\n%!"
        m n k_total stages bn smem_bytes total_tiles;

      let runs = List.concat_map pairs_list ~f:(fun pairs ->
        let clusters =
          if pairs = 0 then total_tiles
          else if pairs < 0 then begin
            let raw =
              Kernel_launch.max_active_clusters func
                ~block:(192, 1, 1) ~smem:smem_bytes ~cluster:(2, 1, 1) in
            let res = Int.max 1 (Int.min total_tiles raw) in
            (* Wave-aware sizing. The makespan is ceil(total/res) rounds; keep
               that minimum but shrink the cluster count to the smallest value
               that still achieves it, so the final round has no idle clusters.
               E.g. total=512,res=74 -> 7 rounds -> 74; total=256,res=74 -> 4
               rounds -> 64 (a divisor, unlike 74 which leaves 40 idle). *)
            let waves = (total_tiles + res - 1) / res in
            let c = (total_tiles + waves - 1) / waves in
            Stdio.printf
              "  [auto] cuOccupancyMaxActiveClusters=%d -> %d waves, using %d clusters\n%!"
              raw waves c;
            c
          end else Int.min total_tiles pairs in
        let grid_x = 2 * clusters in
        List.map l2_modes ~f:(fun l2_mode ->
          let tma_a = create_tma_descriptor_sw128_l2
            (Gpu_buffer.ptr buf_a) m k_total 128 l2_mode in
          let tma_b = create_tma_descriptor_sw128_l2
            (Gpu_buffer.ptr buf_b) n k_total (bn / 2) l2_mode in
          let tma_c = create_tma_descriptor_f32_sw128_l2
            (Gpu_buffer.ptr buf_c) m n 128 l2_mode in

          let launch () =
            Kernel_launch.launch func
              ~grid:(grid_x, 1, 1)
              ~block:(192, 1, 1)
              ~smem:smem_bytes
              ~cluster:(2, 1, 1)
              ~args:[ tma_a; tma_b; tma_c
                    ; Nativeint.of_int m; Nativeint.of_int n ] ()
          in

          for _ = 1 to warmup do launch () done;
          Kernel_launch.synchronize ();

          let result_host = Gpu_buffer.to_host buf_c in
          Stdlib.Random.init 42;
          let samples = 256 in
          let bad = ref 0 in
          for _ = 1 to samples do
            let r = Stdlib.Random.int m and c = Stdlib.Random.int n in
            let acc = ref 0.0 in
            for kk = 0 to k_total - 1 do
              acc := !acc +. host_a.(r * k_total + kk) *. host_b.(c * k_total + kk)
            done;
            if not (Float.equal result_host.(r * n + c) !acc) then Int.incr bad
          done;
          Stdio.printf "  [pairs=%3d clusters=%3d %s] spot check: %d/%d wrong\n%!"
            pairs clusters (l2_name l2_mode) !bad samples;

          let times = Array.init repeats ~f:(fun _ ->
            let t0 = now () in
            for _ = 1 to iters do launch () done;
            Kernel_launch.synchronize ();
            let t1 = now () in
            (t1 -. t0) /. Float.of_int iters) in
          Array.sort times ~compare:Float.compare;

          Kernel_launch.free_tma_descriptor tma_a;
          Kernel_launch.free_tma_descriptor tma_b;
          Kernel_launch.free_tma_descriptor tma_c;

          (pairs, clusters, l2_mode,
           times.(0), times.(Array.length times / 2), !bad)
        ))
      in

      Kernel_launch.unload module_;
      runs
  in
  Gpu_buffer.free buf_a;
  Gpu_buffer.free buf_b;
  Gpu_buffer.free buf_c;
  result

(* least squares fit y = a + b x *)
let fit (pts : (float * float) list) =
  let n = Float.of_int (List.length pts) in
  let sx = List.sum (module Float) pts ~f:fst in
  let sy = List.sum (module Float) pts ~f:snd in
  let sxx = List.sum (module Float) pts ~f:(fun (x, _) -> x *. x) in
  let sxy = List.sum (module Float) pts ~f:(fun (x, y) -> x *. y) in
  let b = (n *. sxy -. sx *. sy) /. (n *. sxx -. sx *. sx) in
  let a = (sy -. b *. sx) /. n in
  (a, b)

let () =
  let m = ref 4096 and n = ref 4096 and k = ref 4104 in
  let ks = ref "" in
  let stages = ref 3 in
  let warmup = ref 3 and iters = ref 20 and repeats = ref 5 in
  let peak = ref 0.0 in
  let l2_spec = ref "128" in
  let multicast = ref false in
  let pairs_spec = ref "0,60,64,68,72,74" in
  let bn = ref 256 in
  let bk = ref 128 in
  let spec = [
    ("-m", Stdlib.Arg.Set_int m, " rows of A and C (default 4096)");
    ("-n", Stdlib.Arg.Set_int n, " rows of B / columns of C (default 4096)");
    ("-k", Stdlib.Arg.Set_int k, " reduction size, multiple of 8 (default 4104)");
    ("-ks", Stdlib.Arg.Set_string ks,
     " comma-separated K values, e.g. 512,1024,2048,4096 (overrides -k)");
    ("-stages", Stdlib.Arg.Set_int stages,
     " pipeline stages (clamped to the smem limit; bk=128 allows 4 at bn=128, "
     ^ "3 at bn=256; bk=64 allows up to 8 at bn=128)");
    ("-warmup", Stdlib.Arg.Set_int warmup, " untimed launches per config (default 3)");
    ("-iters", Stdlib.Arg.Set_int iters, " launches per timed batch (default 20)");
    ("-repeats", Stdlib.Arg.Set_int repeats, " timed batches (default 5)");
    ("-l2", Stdlib.Arg.Set_string l2_spec,
     " comma-separated L2 promotion modes: none,128,256 (default: 128)");
    ("-multicast", Stdlib.Arg.Set multicast,
     " enable TMA multicast for B (default false)");
    ("-pairs", Stdlib.Arg.Set_string pairs_spec,
     " comma-separated persistent-cluster counts. "
     ^ "0 = full tile-shaped grid (one cluster per output tile). "
     ^ "N > 0 = up to N resident clusters, each processes multiple tiles. "
     ^ "N < 0 = occupancy-sized: use cuOccupancyMaxActiveClusters, then shrink "
     ^ "to the smallest cluster count that keeps the same number of tile-rounds. "
     ^ "(default: 0,60,64,68,72,74)");
    ("-bn", Stdlib.Arg.Set_int bn,
     " B tile width / CTA output columns, 128 or 256 (default 256)");
    ("-bk", Stdlib.Arg.Set_int bk,
     " K tile / stage depth, 64 or 128 (default 128). bk=64 halves per-stage "
     ^ "smem, enabling 2 CTAs/SM at bn=128 or a deeper pipeline.");
    ("-peak", Stdlib.Arg.Set_float peak,
     " dense f16 peak TFLOP/s of your GPU, to print percent of peak (optional)");
  ] in
  Stdlib.Arg.parse spec (fun _ -> ()) "bench_gemm [options]";
  let m = !m and n = !n in
  let k_list =
    if String.is_empty !ks then [ !k ]
    else List.map (String.split !ks ~on:',') ~f:(fun s ->
      Int.of_string (String.strip s))
  in
  let l2_modes =
    List.map (String.split !l2_spec ~on:',') ~f:(fun s ->
      match String.lowercase (String.strip s) with
      | "none" | "0" -> 0
      | "128" | "1" -> 1
      | "256" | "2" -> 2
      | s -> failwith (Printf.sprintf "invalid L2 mode: %s" s))
  in
  let pairs_list =
    List.map (String.split !pairs_spec ~on:',') ~f:(fun s ->
      Int.of_string (String.strip s))
  in
  Stdio.printf "reminder: debug_consumer_delay_us must be 0 for this to mean anything\n%!";

  let rows =
    List.map k_list ~f:(fun k_total ->
      let runs = bench_shape ~l2_modes ~m ~n ~k_total ~stages:!stages ~bk:!bk
                   ~warmup:!warmup ~iters:!iters ~repeats:!repeats
                   ~multicast:!multicast ~pairs_list ~bn:!bn () in
      List.map runs ~f:(fun (pairs, clusters, l2_mode, best, median, bad) ->
        let flops = 2.0 *. Float.of_int m *. Float.of_int n *. Float.of_int k_total in
        let tf t = flops /. t /. 1e12 in
        Stdio.printf "  [pairs=%3d clusters=%3d %s] best   : %8.3f ms  %8.1f TFLOP/s\n"
          pairs clusters (l2_name l2_mode) (best *. 1e3) (tf best);
        Stdio.printf "  [pairs=%3d clusters=%3d %s] median : %8.3f ms  %8.1f TFLOP/s\n%!"
          pairs clusters (l2_name l2_mode) (median *. 1e3) (tf median);
        if Float.(!peak > 0.0) then
          Stdio.printf "  [pairs=%3d] best is %.1f%% of the stated %.0f TFLOP/s peak\n%!"
            pairs (100.0 *. tf best /. !peak) !peak;
        (k_total, pairs, clusters, l2_mode, best, median, bad)))
    |> List.concat
  in

  Stdio.printf "\nsummary (M=%d N=%d stages=%d bn=%d)\n" m n !stages !bn;
  Stdio.printf "     K | pairs | clusters | L2      |  best ms |  med ms | TFLOP/s | wrong\n";
  List.iter rows ~f:(fun (k_total, pairs, clusters, l2_mode, best, median, bad) ->
    let flops = 2.0 *. Float.of_int m *. Float.of_int n *. Float.of_int k_total in
    Stdio.printf "%6d | %5d | %8d | %-7s | %9.3f | %8.3f | %7.1f | %d\n"
      k_total pairs clusters (l2_name l2_mode)
      (best *. 1e3) (median *. 1e3) (flops /. best /. 1e12) bad);

  if List.length l2_modes = 1 && List.length pairs_list = 1
     && List.length rows >= 2 then begin
    let pts = List.map rows ~f:(fun (k_total, _, _, _, best, _, _) ->
      (Float.of_int k_total, best)) in
    let (a, b) = fit pts in
    let flops_per_k = 2.0 *. Float.of_int m *. Float.of_int n in
    Stdio.printf "\nfit: time = %.1f us + %.3f us per 1024 K\n"
      (a *. 1e6) (b *. 1024.0 *. 1e6);
    Stdio.printf "main-loop rate from the slope: %.1f TFLOP/s\n"
      (flops_per_k /. b /. 1e12);
    Stdio.printf "per-launch overhead (intercept): %.1f us\n%!" (a *. 1e6)
  end
