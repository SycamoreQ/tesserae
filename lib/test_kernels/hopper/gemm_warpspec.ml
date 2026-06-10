open Tesserae_kernel
open Kernel_ast

(* A Hopper GEMM kernel written in Tesserae.

The producer iterates over K‑tiles (k_tile from 0 to 3), using TMA to load two 2D tiles from global memory into shared memory buffers:

Matrix	Shape	Layout in smem	Byte size per tile
A	128×64	M‑major (M×K)	128×64×2 = 16 KiB
B	64×128	K‑major (K×N)	64×128×2 = 16 KiB

TMA loads are pipelined with depth 4 (separate smem banks smem_A[4][8192] and smem_B[4][8192]).
For each TMA load, the producer issues an mbarrier.arrive.expect_tx with bytes = 16384 (the total tile size) and the load itself completes the transaction.
The same full_mbar is used for both loads; the mbarrier is initialised with a count of 2 (one for A, one for B).

- Synchronisation:
The consumer warpgroup waits on the full_mbar using mbarrier.try_wait.parity, toggling the phase every stage (phase = (k_tile / 4) % 2).
This ensures that the tile for the current K‑iteration is fully loaded before WGMMA begins.

- Matrix Multiply–Accumulate:

The consumer uses WGMMA instructions (wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16) to compute the GEMM.

Each iteration over k (0..3) processes one 64×128×16 fragment of the accumulator.
The M dimension (128) is split into two m64 sub‑tiles (tile_m = 0 and 1).
The K dimension (64) is split into 4 k‑fragments (each 16 columns of A / 16 rows of B).
Accumulator: 128 float registers per warpgroup (64 regs × 2 m64 sub‑tiles).

- Store to global memory:
After all K‑iterations, the consumer warpgroup stores the accumulator to float* C using a simple loop:

local_tid = threadIdx.x % 128 (only first 128 threads of the 256‑thread block are used).
Each thread writes acc_frag[i] to C[local_tid * 128 + i] for i = 0..127.
This gives a coalesced column‑major store (each warp writes 32 consecutive floats).

Swizzle = 0:
no swizzling, which avoids padding complexities but leaves performance on the table.
Later you can increase swizzle and adjust sbo accordingly.

Single producer warp:
the kernel is lightly loaded; for larger tiles you would use a full producer warpgroup (warps 0–3) to hide latency.

No double‑buffering via empty mbar:
the empty_mbar is initialised (count = 4) but not yet used for consumer→producer handshake;
the pipeline depth alone is enough for correctness because the consumer waits on parity.
*)

let gemm_hopper_ws () =
  Kernel_ast.make
    ~name:"gemm_hopper_ws"
    ~arch:SM90a
    ~elem:F16
    ~tile:{ m = 128; n = 128; k = 64 }
    ~stages:4
    ~args:[ ("A", F16, Global, In)
          ; ("B", F16, Global, In)
          ; ("C", F32, Global, Out) ]
    ~body:(
      warp_dispatch [
        (WarpIs 0, [
          For ("k_tile", 0, 4, [
            load ~src:(Arg ("A", F16, Global))
                 ~dst:(Smem ("smem_A", F16, { m = 128; n = 0; k = 64 })) ();
            load ~src:(Arg ("B", F16, Global))
                 ~dst:(Smem ("smem_B", F16, { m = 0; n = 128; k = 64 })) ();
          ])
        ]);
        (WarpIn [4; 5; 6; 7], [
          For ("k", 0, 4, [
            mma
              (Smem ("smem_A", F16, { m = 128; n = 0; k = 64 }))
              (Smem ("smem_B", F16, { m = 0; n = 128; k = 64 }))
              (Arg  ("acc",    F32, Register));
          ]);
          store ~src:(Arg ("acc", F32, Register))
                ~dst:(Arg ("C",   F32, Global)) ();
        ]);
      ]
    )
