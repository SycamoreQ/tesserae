open Tesserae_kernel
open Kernel_ast


(* B tile = bn/2 columns per CTA: the 2-CTA tcgen05.mma shards B along N
   across the pair, so each CTA loads and holds its own half (bn=256 ->
   128 cols). With bk=128 each stage holds 32 KB of A and 32 KB of B
   (64 KB total), capping the pipeline at 3 stages; SW128 TMA sub-tiles
   stay 64 elements wide, so each stage is filled by two 64-wide TMA ops.
   The pair accumulator is 2 * 256 = all 512 TMEM columns across the two
   per-tile buffers, and the epilogue reads TMEM in 128-column chunks. *)
let gemm_blackwell_ws ?(k_total = 256) ?(stages = 4) ?(multicast = false)
    ?(tiles_per_cta = 2) ?(bn = 256) () =
  let k_tiles = (k_total+127)/128 in
  let k = Kernel_ast.make
    ~name:"gemm_blackwell_ws"
    ~arch:SM100a
    ~elem:F16
    ~tile:{ m = 128; n = bn/2; k = 128 }
    ~stages
    ~args:[ ("A", F16, Global, In)
          ; ("B", F16, Global, In)
          ; ("C", F32, Global, Out)
          ; ("M", S32, Global, In)
          ; ("N", S32, Global, In)]
    ~body:(
      warp_dispatch [
        (WarpIs 0, [
          For ("tile", 0, tiles_per_cta, [
            For ("k_tile", 0, k_tiles, [
              load ~src:(Arg ("A", F16, Global))
                   ~dst:(Smem ("smem_A", F16, { m = 128; n = 0; k = 128 })) ();
              load ~src:(Arg ("B", F16, Global))
                   ~dst:(Smem ("smem_B", F16, { m = 0; n = bn/2; k = 128 })) ();
            ])
          ])
        ]);
        (WarpIs 1, [
          For ("tile", 0, tiles_per_cta, [
            For ("k", 0, k_tiles, [
              mma
                (Smem ("smem_A", F16, { m = 128; n = 0; k = 128 }))
                (Smem ("smem_B", F16, { m = 0; n = bn/2; k = 128 }))
                (Arg  ("acc",    F32, Register));
            ]);
            (* No store here, it goes to the Epilogue warps *)
          ])
        ]);
        (WarpIn [2; 3; 4; 5], [
          For ("tile", 0, tiles_per_cta, [
            store ~src:(Arg ("acc", F32, Register))
                  ~dst:(Arg ("C",   F32, Global)) ();
          ])
        ]);
      ]
    )
  in
  { k with multicast }
