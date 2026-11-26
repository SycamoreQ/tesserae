open Tesserae_kernel
open Kernel_ast


(* B tile = bn/2 columns per CTA: the 2-CTA tcgen05.mma shards B along N
   across the pair, so each CTA loads and holds its own half (bn=256 ->
   128 cols). Per-stage smem = (128 + bn/2) * bk * 2 bytes.
   NOTE: at bn=256 occupancy is pinned at 1 cluster per SM-pair regardless of
   smem — the two 256-column accumulators consume an SM's entire TMEM
   (512/512 columns). At bn=128 the pair accumulator is only 2*128 = 256
   columns, so 2 clusters/SM can co-reside if smem also fits (bk=64 keeps a
   stage small enough). *)
let gemm_blackwell_ws ?(k_total = 256) ?(stages = 4) ?(multicast = false)
    ?(tiles_per_cta = 2) ?(bn = 256) ?(bk = 128) () =
  let k_tiles = (k_total + bk - 1) / bk in
  let k = Kernel_ast.make
    ~name:"gemm_blackwell_ws"
    ~arch:SM100a
    ~elem:F16
    ~tile:{ m = 128; n = bn/2; k = bk }
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
                   ~dst:(Smem ("smem_A", F16, { m = 128; n = 0; k = bk })) ();
              load ~src:(Arg ("B", F16, Global))
                   ~dst:(Smem ("smem_B", F16, { m = 0; n = bn/2; k = bk })) ();
            ])
          ])
        ]);
        (WarpIs 1, [
          For ("tile", 0, tiles_per_cta, [
            For ("k", 0, k_tiles, [
              mma
                (Smem ("smem_A", F16, { m = 128; n = 0; k = bk }))
                (Smem ("smem_B", F16, { m = 0; n = bn/2; k = bk }))
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
