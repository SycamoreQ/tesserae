open Tesserae_kernel
open Kernel_ast


(* B tile = bn columns per CTA stage. bn=128: 32 KB/stage, 2.1 MFLOP;
   bn=256: 48 KB/stage, 4.2 MFLOP — halves the smem/traffic per FLOP.
   The two per-tile accumulators then use 2 * 256 = all 512 TMEM columns
   (one CTA per SM) and the epilogue reads TMEM in 128-column chunks. *)
let gemm_blackwell_ws ?(k_total = 256) ?(stages = 4) ?(multicast = false)
    ?(tiles_per_cta = 2) ?(bn = 256) () =
  let k_tiles = (k_total + 63) / 64 in
  let k = Kernel_ast.make
    ~name:"gemm_blackwell_ws"
    ~arch:SM100a
    ~elem:F16
    ~tile:{ m = 128; n = bn; k = 64 }
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
                   ~dst:(Smem ("smem_A", F16, { m = 128; n = 0; k = 64 })) ();
              load ~src:(Arg ("B", F16, Global))
                   ~dst:(Smem ("smem_B", F16, { m = 0; n = bn; k = 64 })) ();
            ])
          ])
        ]);
        (WarpIs 1, [
          For ("tile", 0, tiles_per_cta, [
            For ("k", 0, k_tiles, [
              mma
                (Smem ("smem_A", F16, { m = 128; n = 0; k = 64 }))
                (Smem ("smem_B", F16, { m = 0; n = bn; k = 64 }))
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
