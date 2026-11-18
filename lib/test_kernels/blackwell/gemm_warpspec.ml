open Tesserae_kernel
open Kernel_ast


let gemm_blackwell_ws ?(k_total = 256) ?(stages = 4) ?(multicast = false) () =
  let k_tiles = (k_total + 63) / 64 in
  let k = Kernel_ast.make
    ~name:"gemm_blackwell_ws"
    ~arch:SM100a
    ~elem:F16
    ~tile:{ m = 128; n = 128; k = 64 }
    ~stages
    ~args:[ ("A", F16, Global, In)
          ; ("B", F16, Global, In)
          ; ("C", F32, Global, Out)
          ; ("M", S32, Global, In)
          ; ("N", S32, Global, In)]
    ~body:(
      warp_dispatch [
        (WarpIs 0, [
          For ("k_tile", 0, k_tiles, [
            load ~src:(Arg ("A", F16, Global))
                 ~dst:(Smem ("smem_A", F16, { m = 128; n = 0; k = 64 })) ();
            load ~src:(Arg ("B", F16, Global))
                 ~dst:(Smem ("smem_B", F16, { m = 0; n = 128; k = 64 })) ();
          ])
        ]);
        (WarpIs 1, [
          For ("k", 0, k_tiles, [
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
  in
  { k with multicast }
