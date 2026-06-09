open Tesserae_kernel
open Kernel_ast

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
        (* Warp 0 issues TMA — warpgroup 0, only 1 thread calls cp.async.bulk *)
        (WarpIs 0, [  (*no issues with 0 being producer*)
          For ("k_tile", 0, 4, [
            load ~src:(Arg ("A", F16, Global))
                 ~dst:(Smem ("smem_A", F16, { m = 128; n = 0; k = 64 })) ();
            load ~src:(Arg ("B", F16, Global))
                 ~dst:(Smem ("smem_B", F16, { m = 0; n = 128; k = 64 })) ();
          ])
        ]);

        (* ALL of warpgroup 1 (threads 128-255) — wgmma.sync requires
           every thread in the warpgroup to execute the instruction         *)
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
