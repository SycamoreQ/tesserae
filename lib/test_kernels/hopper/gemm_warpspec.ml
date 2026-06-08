open Tesserae_kernel
open Kernel_ast

let gemm_hopper_ws () =
  Kernel_ast.make
    ~name:"gemm_hopper_ws"
    ~arch:SM90a
    ~elem:F16
    ~tile:{ m = 128; n = 128; k = 64 }
    ~stages:4                        (* single smem slot, one TMA load *)
    ~args:[ ("A", F16, Global, In)
          ; ("B", F16, Global, In)
          ; ("C", F32, Global, Out) ]
    ~body:(
      warp_dispatch [
        (WarpIs 0, [                   (* producer: one load, no k loop *)
          load ~src:(Arg ("A", F16, Global))
               ~dst:(Smem ("smem_A", F16, { m = 128; n = 0; k = 64 })) ();
          load ~src:(Arg ("B", F16, Global))
               ~dst:(Smem ("smem_B", F16, { m = 0; n = 128; k = 64 })) ();
        ]);
        (WarpIn [4; 5; 6; 7], [ (* consumer: WGMMA + store *)

          mma
            (Smem ("smem_A", F16, { m = 128; n = 0; k = 64 }))
            (Smem ("smem_B", F16, { m = 0; n = 128; k = 64 }))
            (Arg  ("acc",    F32, Register));
          store ~src:(Arg ("acc", F32, Register))
                ~dst:(Arg ("C",   F32, Global)) ();
        ]);
      ]
    )
