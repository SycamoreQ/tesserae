open Tesserae_kernel
open Kernel_ast

(* A Blackwell (SM100) GEMM kernel written in Tesserae.
   Cluster topology: 2 SMs (Cluster dims 2x1x1), 6 warps per CTA (192 threads).
   - Warp 0: Producer (TMA Multicast loads A & B tiles into shared memory)
   - Warp 1: Consumer (tcgen05.mma / MMA compute in warpgroup)
*)
let gemm_blackwell_ws () =
  Kernel_ast.make
    ~name:"gemm_blackwell_ws"
    ~arch:SM100a
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
        (WarpIs 1, [
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

(* Lowered pipeline variant using Kernel_desc.make_blackwell *)
let gemm_blackwell () =
  Kernel_ast.make
    ~name:"gemm_blackwell"
    ~arch:SM100a
    ~elem:F16
    ~tile:{ m = 128; n = 128; k = 64 }
    ~stages:4
    ~args:[ in_arg "A" F16
          ; in_arg "B" F16
          ; out_arg "C" F32 ]
    ~body:(Seq [])
