open Tesserae_kernel

(* B200 Blackwell copy kernel performing TMA multicast load and global store *)
let copy_kernel () =
  Kernel_ast.make
    ~name:"copy_blackwell"
    ~arch:SM100a
    ~elem:F16
    ~tile:{ m = 64; n = 64; k = 64 }
    ~stages:1
    ~args:[ ("A", F16, Global, In); ("C", F16, Global, Out) ]
    ~body:(
      Seq [
        Load (Arg ("A", F16, Global),
              Smem ("smem_A", F16, { m = 64; n = 0; k = 64 }), None);
        Store (Smem ("smem_A", F16, { m = 64; n = 0; k = 64 }),
               Arg ("C", F16, Global), None);
      ]
    )
