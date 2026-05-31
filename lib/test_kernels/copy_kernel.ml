open Tesserae_kernel

(*design a simple copy operation from global mem to shared mem*)

let copy_kernel () =
  Kernel_ast.make
    ~name:"copy_hopper"
    ~arch:Kernel_ast.SM90a
    ~elem:Kernel_ast.F16
    ~tile:{ Kernel_ast.m = 64; n = 64; k = 8 }
    ~stages:2
    ~args:[ ("A", Kernel_ast.F16, Kernel_ast.Global)
          ; ("C", Kernel_ast.F32, Kernel_ast.Global) ]
    ~body:(
      Kernel_ast.For ("k", 0, 8, [
        Kernel_ast.Load (
          Kernel_ast.Arg ("A", Kernel_ast.F16, Kernel_ast.Global),
          Kernel_ast.Smem ("smem_A", Kernel_ast.F16,
            { Kernel_ast.m = 64; n = 64; k = 8 }),
          None
        );
        Kernel_ast.Store (
          Kernel_ast.Smem ("smem_A", Kernel_ast.F16,
            { Kernel_ast.m = 64; n = 64; k = 8 }),
          Kernel_ast.Arg ("C", Kernel_ast.F32, Kernel_ast.Global),
          None
        );
      ])
    )
