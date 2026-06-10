# Tesserae

A native OCaml DSL for writing high-performance GPU kernels (GEMM, convolutions, etc.) targeting NVIDIA CUDA via NVRTC. Tesserae lets you express kernels in a typed, functional style, then compiles them to PTX and launches them on the device without leaving the OCaml runtime.

The project is still under development.

## Features

- **CuTe and CUTLASS** - Tesserae has implemented CuTe and CUTLASS algebra for example Layouts, Tensors, Tiling, Atoms and has baked that into the compile pipeline. So any kernel that is being written will go through these core modules. 

- **Type-safe kernel AST** — define kernels with typed tensor expressions, memory spaces (Global/Shared/Register/Tensor), and architecture-specific operations.

- **Multi-arch codegen** — SM80 (Ampere), SM90a (Hopper), SM100a (Blackwell) with automatic TMA, WGMMA, and tcgen05 lowering.

- **NVRTC JIT compilation** — kernels are compiled at runtime via NVRTC, linked with the CUDA driver API, and launched from OCaml.

- **Tirix** - Tesserae has its own built in IR called tirix that lets 
you also write kernels which passes through tirix and therefore you can write kernels in native OCaml and can be compiled through the nvrtc. 

- More details in the module specific README

## Quick start
- If you have a H100 locally: 

```bash
# Install OCaml dependencies
opam install dune base stdio alcotest

# Build
dune build

# Run all tests
dune test

# Run a specific test executable
dune exec lib/runtime/tests/test_runtime.exe
```

- If not: 

```bash
# use the modal run script if you have a GPU instance. 
modal run run_modal.py
```

## Project layout

```
lib/
  atoms/           Contains CUTLASS Atom implementations (MMA atoms etc)
  core/            Contains CuTe layout algebra implementations
  backend/         NVRTC compilation and PTX emission 
  kernel/          Kernel AST, lowering, and CUDA driver stubs
  pipeline/        Contains pipeline helpers like shared mem descs, TMEM
  runtime/         GPU buffer management, kernel launch, and tests
  test_kernels/    Example kernels (copy, GEMM stubs)
  tirix/           Intermediate IR (Tirix) and arch-specific emission
```

## Architecture

```
Kernel_ast  →  Tirix IR  →  Backend (NVRTC/PTX)  →  CUDA Driver Launch
     ↑                                              ↓
  OCaml DSL                                   GPU Buffer (host/device)
```

## Current Stand on the Project 

- Tesserae now has all of its tests passing and also has successfully ran a Load and Store on a H100 instance through the TMA. The next goal of the project will be to write full GEMM kernels in Hopper and fix any existing bugs. 

- Wrote a basic Hopper kernel with swizzle = 0 , one producer and consumer warp. I will iterate over it later.

- A Lexer should also be built on top of kernel_ast which helps in having some syntactic sugar to write native Tesserae Kernels. 

## Requirements

- OCaml 5.x + opam
- CUDA 12.x/13.x with NVRTC
- NVIDIA GPU (Ampere/Hopper/Blackwell)

## License

MIT
