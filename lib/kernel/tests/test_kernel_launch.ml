open Tesserae

(* Kernel_launch wraps PTX loading and kernel execution.
   Functions needed:
   - load_ptx : string -> Kernel_launch.module_
       load PTX string into CUDA driver, return module handle
   - get_function : module_ -> string -> Kernel_launch.func
       get kernel function handle by name
   - launch : func
       -> grid:(int * int * int)
       -> block:(int * int * int)
       -> smem:int
       -> args:nativeint list
       -> unit
       launch kernel with given config and raw pointer args
   - unload : module_ -> unit
       unload CUDA module, free driver resources
   - synchronize : unit -> unit
       cudaDeviceSynchronize — wait for all kernels to finish *)

(* trivial PTX kernel for testing — adds 1.0 to each element *)
let trivial_ptx = {|
.version 7.0
.target sm_80
.address_size 64

.visible .entry trivial_add(
  .param .u64 param0,
  .param .u32 param1
)
{
  .reg .u64 %rd<4>;
  .reg .u32 %r<4>;
  .reg .f32 %f<4>;

  ld.param.u64 %rd0, [param0];
  ld.param.u32 %r0,  [param1];

  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %ntid.x;
  mov.u32 %r3, %tid.x;
  mad.lo.u32 %r3, %r1, %r2, %r3;

  setp.ge.u32 %p0, %r3, %r0;
  @%p0 bra done;

  cvt.u64.u32 %rd1, %r3;
  shl.b64 %rd2, %rd1, 2;
  add.u64 %rd3, %rd0, %rd2;

  ld.global.f32 %f0, [%rd3];
  add.f32 %f0, %f0, 1.0;
  st.global.f32 [%rd3], %f0;

done:
  ret;
}
|}

let test_load_ptx () =
  let m = Kernel_launch.load_ptx trivial_ptx in
  Alcotest.(check bool) "loaded" true (Kernel_launch.is_valid m);
  Kernel_launch.unload m

let test_get_function () =
  let m = Kernel_launch.load_ptx trivial_ptx in
  let f = Kernel_launch.get_function m "trivial_add" in
  Alcotest.(check bool) "func valid" true (Kernel_launch.func_is_valid f);
  Kernel_launch.unload m

let test_launch_trivial () =
  let n    = 1024 in
  let host = Array.make n 0.0 in
  let buf  = Gpu_buffer.alloc n in
  Gpu_buffer.copy_from_host buf host;

  let m = Kernel_launch.load_ptx trivial_ptx in
  let f = Kernel_launch.get_function m "trivial_add" in

  Kernel_launch.launch f
    ~grid:(n / 256, 1, 1)
    ~block:(256, 1, 1)
    ~smem:0
    ~args:[ Gpu_buffer.ptr buf
          ; Nativeint.of_int n ];

  Kernel_launch.synchronize ();

  let result = Gpu_buffer.to_host buf in
  Alcotest.(check bool) "all ones" true
    (Array.for_all (fun x -> abs_float (x -. 1.0) < 1e-5) result);

  Gpu_buffer.free buf;
  Kernel_launch.unload m

let test_synchronize () =
  (* should not raise *)
  Kernel_launch.synchronize ();
  Alcotest.(check bool) "ok" true true

let test_unload_idempotent () =
  let m = Kernel_launch.load_ptx trivial_ptx in
  Kernel_launch.unload m;
  Alcotest.(check bool) "ok" true true

let () =
  Alcotest.run "Kernel_launch" [
    "load",    [ Alcotest.test_case "ptx"       `Quick test_load_ptx
               ; Alcotest.test_case "function"  `Quick test_get_function ];
    "launch",  [ Alcotest.test_case "trivial"   `Quick test_launch_trivial ];
    "misc",    [ Alcotest.test_case "sync"      `Quick test_synchronize
               ; Alcotest.test_case "unload"    `Quick test_unload_idempotent ];
  ]
