open Base
open Tesserae_kernel
open Tesserae_runtime

(* Trivial PTX kernel for testing — adds 1.0 to each element *)
let trivial_ptx = {|
.version 7.8
.target sm_90
.address_size 64

.visible .entry trivial_add(
  .param .u64 param0,
  .param .u64 param1
)
{
  .reg .u64 %rd<4>;
  .reg .u32 %r<4>;
  .reg .f32 %f<2>;
  .reg .pred %p<2>;

  ld.param.u64 %rd0, [param0];
  ld.param.u64 %rd3, [param1];

  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %ntid.x;
  mov.u32 %r3, %tid.x;
  mad.lo.u32 %r3, %r1, %r2, %r3;

  // Convert thread index to 64-bit for the boundary check
  cvt.u64.u32 %rd1, %r3;
  setp.ge.u64 %p0, %rd1, %rd3;
  @%p0 bra done;

  // Calculate byte offset correctly (thread_idx * 4 bytes for float)
  mul.wide.u32 %rd2, %r3, 4;
  add.u64 %rd1, %rd0, %rd2;

  ld.global.f32 %f0, [%rd1];
  add.f32 %f0, %f0, 1.0;
  st.global.f32 [%rd1], %f0;

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
  let n = 1024 in
  let host = Array.init n ~f:(fun _ -> 0.0) in
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
    let is_all_ones = Array.for_all result ~f:(fun x -> Float.(abs (x - 1.0) < 1e-5)) in
    Alcotest.(check bool) "all ones" true is_all_ones;

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
    "load",   [ Alcotest.test_case "ptx"      `Quick test_load_ptx
               ; Alcotest.test_case "function" `Quick test_get_function ];
    "launch", [ Alcotest.test_case "trivial"   `Quick test_launch_trivial ];
    "misc",   [ Alcotest.test_case "sync"      `Quick test_synchronize
               ; Alcotest.test_case "unload"    `Quick test_unload_idempotent ];
  ]
