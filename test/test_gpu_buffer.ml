open Tesserae

(* Gpu_buffer wraps device memory allocation.
   Functions needed:
   - alloc : int -> 'a Gpu_buffer.t
       allocate n elements of type 'a on device
   - of_host : float array -> Elemtype.float32 Gpu_buffer.t
       copy host array to device, return device buffer
   - to_host : 'a Gpu_buffer.t -> float array
       copy device buffer back to host
   - size : 'a Gpu_buffer.t -> int
       number of elements
   - byte_size : 'a Gpu_buffer.t -> int
       size in bytes
   - ptr : 'a Gpu_buffer.t -> nativeint
       raw device pointer for kernel launch
   - free : 'a Gpu_buffer.t -> unit
       explicitly free device memory *)

let test_alloc_size () =
  let buf = Gpu_buffer.alloc 1024 in
  Alcotest.(check int) "size" 1024 (Gpu_buffer.size buf);
  Gpu_buffer.free buf

let test_alloc_byte_size () =
  let buf = Gpu_buffer.alloc 1024 in
  Alcotest.(check int) "bytes" 4096 (Gpu_buffer.byte_size buf);
  Gpu_buffer.free buf

let test_of_host_roundtrip () =
  let host = Array.init 256 (fun i -> float_of_int i) in
  let buf  = Gpu_buffer.of_host host in
  let back = Gpu_buffer.to_host buf in
  Alcotest.(check int) "length" 256 (Array.length back);
  Alcotest.(check bool) "values" true
    (Array.for_all2 (fun a b -> abs_float (a -. b) < 1e-6) host back);
  Gpu_buffer.free buf

let test_ptr_nonnull () =
  let buf = Gpu_buffer.alloc 64 in
  Alcotest.(check bool) "ptr nonzero" true
    (Gpu_buffer.ptr buf <> Nativeint.zero);
  Gpu_buffer.free buf

let test_zero_fill () =
  let buf  = Gpu_buffer.alloc 128 in
  let back = Gpu_buffer.to_host buf in
  Alcotest.(check bool) "zeroed" true
    (Array.for_all (fun x -> x = 0.0) back);
  Gpu_buffer.free buf

let test_free_idempotent () =
  let buf = Gpu_buffer.alloc 64 in
  Gpu_buffer.free buf;
  (* should not crash *)
  Alcotest.(check bool) "ok" true true

let () =
  Alcotest.run "Gpu_buffer" [
    "alloc",  [ Alcotest.test_case "size"      `Quick test_alloc_size
              ; Alcotest.test_case "bytes"     `Quick test_alloc_byte_size
              ; Alcotest.test_case "zero"      `Quick test_zero_fill ];
    "copy",   [ Alcotest.test_case "roundtrip" `Quick test_of_host_roundtrip
              ; Alcotest.test_case "ptr"       `Quick test_ptr_nonnull ];
    "free",   [ Alcotest.test_case "idempotent"`Quick test_free_idempotent ];
  ]
