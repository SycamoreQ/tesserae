open Base
open Tesserae_kernel
open Tesserae_pipeline
open Tesserae_core

let is_available () : bool =
  match Kernel_launch.device_info () with
  | _ -> true
  | exception _ -> false

let device_info () : string =
  Kernel_launch.device_info ()


let run (k : Kernel_ast.kernel) ~m ~n ~k_ ~a ~b =
  match Compile.to_ptx k with
  | Error e ->
    let msg = Stdlib.Format.asprintf "%a" Compile.pp_error e in
    Stdio.eprintf "COMPILE ERROR: %s\n%!" msg;
    Error msg
  | Ok r ->
    let ptx = Option.value_exn r.Compile.ptx in
    Stdio.eprintf "=== PTX ===\n%s\n=== END PTX ===\n%!" ptx;
    match (try Ok (Kernel_launch.load_ptx ptx)
           with Failure msg -> Error ("load_ptx: " ^ msg)) with
    | Error msg ->
      Stdio.eprintf "LOAD ERROR: %s\n%!" msg;
      Error msg
    | Ok module_ ->
      let func =
        try Kernel_launch.get_function module_ k.Kernel_ast.name
        with Failure msg -> failwith ("get_function: " ^ msg)
      in
      let bm = k.Kernel_ast.tile.Kernel_ast.m in
      let bn = k.Kernel_ast.tile.Kernel_ast.n in
      let bk = k.Kernel_ast.tile.Kernel_ast.k in
      (* Compute launch parameters from the correct descriptor *)
      let smem_allocated, n_threads =
        match k.Kernel_ast.arch with
        | Kernel_ast.SM90a ->
          let d = Kernel_desc.make_hopper
            ~name:k.Kernel_ast.name ~bm ~bn ~bk
            ~elem:Elemtype.Float16 ~m ~n ~k:k_ in
          (Kernel_desc.smem_bytes d, Cluster.thread_count d.Kernel_desc.cluster)
        | Kernel_ast.SM80 ->
          let d = Kernel_desc.make_ampere
            ~name:k.Kernel_ast.name ~bm ~bn ~bk
            ~elem:Elemtype.Float16 ~m ~n ~k:k_ in
          (Kernel_desc.smem_bytes d, Cluster.thread_count d.Kernel_desc.cluster)
        | Kernel_ast.SM100a ->
          let d = Kernel_desc.make_blackwell
            ~name:k.Kernel_ast.name ~bm ~bn ~bk
            ~elem:Elemtype.Float16 ~m ~n ~k:k_ in
          (Kernel_desc.smem_bytes d, Cluster.thread_count d.Kernel_desc.cluster)
      in
      let dev_a = Gpu_buffer.of_host a in
      let dev_b = Gpu_buffer.of_host b in
      let dev_c = Gpu_buffer.alloc (m * n) in
      let a_tmap = Kernel_launch.create_tma_descriptor
        (Gpu_buffer.ptr dev_a) m k_ bm bk in
      let b_tmap = Kernel_launch.create_tma_descriptor
        (Gpu_buffer.ptr dev_b) n k_ bn bk in
      let cleanup () =
        Kernel_launch.free_tma_descriptor a_tmap;
        Kernel_launch.free_tma_descriptor b_tmap;
        Gpu_buffer.free dev_a;
        Gpu_buffer.free dev_b;
        Gpu_buffer.free dev_c;
        Kernel_launch.unload module_
      in
      (match (try
                Kernel_launch.launch func
                  ~grid:((m + bm - 1) / bm, (n + bn - 1) / bn, 1)
                  ~block:(n_threads, 1, 1)
                  ~smem:smem_allocated
                  ~args:[ a_tmap
                        ; b_tmap
                        ; Gpu_buffer.ptr dev_c
                        ; Nativeint.of_int m
                        ; Nativeint.of_int n
                        ; Nativeint.of_int k_ ];
                Kernel_launch.synchronize ();
                Ok ()
              with Failure msg -> Error msg) with
      | Error msg ->
        cleanup ();
        Error ("launch: " ^ msg)
      | Ok () ->
        let result = Gpu_buffer.to_host dev_c in
        cleanup ();
        Ok result)

let run_exn k ~m ~n ~k_ ~a ~b =
  match run k ~m ~n ~k_ ~a ~b with
  | Ok r    -> r
  | Error e -> failwith e
