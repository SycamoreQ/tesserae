open Base

type run_error =
  | CompileError of string
  | LaunchError  of string

let is_available () : bool =
  match Kernel_launch.device_info () with
  | _ -> true
  | exception _ -> false

let device_info () : string =
  Kernel_launch.device_info ()

let run (k : Kernel_ast.kernel)
    ~(m : int) ~(n : int) ~(k_ : int)
    ~(a : float array)
    ~(b : float array)
  : (float array, string) Result.t =
  (* 1. compile to PTX *)
  match Compile.to_ptx k with
  | Error e ->
    let msg = Stdlib.Format.asprintf "%a" Compile.pp_error e in
    Error msg
  | Ok r ->
    let ptx = Option.value_exn r.Compile.ptx in
    (* 2. load PTX *)
    match (try Ok (Kernel_launch.load_ptx ptx)
           with Failure msg -> Error msg) with
    | Error msg -> Error ("load_ptx: " ^ msg)
    | Ok module_ ->
      (* 3. get function *)
      let func = Kernel_launch.get_function module_ k.Kernel_ast.name in
      (* 4. allocate device buffers *)
      let dev_a = Gpu_buffer.of_host a in
      let dev_b = Gpu_buffer.of_host b in
      let dev_c = Gpu_buffer.alloc (m * n) in
      (* 5. launch *)
      let bm = k.Kernel_ast.tile.Kernel_ast.m in
      let bn = k.Kernel_ast.tile.Kernel_ast.n in
      let nwarps = Cluster.thread_count
        (Kernel_desc.make_ampere
          ~name:"_" ~bm ~bn ~bk:32
          ~elem:Elemtype.Float16
          ~m ~n ~k:k_).Kernel_desc.cluster
      in
      ignore nwarps;
      (match (try
        Kernel_launch.launch func
          ~grid:((m + bm - 1) / bm, (n + bn - 1) / bn, 1)
          ~block:(128, 1, 1)
          ~smem:r.Compile.source |> String.length |> ignore; 0
          ~args:[ Gpu_buffer.ptr dev_a
                ; Gpu_buffer.ptr dev_b
                ; Gpu_buffer.ptr dev_c
                ; Nativeint.of_int m
                ; Nativeint.of_int n
                ; Nativeint.of_int k_ ];
        Kernel_launch.synchronize ();
        Ok ()
        with Failure msg -> Error msg) with
      | Error msg ->
        Gpu_buffer.free dev_a;
        Gpu_buffer.free dev_b;
        Gpu_buffer.free dev_c;
        Kernel_launch.unload module_;
        Error ("launch: " ^ msg)
      | Ok () ->
        (* 6. copy result back *)
        let result = Gpu_buffer.to_host dev_c in
        Gpu_buffer.free dev_a;
        Gpu_buffer.free dev_b;
        Gpu_buffer.free dev_c;
        Kernel_launch.unload module_;
        Ok result)

let run_exn k ~m ~n ~k_ ~a ~b =
  match run k ~m ~n ~k_ ~a ~b with
  | Ok r    -> r
  | Error e -> failwith e
