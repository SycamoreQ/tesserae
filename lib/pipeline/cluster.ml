open Base

type dims = {
  x : int;
  y : int;
  z : int;
}

type warp_role =
  | Producer
  | Consumer
  | Epilogue
  | Scheduler

type t = {
  dims       : dims;
  num_warps  : int;
  warp_roles : (int * warp_role) list;
}

let make (dims : dims) (num_warps : int) (warp_roles : (int * warp_role) list) : t =
  if (dims.x * dims.y * dims.z) > 8 then
    invalid_arg "cluster CTA count must be <= 8"

  else if List.exists warp_roles ~f:(fun (id, _) -> id >= num_warps) then
    invalid_arg "warp_id out of range"

  else if num_warps <= 0 then
    invalid_arg "num_warps must be greater than 0"

  else
    { dims; num_warps; warp_roles }


let cta_count (t : t) : int =
  t.dims.x * t.dims.y * t.dims.z

let is_2sm (t : t) : bool =
  match t.dims.x , t.dims.y , t.dims.z with
    | (2 , 1 , 1) -> true
    | (_ , _ , _ ) -> false

let warp_role_of (t : t) (warp_id : int) : warp_role option =
  match List.Assoc.find t.warp_roles ~equal:Int.equal warp_id with
  | Some Producer -> Some Producer
  | Some Consumer -> Some Consumer
  | Some other -> Some other  (* Catch-all for Epilogue/Scheduler *)
  | None -> None


let producer_warp (t : t) : int option =
  match List.find t.warp_roles ~f:(fun (_id, role) ->
    match role with Producer -> true | _ -> false) with
  | Some (id, _) -> Some id
  | None -> None


let consumer_warp (t : t) : int option =
  match List.find t.warp_roles ~f:(fun (_id, role) ->
    match role with Consumer -> true | _ -> false) with
  | Some (id, _) -> Some id
  | None -> None

let epilogue_warps (t : t) : int list =
  List.filter_map t.warp_roles ~f:(fun (id, role) ->
    match role with
    | Epilogue -> Some id
    | _ -> None)

let scheduler_warp (t : t) : int option =
  match List.find t.warp_roles ~f:(fun (_id, role) ->
    match role with Scheduler -> true | _ -> false) with
  | Some (id, _) -> Some id
  | None -> None

let thread_count (t : t) : int =
  t.num_warps * 32

let cluster_arrive_ptx () : string =
  "barrier.cluster.arrive.release.aligned;"

let cluster_wait_ptx () : string =
  "barrier.cluster.wait.acquire.aligned;"

let cluster_ctaid_ptx (reg : string) : string =
  Printf.sprintf "mov.u32 %s, %%cluster_ctaid.x;" reg

let mapa_ptx (dst : string) (src : string) (cta_rank : string) : string =
  Printf.sprintf "mapa.shared::cluster.u32 %s, %s, %s;" dst src cta_rank

let mbarrier_arrive_expect_cluster_ptx (mbar_var : string) (expect_bytes : int) : string =
  Printf.sprintf "mbarrier.arrive.expect_tx.shared::cluster.b64 [%s], %d;" mbar_var expect_bytes

let emit_cluster_attr (t : t) : string =
  Printf.sprintf "__cluster_dims__(%d, %d, %d)" t.dims.x t.dims.y t.dims.z

let emit_smem_mbar (var_name : string) (count : int) : string =
  Printf.sprintf "__shared__ __align__(8) uint64_t %s[%d];" var_name count

let pp (fmt : Stdlib.Format.formatter) (t : t) : unit =
  let open Stdlib.Format in
  fprintf fmt "@[<v 2>Cluster Configuration:@,";
  fprintf fmt "Dims: (%d, %d, %d)@," t.dims.x t.dims.y t.dims.z;
  fprintf fmt "Max Warps: %d@," t.num_warps;
  fprintf fmt "Warp Roles: [@[%a@]]@]"
    (pp_print_list ~pp_sep:(fun f () -> fprintf f "; ")
       (fun f (id, role) ->
          let role_str = match role with
            | Producer -> "Producer" | Consumer -> "Consumer"
            | Epilogue -> "Epilogue" | Scheduler -> "Scheduler"
          in
          fprintf f "%d: %s" id role_str))
    t.warp_roles;
  fprintf fmt "@]"
