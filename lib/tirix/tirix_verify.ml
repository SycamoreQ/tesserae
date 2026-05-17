open Base
open Tesserae_core
open Tesserae_atoms
open Tesserae_pipeline
open Tesserae_kernel
open Tirix

let verify (tir : Tirix.tirix) : (unit, string list) Result.t =
  let errors = ref [] in
  let err msg = errors := msg :: !errors in

  let tensor_names =
    let from_tensors =
      List.map tir.tensors ~f:(fun (name, _) -> name) in
    let from_params =
      List.map tir.params ~f:(fun p -> p.param_name) in
    Hashtbl.of_alist_exn (module String)
      (List.map (from_tensors @ from_params) ~f:(fun n -> (n, ())))
  in

  let check_tensor_name name context =
    if not (Hashtbl.mem tensor_names name) then
      err (Printf.sprintf "unknown tensor '%s' in %s" name context)
  in

  let tensor_name (Tensor t) = t.tensor_name in

  let rec check_expr : type a. Hashtbl.M(String).t -> a expr -> unit =
    fun scope e ->
    match e with
    | Const _ -> ()
    | Cast (_, inner) -> check_expr scope inner
    | Var v ->
        if not (Hashtbl.mem scope v.var_name) then
          err (Printf.sprintf "undefined variable '%s'" v.var_name)
    | Builtin _ -> ()
    | Binop (_, l, r) ->
        check_expr scope l;
        check_expr scope r
    | Unop (_, inner) -> check_expr scope inner
    | AddrConv (_, inner) -> check_expr scope inner
  in

  let check_packed_expr scope (Expr e) = check_expr scope e in

  let check_op scope op =
    match op with
    | Copy c ->
        check_tensor_name (tensor_name c.src_tensor) "copy.src";
        check_tensor_name (tensor_name c.dst_tensor) "copy.dst";
        Option.iter c.pred_expr ~f:(check_expr scope);
        Option.iter c.mbar_var  ~f:(fun v ->
          if not (Hashtbl.mem scope v.var_name) then
            err (Printf.sprintf "undefined mbar '%s' in copy" v.var_name))
    | Mma m ->
        check_tensor_name (tensor_name m.tensor_a) "mma.a";
        check_tensor_name (tensor_name m.tensor_b) "mma.b";
        check_tensor_name (tensor_name m.tensor_c) "mma.c"

    | Barrier (MbarInit { mbar; _ }) ->
        if not (Hashtbl.mem scope mbar.var_name) then
          err (Printf.sprintf "undefined mbar '%s' in MbarInit" mbar.var_name)

    | Barrier (MbarArriveExpect { mbar; bytes }) ->
        if not (Hashtbl.mem scope mbar.var_name) then
          err (Printf.sprintf "undefined mbar '%s' in MbarArriveExpect" mbar.var_name);
        check_expr scope bytes

    | Barrier (MbarWaitParity { mbar; phase }) ->
        if not (Hashtbl.mem scope mbar.var_name) then
          err (Printf.sprintf "undefined mbar '%s' in MbarWaitParity" mbar.var_name);
        check_expr scope phase

    | Barrier (MbarArrive { mbar }) ->
        if not (Hashtbl.mem scope mbar.var_name) then
          err (Printf.sprintf "undefined mbar '%s' in MbarArrive" mbar.var_name)

    | Barrier _ -> ()

    | TmemAlloc { addr_var; _ } ->
        if not (Hashtbl.mem scope addr_var.var_name) then
          err (Printf.sprintf "undefined tmem addr '%s'" addr_var.var_name)

    | TmemDealloc { addr_var; _ } ->
        if not (Hashtbl.mem scope addr_var.var_name) then
          err (Printf.sprintf "undefined tmem addr '%s'" addr_var.var_name)

    | TmemLoad { dst_vars; src_addr; _ } ->
        List.iter dst_vars ~f:(fun v ->
          if not (Hashtbl.mem scope v.var_name) then
            err (Printf.sprintf "undefined dst var '%s' in TmemLoad" v.var_name));
        check_expr scope src_addr

    | TmemCommit { mbar_var; _ } ->
        if not (Hashtbl.mem scope mbar_var.var_name) then
          err (Printf.sprintf "undefined mbar '%s' in TmemCommit" mbar_var.var_name)

    | SmemDescInit { desc_var; ptr_expr; _ } ->
        if not (Hashtbl.mem scope desc_var.var_name) then
          err (Printf.sprintf "undefined desc var '%s'" desc_var.var_name);
        check_expr scope ptr_expr
  in

  let rec check_stmt scope stmt =
    match stmt with
    | SLet (v, e) ->
        check_packed_expr scope e;
        Hashtbl.set scope ~key:v.var_name ~data:();
        scope

    | SLetMut (v, e) ->
        check_packed_expr scope e;
        Hashtbl.set scope ~key:v.var_name ~data:();
        scope

    | SAssign (v, e) ->
        if not (Hashtbl.mem scope v.var_name) then
          err (Printf.sprintf "undefined variable '%s' in SAssign" v.var_name)
        else if not v.var_mutable then
          err (Printf.sprintf "assigning to immutable variable '%s'" v.var_name);
        check_packed_expr scope e;
        scope

    | SOp op ->
        check_op scope op;
        scope

    | SIf (cond, thn, els) ->
        check_expr scope cond;
        List.iter thn ~f:(fun s -> ignore (check_stmt scope s));
        List.iter els ~f:(fun s -> ignore (check_stmt scope s));
        scope

    | SFor { var; start; stop; step; body; _ } ->
        check_expr scope start;
        check_expr scope stop;
        (match step with
         | Const (_, v) when Int32.equal v 0l ->
             err (Printf.sprintf "SFor '%s' has zero step" var.var_name)
         | _ -> ());
        let inner_scope = Hashtbl.copy scope in
        Hashtbl.set inner_scope ~key:var.var_name ~data:();
        List.iter body ~f:(fun s -> ignore (check_stmt inner_scope s));
        scope

    | SPipeline { stages; prologue; mainloop; epilogue } ->
        if stages <= 0 then
          err (Printf.sprintf "SPipeline has invalid stage count %d" stages);
        List.iter prologue ~f:(fun s -> ignore (check_stmt scope s));
        List.iter mainloop ~f:(fun s -> ignore (check_stmt scope s));
        List.iter epilogue ~f:(fun s -> ignore (check_stmt scope s));
        scope

    | SWarpGroup (role, body) ->
        let role_exists =
          List.exists tir.cluster.Cluster.warp_roles
            ~f:(fun (_, r) -> Poly.equal r role)
        in
        if not role_exists then
          err (Printf.sprintf "SWarpGroup role not in cluster warp_roles");
        List.iter body ~f:(fun s -> ignore (check_stmt scope s));
        scope

    | SPragma (_, body) ->
        List.iter body ~f:(fun s -> ignore (check_stmt scope s));
        scope

    | SSeq stmts ->
        let _ = List.fold stmts ~init:scope
          ~f:(fun sc s -> check_stmt sc s)
        in
        scope

    | SEmpty -> scope
  in

  let init_scope = Hashtbl.create (module String) in
  let _ = List.fold tir.body ~init:init_scope
    ~f:(fun scope stmt -> check_stmt scope stmt)
  in

  if List.is_empty !errors then Ok ()
  else Error (List.rev !errors)
