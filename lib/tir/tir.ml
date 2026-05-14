open Base
open Tesserae_core
open Tesserae_atoms
open Tesserae_pipeline
open Tesserae_kernel

type packed_tensor = Tensor : ('e, 's) tensor -> packed_tensor

module Type_id = struct
  type _ witness = ..

  module type ID = sig
    type t

    type _ witness += Id : t witness
  end

  type 'a t = (module ID with type t = 'a)

  type (_, _) eq = Refl : ('a, 'a) eq

  let create (type a) () : a t =
    let module M = struct
      type t = a

      type _ witness += Id : t witness
    end in
    (module M)

  let equal (type a b) (id_a : a t) (id_b : b t) : (a, b) eq option =
    let module A = (val id_a : ID with type t = a) in
    let module B = (val id_b : ID with type t = b) in
    match A.Id with B.Id -> Some Refl | _ -> None
end

type _ scalar_ty =
  | U8  : int scalar_ty
  | U32 : int32 scalar_ty
  | S32 : int32 scalar_ty
  | U64 : int64 scalar_ty
  | F32 : float scalar_ty
  | BF16 : float scalar_ty
  | Bool : bool scalar_ty
  | Ptr : 'a scalar_ty

type ('elem  , 'space) tensor = {
  tensor_name: string;
  tensor_id: 'elem Type_id.t;
  tensor_elem_type: 'elem Elemtype.t;
  tensor_memspace: 'space Memspace.space;
  tensor_layout: Layout.t;
  tensor_swizzle: Swizzle.t;
}

type var = {
  var_name : string;
  var_id : int;
  var_type : scalar_ty;
  var_mutable : bool;
}

type const =
  | CInt32 of int32
  | CInt64 of int64
  | CFloat32 of float
  | CFloat64 of float
  | CBool of bool
  | CUnit of unit


type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Mod
  | Eq
  | Ne
  | Lt
  | Le
  | Gt
  | Ge
  | And
  | Or
  | Shl
  | Shr
  | BitAnd
  | BitOr
  | BitXor

type unop = Neg | Not | BitNot

type for_dir = Upto | Downto

type gpu_builtin =
  | ThreadIdx of axis
  | BlockIdx  of axis
  | ClusterCtaId
  | WarpId
  | LaneId

and axis = X | Y | Z

type addr_conv_kind =
  | __cvta_generic_to_global
  | __cvta_generic_to_shared
  | __cvta_generic_to_local
  | __cvta_shared_to_generic
  | __cvta_global_to_generic
  | __cvta_local_to_generic
  | __cvta_to_shared_cluster
  | __cvta_cluster_to_shared


type barrier =
  | CTASync
  | MBarInit
  | MbarArriveExpect
  | MbarWaitParity
  | MbarArrive
  | ClusterArrive
  | ClusterWait
  | CpAsyncWaitAll
  | CpAsyncCommitGroup
  | TCgen05_wait
  | TCgen05_commit
  | TCgen05_relinq
  | TCgen05_fence

type copy_kind =
  | CpAsync
  | TmaLoad
  | TmaMulticast
  | RegToSmem
  | SmemToReg

type copy = {
  copy_kind  : copy_kind;
  src_tensor : ('elem , 'space) tensor; (* Using names or IDs for simpler AST handling *)
  dst_tensor : ('elem , 'space) tensor;
  pred_exp : bool expr option;
  mbarrier : barrier option;
}

type kind = Sm80Mma | Sm90Wgmma | Sm100Tcgen05

type mma_desc = {
  mma_kind: kind ;
  tensor_a: ref tensor;
  tensor_b: ref tensor;
  tensor_c: ref tensor;
  smem_desc_a: ref Smem_desc.t option;
  smem_desc_b: ref Smem_desc.t option;
  accum_flag: bool ;
}

type ops =
  | Copy : copy -> ops
  | Mma : mma_desc -> ops
  | Barrier : barrier -> ops

  | TmemAlloc : var -> ops
  | TmemDealloc : var  -> ops

  | TmemLoad : {
      dst_vars   : var list;
      src_addr   : 'a expr;
      col_offset : int;
    } -> 'n ops

  | TmemCommit : {
      mbar_var : var;
      cta_mask : int option;
    } -> ops

  | SmemDescInit : {
      desc_var     : var;
      ptr_base     : 'a expr;
      leading_dim  : int;
      stride       : int;
      swizzle_mode : Swizzle.mode;
    } -> ops


type stmt =
  | SAssign of lvalue * expr
  | SSeq of stmt list
  | SIf of expr * stmt * stmt option
  | SWhile of expr * stmt
  | SFor of var * expr * expr * for_dir * stmt
  | SMatch of expr * (pattern * stmt) list
  | SReturn of expr
  | SBarrier
  | SWarpBarrier  (** Warp-level sync (__syncwarp) *)
  | SExpr of expr  (** Side-effecting expression *)
  | SEmpty
  | SLet of var * expr * stmt  (** Let binding: let v = e in body *)
  | SLetMut of var * expr * stmt  (** Mutable let: let v = ref e in body *)
  | SPragma of string list * stmt  (** Pragma hints wrapping a statement *)
  | SMemFence  (** Memory fence (threadfence) *)
  | SBlock of stmt
  | Seq of stmt list


and _ expr =
  | Const : 'a scalar_ty * 'a -> 'a expr
  | Cast : 'a scalar_ty * 'b expr -> 'a expr
  | Var : var -> 'a expr
  | Builtin : gpu_builtin -> int expr
  | Binop : binop * 'a expr * 'a expr -> 'a expr
  | Unop : unop * 'a expr -> 'a expr
  | AddrConv : AddrConv * 'a expr -> int64 expr


and helper_func = {
  hf_name : string;
  hf_params : var list;
  hf_ret_type : scalar_ty ;
  hf_body : stmt;
}


and decl =
  | DParam of
      var * array_info option (* kernel parameter, optional array info *)
  | DLocal of var * expr option (* local variable, optional init *)
  | DShared of
      string * Elemtype.t * expr option (* shared array: name, elem type, size *)

and array_info = {arr_elttype : Elemtype.t; arr_memspace : Memspace.space.t}


type tir = {
  name  : string;
  family : Kernel_desc.family;
  params : param list;
  tensors : (string * packed_tensor) list;
  smem_bytes : int;
  cluster : Cluster.t;
  body  : stmt list;
  helpers : helper_func list;
}
