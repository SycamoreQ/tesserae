open Base
open Tesserae_core
open Tesserae_pipeline
open Tesserae_kernel

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
  | U8 : int scalar_ty
  | U32 : int32 scalar_ty
  | S32 : int32 scalar_ty
  | U64 : int64 scalar_ty
  | F16 : float scalar_ty
  | F32 : float scalar_ty
  | BF16 : float scalar_ty
  | Bool : bool scalar_ty
  | Ptr : int64 scalar_ty    (* generic pointer as u64 *)

type packed_scalar = Scalar : _ scalar_ty -> packed_scalar


type ('elem, 'space) tensor = {
  tensor_name : string;
  tensor_id  : 'elem Type_id.t;
  tensor_elem_type: 'elem Elemtype.t;
  tensor_memspace : 'space Memspace.space;
  tensor_layout  : Layout.t;
  tensor_swizzle : Swizzle.t;
}

type packed_tensor = Tensor : ('e, 's) tensor -> packed_tensor

type var = {
  var_name : string;
  var_id  : int;
  var_type : packed_scalar;
  var_mutable : bool;
}

type axis = X | Y | Z

type gpu_builtin =
  | ThreadIdx of axis
  | BlockIdx  of axis
  | ClusterCtaId
  | WarpId
  | LaneId

type addr_conv_kind =
  | GenericToShared
  | GenericToGlobal
  | GenericToLocal
  | SharedToGeneric
  | GlobalToGeneric
  | LocalToGeneric
  | ToSharedCluster
  | ClusterToShared

type binop =
  | Add | Sub | Mul | Div | Mod
  | Eq  | Ne  | Lt  | Le  | Gt | Ge
  | And | Or
  | Shl | Shr
  | BitAnd | BitOr | BitXor

type unop = Neg | Not | BitNot

type _ expr =
  | Const : 'a scalar_ty * 'a -> 'a expr
  | Cast : 'a scalar_ty * 'b expr -> 'a expr
  | Var : var -> 'a expr
  | Builtin : gpu_builtin -> int32 expr
  | Binop : binop * 'a expr * 'a expr -> 'a expr
  | Unop : unop * 'a expr -> 'a expr
  | AddrConv: addr_conv_kind * 'a expr -> int64 expr

type packed_expr = Expr : _ expr -> packed_expr

type barrier =
  | CtaSync
  | WarpSync
  | MemFence
  | MbarInit of { mbar : var; count : int }
  | MbarArriveExpect of { mbar : var; bytes : int32 expr }
  | MbarWaitParity of { mbar : var; phase : int32 expr }
  | MbarArrive of { mbar : var }
  | ClusterArrive
  | ClusterWait
  | CpAsyncWaitAll
  | CpAsyncCommitGroup
  | Tcgen05Wait
  | Tcgen05Fence

type copy_kind =
  | CpAsync
  | TmaLoad
  | TmaMulticast
  | RegToSmem
  | SmemToReg

type copy = {
  copy_kind : copy_kind;
  src_tensor : packed_tensor;
  dst_tensor : packed_tensor;
  pred_expr  : bool expr option;
  mbar_var : var option;
}


type mma_kind = Sm80Mma | Sm90Wgmma | Sm100Tcgen05

type mma_desc = {
  mma_kind : mma_kind;
  tensor_a : packed_tensor;
  tensor_b : packed_tensor;
  tensor_c : packed_tensor;
  smem_desc_a : Smem_desc.t option;
  smem_desc_b : Smem_desc.t option;
  accum_flag : bool;
}

type op =
  | Copy of copy
  | Mma  of mma_desc
  | Barrier of barrier
  | TmemAlloc of { addr_var : var; col_count : int }
  | TmemDealloc of { addr_var : var; col_count : int }
  | TmemLoad of { dst_vars : var list; src_addr : int64 expr; col_offset : int }
  | TmemCommit of { mbar_var : var; cta_mask : int option }
  | SmemDescInit of {
      desc_var    : var;
      ptr_expr    : int64 expr;
      leading_dim : int;
      stride      : int;
      swizzle     : Swizzle.t;
    }


type for_dir = Upto | Downto

type stmt =
  | SLet of var * packed_expr
  | SLetMut of var * packed_expr
  | SAssign of var * packed_expr
  | SOp of op
  | SIf of bool expr * stmt list * stmt list
  | SFor of {
      var : var;
      start : int32 expr;
      stop : int32 expr;
      step : int32 expr;
      dir : for_dir;
      unroll : bool;
      body : stmt list;
    }
  | SPipeline of {
      stages : int;
      prologue : stmt list;
      mainloop : stmt list;
      epilogue : stmt list;
    }
  | SWarpGroup of Cluster.warp_role * stmt list
  | SPragma of string * stmt list
  | SSeq of stmt list
  | SEmpty


type helper_func = {
  hf_name      : string;
  hf_params    : var list;
  hf_ret_type  : packed_scalar;
  hf_body      : stmt list;
}

type param = {
  param_name   : string;
  param_tensor : packed_tensor;
  param_is_tma : bool;
}


type tirix = {
  name       : string;
  family     : Kernel_desc.family;
  params     : param list;
  tensors    : (string * packed_tensor) list;
  smem_bytes : int;
  cluster    : Cluster.t;
  body       : stmt list;
  helpers    : helper_func list;
}
