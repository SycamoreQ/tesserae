(** Kernel AST — the user-facing tile-level IR.
    Machine-independent. No arch-specific details here.
    The user constructs these expressions; the compiler lowers them. *)

(** Target architecture. *)
type arch =
  | SM80   (** Ampere*)
  | SM90   (** Hopper*)
  | SM100  (** Blackwell *)

(** Element type tags for the DSL. *)
type elem =
  | F16
  | BF16
  | F32
  | S8
  | S32

(** Memory space tags. *)
type space =
  | Global
  | Shared
  | Register
  | TensorMem

(** A tensor expression — the fundamental value in the DSL. *)
type tensor_expr =
  | Arg       of string * elem * space
      (** A kernel argument — name, element type, memory space *)
  | Tile      of tensor_expr * tile_shape
      (** A tiled view of a tensor *)
  | LocalTile of tensor_expr * tile_shape
      (** A per-thread local tile *)
  | Smem      of string * elem * tile_shape
      (** A shared memory buffer — name, type, shape *)

and tile_shape = {
  m : int;
  n : int;
  k : int;
}

(** A statement in the kernel body. *)
type stmt =
  | Load    of tensor_expr * tensor_expr * mask option
      (** Load src into dst with optional predicate mask *)
  | Store   of tensor_expr * tensor_expr * mask option
      (** Store src to dst with optional predicate mask *)
  | Mma     of tensor_expr * tensor_expr * tensor_expr
      (** MMA: accumulate A * B into C *)
  | Pipeline of pipeline_desc * stmt list
      (** Software-pipelined loop over K *)
  | Barrier  of barrier_kind
      (** Synchronization barrier *)
  | For      of string * int * int * stmt list
      (** for loop: var from lo to hi *)
  | If       of pred_expr * stmt list * stmt list
      (** conditional *)
  | Seq      of stmt list
      (** sequence of statements *)

and pipeline_desc = {
  stages  : int;
  k_iters : string;  (** variable name for K / BK *)
}

and mask = {
  coord_var : string;
  bounds    : int list;
}

and barrier_kind =
  | MbarFull   of string  (** wait on full mbarrier [var] *)
  | MbarEmpty  of string  (** wait on empty mbarrier [var] *)
  | ClusterSync           (** barrier.cluster.arrive + wait *)
  | ThreadSync            (** __syncthreads() *)

and pred_expr =
  | WarpIs    of int           (** warp_id == n *)
  | WarpIn    of int list      (** warp_id in [n...] *)
  | InBounds  of string * int list  (** coordinate in bounds *)

(** A complete kernel. *)
type kernel = {
  name    : string;
  arch    : arch;
  elem    : elem;
  tile    : tile_shape;
  stages  : int;
  args    : (string * elem * space) list;
  body    : stmt;
}

(** [kernel name arch elem tile stages args body] constructs a kernel. *)
val make :
  name:string ->
  arch:arch ->
  elem:elem ->
  tile:tile_shape ->
  stages:int ->
  args:(string * elem * space) list ->
  body:stmt ->
  kernel


(** [arg name elem space] declares a kernel argument. *)
val arg : string -> elem -> space -> tensor_expr

(** [smem name elem m k] declares a shared memory buffer. *)
val smem : string -> elem -> int -> int -> tensor_expr

(** [load ~src ~dst ?mask ()] emits a load statement. *)
val load : src:tensor_expr -> dst:tensor_expr -> ?mask:mask -> unit -> stmt

(** [store ~src ~dst ?mask ()] emits a store statement. *)
val store : src:tensor_expr -> dst:tensor_expr -> ?mask:mask -> unit -> stmt

(** [mma a b c] emits an MMA statement. *)
val mma : tensor_expr -> tensor_expr -> tensor_expr -> stmt

(** [pipeline ~stages ~k body] wraps statements in a software pipeline. *)
val pipeline : stages:int -> k:string -> stmt list -> stmt

(** [syncthreads ()] emits a thread barrier. *)
val syncthreads : unit -> stmt

(** [warp_dispatch cases] emits warp-role dispatch.
    cases is a list of (pred_expr, stmt list) pairs. *)
val warp_dispatch : (pred_expr * stmt list) list -> stmt

(** [pp fmt k] pretty-prints the kernel AST. *)
val pp : Stdlib.Format.formatter -> kernel -> unit

(** [pp_stmt fmt s] pretty-prints a statement. *)
val pp_stmt : Stdlib.Format.formatter -> stmt -> unit
