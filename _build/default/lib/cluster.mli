(** Thread block cluster descriptor for Blackwell 2SM MMA.

    Clusters group CTAs onto neighbouring SMs and enable:
    - Distributed Shared Memory (DSM) across CTAs
    - 2-CTA MMA via tcgen05.mma with cta_group=2
    - Cluster-level mbarrier synchronization
    - Cluster Launch Control (CLC) for dynamic scheduling *)

(** Cluster dimensions — how many CTAs per cluster in each axis. *)
type dims = {
  x : int;
  y : int;
  z : int;
}

(** Warp role in a warp-specialized kernel. *)
type warp_role =
  | Producer    (** Issues TMA loads into smem                    *)
  | Consumer    (** Issues tcgen05.mma instructions               *)
  | Epilogue    (** Reads TMEM, writes to smem, issues TMA stores *)
  | Scheduler   (** Issues CLC try_cancel, feeds tile ids         *)

(** A cluster descriptor. *)
type t = {
  dims         : dims;
  num_warps    : int;   (** total warps per CTA                   *)
  warp_roles   : (int * warp_role) list;
    (** list of (warp_id, role) assignments                       *)
}

(** [make dims num_warps warp_roles] constructs a cluster descriptor.
    Raises [Invalid_argument] if:
    - dims.x * dims.y * dims.z > 8 (hardware limit)
    - any warp_id >= num_warps
    - num_warps = 0 *)
val make : dims -> int -> (int * warp_role) list -> t

(** [cta_count t] returns total CTAs per cluster = x * y * z. *)
val cta_count : t -> int

(** [is_2sm t] returns true iff this is a 2-CTA cluster (dims.x=2, y=z=1). *)
val is_2sm : t -> bool

(** [warp_role_of t warp_id] returns the role of the given warp.
    Returns None if warp_id has no assigned role. *)
val warp_role_of : t -> int -> warp_role option

(** [producer_warp t] returns the warp_id of the producer warp. *)
val producer_warp : t -> int option

(** [consumer_warp t] returns the warp_id of the consumer warp. *)
val consumer_warp : t -> int option

(** [epilogue_warps t] returns all warp_ids with Epilogue role. *)
val epilogue_warps : t -> int list

(** [scheduler_warp t] returns the warp_id of the scheduler warp. *)
val scheduler_warp : t -> int option

(** [thread_count t] returns total threads per CTA = num_warps * 32. *)
val thread_count : t -> int

(** PTX emission *)

(** [cluster_arrive_ptx ()] emits cluster barrier arrive.
    "barrier.cluster.arrive.release.aligned;" *)
val cluster_arrive_ptx : unit -> string

(** [cluster_wait_ptx ()] emits cluster barrier wait.
    "barrier.cluster.wait.acquire.aligned;" *)
val cluster_wait_ptx : unit -> string

(** [cluster_ctaid_ptx reg] emits cluster CTA id read.
    "mov.u32 {reg}, %cluster_ctaid.x;" *)
val cluster_ctaid_ptx : string -> string

(** [mapa_ptx dst src cta_rank] emits shared memory address mapping
    across CTAs in the cluster.
    "mapa.shared::cluster.u32 {dst}, {src}, {cta_rank};" *)
val mapa_ptx : string -> string -> string -> string

(** [mbarrier_arrive_expect_cluster_ptx mbar_var expect_bytes] emits
    cluster-scoped mbarrier arrive with tx count.
    "mbarrier.arrive.expect_tx.shared::cluster.b64 [mbar_var], expect_bytes;" *)
val mbarrier_arrive_expect_cluster_ptx : string -> int -> string

(** [emit_cluster_attr t] emits the kernel attribute for cluster dims.
    "__cluster_dims__({x}, {y}, {z})" *)
val emit_cluster_attr : t -> string

(** [emit_smem_mbar var_name count] emits shared memory mbarrier declarations.
    "__shared__ __align__(8) uint64_t {var_name}[{count}];" *)
val emit_smem_mbar : string -> int -> string

(** [pp fmt t] pretty-prints the cluster descriptor. *)
val pp : Stdlib.Format.formatter -> t -> unit
