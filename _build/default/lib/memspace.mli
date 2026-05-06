(** Phantom types representing CUDA memory spaces.
    These types are never instantiated — they exist only to be used
    as type parameters on tensors. *)

type global
type shared
type register
type tensor

(** A witness GADT that reifies a memory space at the value level.
    This lets functions inspect which space they're dealing with
    at runtime when needed, while still being type-safe. *)
type _ space =
  | Global : global space
  | Shared : shared space
  | Register : register space
  | Tensor : tensor space

(** [name s] returns the CUDA memory space name as a string.
    Used for code generation in Phase 3. *)
val name : _ space -> string

(** [pp fmt s] pretty-prints the space name. *)
val pp : Stdlib.Format.formatter -> _ space -> unit

(** Transfer validity: not all memory transfers are legal in CUDA.
    [can_transfer ~src ~dst] returns true iff a direct copy
    from [src] to [dst] is valid without going through an intermediate.

    Rules:
    - global  → shared  : valid (bulk async copy, cp.async)
    - global  → register: valid (load)
    - shared  → register: valid (load from smem)
    - register → shared : valid (store to smem)
    - register → global : valid (store)
    - shared  → global  : invalid in CuTe (must go through register)
    - same    → same    : valid *)
val can_transfer : src:_ space -> dst:_ space -> bool
