(** Lower — compiles a Kernel_ast.kernel into a Kernel_desc.t.

    This is the heart of the Tesserae compiler. It:
    - Infers the right MMA atom from element types + arch
    - Selects TMA vs cp.async based on arch
    - Assigns warp roles (producer/consumer/epilogue/scheduler)
    - Selects swizzle based on tile dimensions and element type
    - Builds the full Kernel_desc.t

    The user never touches Kernel_desc directly —
    they write Kernel_ast and Lower produces the descriptor. *)

(** Lowering errors. *)
type error =
  | UnsupportedArch    of string
  | UnsupportedElem    of string
  | IncompatibleTile   of string
  | MissingArg         of string
  | InvalidPipeline    of string

(** [lower k] lowers a kernel AST to a descriptor.
    Returns [Ok desc] or [Error e].

    Arch → copy strategy:
    - SM80  → CpAsync
    - SM90  → TmaLoad
    - SM100 → TmaMulticast

    Arch + elem → MMA atom:
    - SM80  + F16/BF16 → sm80_16x8x16_f32f16f16f32
    - SM80  + S8       → sm80_16x8x32_s32s8s8s32
    - SM90  + F16      → sm90_64x64x16_f32f16f16f32
    - SM90  + BF16     → sm90_64x64x16_f32bf16bf16f32
    - SM100 + F16/BF16 → sm100_128x128x16_f32f16f16f32
    - SM100 + S8       → sm100_64x64x32_s32s8s8s32

    Arch → accumulator location:
    - SM80/SM90  → Registers
    - SM100      → TensorMem

    Arch → cluster:
    - SM80/SM90  → single SM, 4 warps
    - SM100      → 2SM, 6 warps with Scheduler *)


module type Elem_witness = sig
  type t
  val witness : t Elemtype.t
end

type packed = Pack : (_, _, _, _, _, _) Kernel_desc.t -> packed

val lower : Kernel_ast.kernel -> (packed, error) Result.t

(** [lower_exn k] like [lower] but raises on error. *)
val lower_exn : Kernel_ast.kernel -> packed

(** [elem_to_elemtype e] converts a DSL elem tag to an Elemtype.t witness. *)
val elem_to_elemtype : Kernel_ast.elem -> (module Elem_witness)

(** [arch_to_strategy a] returns the copy strategy for an arch. *)
val arch_to_strategy : Kernel_ast.arch -> Tile_io.strategy

(** [arch_to_accum a] returns the accumulator location for an arch. *)
val arch_to_accum : Kernel_ast.arch -> Tile_op.accum_loc

(** [infer_m k] extracts the M problem dimension from kernel args.
    Looks for an arg named "M" or infers from tile shape. *)
val infer_m : Kernel_ast.kernel -> int

(** [infer_n k] extracts the N problem dimension. *)
val infer_n : Kernel_ast.kernel -> int

(** [infer_k k] extracts the K problem dimension. *)
val infer_k : Kernel_ast.kernel -> int

(** [pp_error fmt e] pretty-prints a lowering error. *)
val pp_error : Stdlib.Format.formatter -> error -> unit
