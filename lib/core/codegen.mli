(** Emit CuTe C++ type strings and declarations from Tesserae descriptors. *)

(** [emit_int_mode m] emits a CuTe compile-time integer or tuple.
    Int 4       → "_4"
    Int 1       → "_1"
    Tuple [4;2] → "Shape<_4,_2>"  (used for both shape and stride contexts) *)
val emit_int_mode : Modes.t -> string

(** [emit_shape m] emits a CuTe shape type.
    Int 4            → "_4"
    Tuple [Int 4; Int 2] → "Shape<_4,_2>"
    Nested tuples    → "Shape<_4,Shape<_2,_2>>" *)
val emit_shape : Modes.t -> string

(** [emit_stride m] emits a CuTe stride type.
    Same structure as shape but uses "Stride" as the wrapper.
    Int 1                → "_1"
    Tuple [Int 1; Int 4] → "Stride<_1,_4>" *)
val emit_stride : Modes.t -> string

(** [emit_layout l] emits a full CuTe Layout type string.
    lay (2,3):(1,2) → "Layout<Shape<_2,_3>,Stride<_1,_2>>" *)
val emit_layout : Layout.t -> string

(** [emit_tensor_type elem space layout] emits a CuTe Tensor type.
    For global float32 with layout (2,3):(1,2):
    "Tensor<float*, Layout<Shape<_2,_3>,Stride<_1,_2>>>"
    For shared __half:
    "Tensor<__half*, Layout<...>>" *)
val emit_tensor_type : _ Elemtype.t -> _ Memspace.space -> Layout.t -> string

(** [emit_make_tensor ptr_name elem space layout] emits a make_tensor call.
    "make_tensor<float>({ptr_name}, Layout<Shape<_2,_3>,Stride<_1,_2>>{})" *)
val emit_make_tensor : string -> _ Elemtype.t -> _ Memspace.space -> Layout.t -> string

(** [emit_smem_decl var_name elem layout] emits a shared memory array declaration.
    "__shared__ float {var_name}[{size}];"
    where size = Layout.size layout *)
val emit_smem_decl : string -> _ Elemtype.t -> Layout.t -> string

(** [emit_include_guard name] emits a header include guard.
    "#pragma once" *)
val emit_include_guard : unit -> string

(** [emit_cute_includes] emits the standard CuTe include block as a string. *)
val emit_cute_includes : unit -> string
