open Base

type sm80
type sm90
type sm100

type _ arch =
  | SM80  : sm80  arch
  | SM90  : sm90  arch
  | SM100 : sm100 arch

type major =
  | RowMajor
  | ColMajor

type ('arch, 'a, 'b, 'c, 'd) t = {
  arch    : 'arch arch;
  m       : int;
  n       : int;
  k       : int;
  a_type  : 'a Elemtype.t;
  b_type  : 'b Elemtype.t;
  c_type  : 'c Elemtype.t;
  d_type  : 'd Elemtype.t;
  a_major : major;
  b_major : major;
}

(* --- Ampere --- *)

let sm80_16x8x8_f32f16f16f32 a_major b_major =
  { arch = SM80; m = 16; n = 8; k = 8
  ; a_type = Elemtype.Float16
  ; b_type = Elemtype.Float16
  ; c_type = Elemtype.Float32
  ; d_type = Elemtype.Float32
  ; a_major; b_major }

let sm80_16x8x16_f32f16f16f32 a_major b_major =
  { arch = SM80; m = 16; n = 8; k = 16
  ; a_type = Elemtype.Float16
  ; b_type = Elemtype.Float16
  ; c_type = Elemtype.Float32
  ; d_type = Elemtype.Float32
  ; a_major; b_major }

let sm80_16x8x16_f32bf16bf16f32 a_major b_major =
  { arch = SM80; m = 16; n = 8; k = 16
  ; a_type = Elemtype.Bfloat16
  ; b_type = Elemtype.Bfloat16
  ; c_type = Elemtype.Float32
  ; d_type = Elemtype.Float32
  ; a_major; b_major }

let sm80_16x8x32_s32s8s8s32 a_major b_major =
  { arch = SM80; m = 16; n = 8; k = 32
  ; a_type = Elemtype.Int8
  ; b_type = Elemtype.Int8
  ; c_type = Elemtype.Int32
  ; d_type = Elemtype.Int32
  ; a_major; b_major }

(* --- Hopper --- *)

let sm90_64x64x16_f32f16f16f32 a_major b_major =
  { arch = SM90; m = 64; n = 64; k = 16
  ; a_type = Elemtype.Float16
  ; b_type = Elemtype.Float16
  ; c_type = Elemtype.Float32
  ; d_type = Elemtype.Float32
  ; a_major; b_major }

let sm90_64x128x16_f32f16f16f32 a_major b_major =
  { arch = SM90; m = 64; n = 128; k = 16
  ; a_type = Elemtype.Float16
  ; b_type = Elemtype.Float16
  ; c_type = Elemtype.Float32
  ; d_type = Elemtype.Float32
  ; a_major; b_major }

let sm90_64x64x16_f32bf16bf16f32 a_major b_major =
  { arch = SM90; m = 64; n = 64; k = 16
  ; a_type = Elemtype.Bfloat16
  ; b_type = Elemtype.Bfloat16
  ; c_type = Elemtype.Float32
  ; d_type = Elemtype.Float32
  ; a_major; b_major }

(* --- Blackwell --- *)

let sm100_64x64x16_f32f16f16f32 a_major b_major =
  { arch = SM100; m = 64; n = 64; k = 16
  ; a_type = Elemtype.Float16
  ; b_type = Elemtype.Float16
  ; c_type = Elemtype.Float32
  ; d_type = Elemtype.Float32
  ; a_major; b_major }

let sm100_128x128x16_f32f16f16f32 a_major b_major =
  { arch = SM100; m = 128; n = 128; k = 16
  ; a_type = Elemtype.Float16
  ; b_type = Elemtype.Float16
  ; c_type = Elemtype.Float32
  ; d_type = Elemtype.Float32
  ; a_major; b_major }

let sm100_64x64x32_s32s8s8s32 a_major b_major =
  { arch = SM100; m = 64; n = 64; k = 32
  ; a_type = Elemtype.Int8
  ; b_type = Elemtype.Int8
  ; c_type = Elemtype.Int32
  ; d_type = Elemtype.Int32
  ; a_major; b_major }

(* --- queries --- *)

let shape a = (a.m, a.n, a.k)

let thread_count : type arch a b c d. (arch, a, b, c, d) t -> int =
  fun a -> match a.arch with
  | SM80  -> 32
  | SM90  -> 128
  | SM100 -> 128

let is_wgmma : type arch a b c d. (arch, a, b, c, d) t -> bool =
  fun a -> match a.arch with
  | SM80  -> false
  | SM90  -> true
  | SM100 -> true

(* --- codegen --- *)

let arch_string : type arch a b c d. (arch, a, b, c, d) t -> string =
  fun a -> match a.arch with
  | SM80  -> "SM80"
  | SM90  -> "SM90"
  | SM100 -> "SM100"

let elem_string : type e. e Elemtype.t -> string = function
  | Elemtype.Float32  -> "F32"
  | Elemtype.Float16  -> "F16"
  | Elemtype.Bfloat16 -> "BF16"
  | Elemtype.Int8     -> "S8"
  | Elemtype.Int32    -> "S32"

let major_char = function
  | ColMajor -> "T"
  | RowMajor -> "N"

let emit_cpp a =
  Printf.sprintf "%s_%dx%dx%d_%s%s%s%s_%s%s"
    (arch_string a)
    a.m a.n a.k
    (elem_string a.d_type)
    (elem_string a.a_type)
    (elem_string a.b_type)
    (elem_string a.c_type)
    (major_char a.a_major)
    (major_char a.b_major)

let pp fmt a =
  Stdlib.Format.fprintf fmt "%s" (emit_cpp a)
