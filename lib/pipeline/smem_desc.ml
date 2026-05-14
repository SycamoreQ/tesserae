open Base
open Tesserae_core

type swizzle_mode =
  | NoSwizzle
  | Swizzle32B
  | Swizzle64B
  | Swizzle128B

type t = {
  base_addr : int;
  leading_off  : int;
  stride_off : int;
  swizzle_mode : swizzle_mode;
}

let swizzle_mode_of (sw : Swizzle.t) : swizzle_mode =
  match sw.Swizzle.b with
  | 0 -> NoSwizzle
  | 1 -> Swizzle32B
  | 2 -> Swizzle64B
  | 3 -> Swizzle128B
  | _ -> Swizzle128B

let make ~(base_addr : int) ~(leading_off : int) ~(stride_off : int)
    ~(swizzle_mode : swizzle_mode) : t =
  {
    base_addr ; leading_off ; stride_off ; swizzle_mode
  }

let swizzle_mode_bits (m : swizzle_mode) : int =
  match m with
  | NoSwizzle -> 0
  | Swizzle32B -> 1
  | Swizzle64B -> 2
  | Swizzle128B -> 3

let encode (d : t) : int =
  let s_bits = swizzle_mode_bits d.swizzle_mode in
  d.base_addr
  lor (d.leading_off lsl 16)
  lor (d.stride_off lsl 32)
  lor (s_bits lsl 61)


let emit_make_smem_desc (ptr_var : string) (leading : int) (stride : int) (sw : swizzle_mode) : string =
  let sw_bits = swizzle_mode_bits sw in
  Printf.sprintf "uint64_t desc = make_smem_desc(%s, %d, %d, %d);"
    ptr_var leading stride sw_bits

let emit_cpp_helper () : string =
  "__device__ __forceinline__ uint64_t make_smem_desc(void* ptr, int leading, int stride, int sw) {\n\
  \  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));\n\
  \  uint64_t desc = ((uint64_t)addr & 0xFFFF) |\n\
  \                 (((uint64_t)leading & 0xFFFF) << 16) |\n\
  \                 (((uint64_t)stride & 0x1FFFFFFF) << 32) |\n\
  \                 (((uint64_t)sw & 0x3) << 61);\n\
  \  return desc;\n\
  }"

let pp (fmt : Stdlib.Format.formatter) (d : t) : unit =
  let open Stdlib.Format in
  let sw_str = match d.swizzle_mode with
    | NoSwizzle -> "None" | Swizzle32B -> "32B"
    | Swizzle64B -> "64B" | Swizzle128B -> "128B"
  in
  fprintf fmt "@[<v 2>Smem Descriptor:@,";
  fprintf fmt "Base Addr: 0x%x@," d.base_addr;
  fprintf fmt "Leading:   %d@," d.leading_off;
  fprintf fmt "Stride:    %d@," d.stride_off;
  fprintf fmt "Swizzle:   %s@]" sw_str
