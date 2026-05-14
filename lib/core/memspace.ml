open Base

type global
type shared
type register
type tensor

type _ space =
  | Global : global space
  | Shared : shared space
  | Register : register space
  | Tensor : tensor space

let name : type a. a space -> string = fun s ->
  match s with
  | Global -> "global"
  | Shared -> "shared"
  | Register -> "register"
  | Tensor -> "tensor"

let pp (fmt : Stdlib.Format.formatter) (s : _ space) : unit =
  Stdlib.Format.fprintf fmt "%s" (name s)


let can_transfer : type a b. src:a space -> dst:b space -> bool =
  fun ~src ~dst ->
    match src, dst with
    | Global, Global -> true
    | Shared, Shared -> true
    | Register, Register -> true
    | Tensor, Tensor -> true

    | Global, Shared   -> true  (* cp.async / bulk copy *)
    | Global, Register -> true  (* ld.global *)
    | Shared, Register -> true  (* ld.shared *)

    | Register, Shared -> true  (* st.shared *)
    | Register, Global -> true  (* st.global *)

    | Shared, Global -> false

    | _, _ -> false
