open Base

type t = {
  shape : Modes.t;
  stride : Modes.t;
}

let make (shape : Modes.t) (stride : Modes.t) : t =
  if Modes.compatible shape stride then
    {shape ; stride}
  else
    invalid_arg "incompatible shape and stride"

let rank (l : t) : int = Modes.rank l.shape
let size (l : t) : int = Modes.size l.shape

let cosize (l : t) : int =
  let shapes = Modes.flatten l.shape in
  let strides = Modes.flatten l.stride in
  let total_offset =
    List.fold2_exn shapes strides ~init:0 ~f:(fun acc s st ->
      acc + ((s - 1) * st)
    )
  in

  total_offset + 1

let idx (l : t) (i : Modes.t) : int =
  let strides = Modes.flatten l.stride in
  let elem_index = Modes.flatten i in
  List.map2_exn elem_index strides ~f:(fun c s -> c * s)
  |> List.fold ~init:0 ~f:(+)

let pp (fmt : Stdlib.Format.formatter) (l : t) : unit =
  Stdlib.Format.fprintf fmt "@[<hov>%a:@,%a@]"
    Modes.pp l.shape
    Modes.pp l.stride
