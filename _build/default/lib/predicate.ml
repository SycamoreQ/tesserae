open Base

type t = {
  layout : Layout.t;
  bounds : int list;
}

let make (layout : Layout.t) (bounds : int list) : t =
  if List.length bounds <> Layout.rank layout then
    invalid_arg "bounds length must match layout rank";

  {layout ; bounds }


let is_in_bounds (p : t) (coord : int list) : bool =
  if List.length coord <> Layout.rank p.layout then
    invalid_arg "bounds length must match layout rank";
  List.for_all2_exn coord p.bounds ~f:(fun c b -> c < b)

let count_valid (p : t) : int =
  let shapes = Modes.flatten p.layout.Layout.shape in
  let n = Layout.size p.layout in
  let count = ref 0 in
  for k = 0 to n - 1 do
    let (_, coords) =
      List.fold_left shapes ~init:(k, []) ~f:(fun (rem, acc) sh ->
        (rem / sh, acc @ [rem % sh]))
    in
    if is_in_bounds p coords then
      Int.incr count
  done;
  !count

let needs_predication (p : t) : bool =
  let shapes = Modes.flatten p.layout.Layout.shape in
  List.exists2_exn p.bounds shapes ~f:(fun b s -> b % s <> 0)

let residue (p : t) : int list =
  let shapes = Modes.flatten p.layout.Layout.shape in
  List.map2_exn p.bounds shapes ~f:(fun b s -> b % s)


let emit_predicate_check (p : t) (coord_var : string) : string =
  List.mapi p.bounds ~f:(fun i b ->
    Printf.sprintf "get<%d>(%s) < %d" i coord_var b)
  |> String.concat ~sep:" && "

let emit_identity_tensor (var_name : string) (layout : Layout.t) : string =
  Printf.sprintf "auto %s = make_identity_tensor(%s{});"
    var_name
    (Codegen.emit_layout layout)

let pp (fmt : Stdlib.Format.formatter) (p : t) : unit =
  let bounds_str = List.map p.bounds ~f:Int.to_string
                    |> String.concat ~sep:", " in
  Stdlib.Format.fprintf fmt "Predicate(layout=%a bounds=[%s] needs_pred=%b)"
    Layout.pp p.layout
    bounds_str
    (needs_predication p)
