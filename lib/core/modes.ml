open Base

type t =
  | Int of int
  | Tuple of t list

let rec size = function
  | Int n -> n
  | Tuple elts -> List.fold elts ~init:1 ~f:(fun acc x -> acc * size x)

let rec depth = function
  | Int _ -> 0
  | Tuple elts ->
      1 + (List.map elts ~f:depth
           |> List.fold ~init:0 ~f:Int.max)

let rank (m : t) : int =
  match m with
  | Int _ -> 1
  | Tuple elts ->
    List.length elts

let rec flatten (m : t) : int list =
  match m with
  | Int n -> [n]
  | Tuple elts ->
      List.concat_map elts ~f:flatten

let rec compatible (shape : t) (stride : t) : bool =
  match shape , stride with
  | Int _ , Int _ -> true
  | Tuple s_elts , Tuple t_elts ->
    if List.length s_elts <> List.length t_elts then
      false
    else
      List.for_all2_exn s_elts t_elts ~f:compatible

  | _ -> false

let rec pp fmt = function
  | Int n ->
      Stdlib.Format.fprintf fmt "%d" n
  | Tuple elts ->
      Stdlib.Format.fprintf fmt "@[<hv 1>(";
      let rec pp_list = function
        | [] -> ()
        | [x] -> pp fmt x
        | x :: xs ->
            Stdlib.Format.fprintf fmt "%a,@ " pp x;
            pp_list xs
      in
      pp_list elts;
      Stdlib.Format.fprintf fmt ")@]"
