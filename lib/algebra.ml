open Base

let cosize (l : Layout.t) : int =
  Layout.cosize l

let sort (l : Layout.t) : Layout.t =
  let shapes  = Modes.flatten l.shape  in   (* [N0; N1; ...; N_alpha] *)
  let strides = Modes.flatten l.stride in   (* [d0; d1; ...; d_alpha] *)

  let pairs = List.zip_exn shapes strides in
  let sorted = List.sort pairs ~compare:(fun (_,d1) (_,d2) -> Int.compare d1 d2) in
  let sorted_shapes  = List.map sorted ~f:fst in
  let sorted_strides = List.map sorted ~f:snd in
  let rebuilt_shape = Modes.Tuple (List.map sorted_shapes ~f:(fun n -> Modes.Int n)) in
  let rebuilt_stride = Modes.Tuple (List.map sorted_strides ~f:(fun n -> Modes.Int n)) in
    {Layout.shape = rebuilt_shape ; Layout.stride = rebuilt_stride}

let coalesce (l : Layout.t) : Layout.t =
  let shapes = Modes.flatten l.shape in
  let strides = Modes.flatten l.stride in

  let rec walk ss st =
    match ss, st with
    | 1 :: ns, _ :: ds -> walk ns ds
    | n0 :: n1 :: ns, d0 :: d1 :: ds when d1 = n0 * d0 ->
        walk ((n0 * n1) :: ns) (d0 :: ds)

    | n :: ns, d :: ds ->
        let (next_n, next_d) = walk ns ds in
        (n :: next_n, d :: next_d)

    | _ -> (ss, st)
  in

  let (final_shapes, final_strides) = walk shapes strides in
  let rebuilt_shape = Modes.Tuple (List.map final_shapes ~f:(fun n -> Modes.Int n)) in
  let rebuilt_stride = Modes.Tuple (List.map final_strides ~f:(fun d -> Modes.Int d)) in

  { Layout.shape = rebuilt_shape; Layout.stride = rebuilt_stride }


let is_admissible (l : Layout.t) (m : int) : bool =
  let shapes = Modes.flatten l.shape in
  let strides = Modes.flatten l.stride in

  let is_sorted = List.is_sorted strides ~compare:Int.compare in

  if not is_sorted then false
  else
    let rec check_divisibility ss ds =
      match ss, ds with
      | n_i :: n_next :: ss_rest, d_i :: d_next :: ds_rest ->
          if d_next % (n_i * d_i) = 0 then
            check_divisibility (n_next :: ss_rest) (d_next :: ds_rest)
          else
            false

      | [n_last], [d_last] ->
          m % (n_last * d_last) = 0

      | [], [] -> true
      | _ -> false
    in
    check_divisibility shapes strides


let complement (l : Layout.t) (m : int) : Layout.t =
  if m <= 0 || not (is_admissible l m) then
    invalid_arg "layout is not admissible for complementation";

  let ss = Modes.flatten l.shape in
  let ds = Modes.flatten l.stride in

  let rec build_comp shapes strides last_pos =
    match shapes, strides with
    | n_i :: ss_rest, d_i :: ds_rest ->
        let comp_shape = d_i / last_pos in
        let comp_stride = last_pos in
        let (tail_s, tail_d) = build_comp ss_rest ds_rest (n_i * d_i) in
        (comp_shape :: tail_s, comp_stride :: tail_d)

    | [], [] ->
        ([m / last_pos], [last_pos])

    | _ -> ([], [])
  in

  let comp_shapes, comp_strides = build_comp ss ds 1 in

  let filtered = List.zip_exn comp_shapes comp_strides in
  let final_s, final_d = List.unzip filtered in

  Layout.{
    shape = Modes.Tuple (List.map final_s ~f:(fun x -> Modes.Int x));
    stride = Modes.Tuple (List.map final_d ~f:(fun x -> Modes.Int x));
  }

let flat_divide (layout : Layout.t) (divisor : int) : Layout.t =
  let flat_l = coalesce layout in

  let s0 = divisor in
  let d0 = (List.hd_exn (Modes.flatten flat_l.stride)) in

  if Layout.size flat_l % divisor <> 0 then
    invalid_arg "divisor does not divide layout size";

  let s1 = Layout.size flat_l / divisor in
  let d1 = divisor * d0 in

  { shape  = Tuple [Int s0; Int s1];
    stride = Tuple [Int d0; Int d1] }


let logical_divide (layout : Layout.t) (tile : Layout.t) : Layout.t =
  if not (List.for_all2_exn
    (Modes.flatten layout.shape)
    (Modes.flatten tile.shape)
    ~f:(fun ls ts -> ls % ts = 0)) then
    invalid_arg "tile does not divide layout";
  Compose.tile layout tile.shape

let zipped_divide (layout : Layout.t) (tile : Layout.t) : Layout.t =
  let ld = logical_divide layout tile in
  match ld.Layout.shape, ld.Layout.stride with
  | Modes.Tuple [within_s; across_s], Modes.Tuple [within_d; across_d] ->
    let zip m1 m2 =
      match m1, m2 with
      | Modes.Tuple ts1, Modes.Tuple ts2 ->
        Modes.Tuple (List.map2_exn ts1 ts2 ~f:(fun a b -> Modes.Tuple [a; b]))
      | Modes.Int _, Modes.Int _ -> Modes.Tuple [m1; m2]
      | _ -> failwith "rank mismatch"
    in
    { Layout.shape  = zip within_s across_s;
      Layout.stride = zip within_d across_d }
  | _ -> failwith "unexpected logical_divide structure"
