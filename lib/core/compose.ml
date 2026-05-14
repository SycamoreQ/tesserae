open Base

let compose (outer : Layout.t) (inner : Layout.t) : Layout.t =
  if Layout.cosize inner > Layout.size outer then
    invalid_arg "inner cosize exceeds outer size";
  let inner_shapes  = Modes.flatten inner.shape in
  let outer_strides = Modes.flatten outer.stride in
  let apply_outer (s : int) : int =
    let (_, digits) =
      List.fold_map inner_shapes ~init:s ~f:(fun rem sh ->
        (rem / sh, rem % sh))
    in
    List.fold2_exn digits outer_strides ~init:0 ~f:(fun acc d st ->
      acc + d * st)
  in
  let rec new_stride (m : Modes.t) : Modes.t =
    match m with
    | Modes.Int s -> Modes.Int (apply_outer s)
    | Modes.Tuple elts -> Modes.Tuple (List.map elts ~f:new_stride)
  in
  { Layout.shape = inner.shape; stride = new_stride inner.stride }


let tile (layout : Layout.t) (tile_shape : Modes.t) : Layout.t =
  let l_shapes = Modes.flatten layout.shape in
  let l_strides = Modes.flatten layout.stride in
  let t_shapes = Modes.flatten tile_shape in

  let rec build_tiled (ls, lst, ts) =
    match ls, lst, ts with
    | [], [], [] -> ([], [], [], [])
    | l_s :: ls_r, l_st :: lst_r, t_s :: ts_r ->
      if t_s = 0 || l_s % t_s <> 0 then
        invalid_arg "tile shape does not divide layout shape";

        let (in_s, out_s, in_st, out_st) = build_tiled (ls_r, lst_r, ts_r) in
        ( Modes.Int t_s :: in_s,
          Modes.Int (l_s / t_s) :: out_s,
          Modes.Int l_st :: in_st,
          Modes.Int (l_st * t_s) :: out_st )
    | _ -> invalid_arg "Layout.tile: structural mismatch"
  in

  let (in_s, out_s, in_st, out_st) = build_tiled (l_shapes, l_strides, t_shapes) in

  { shape = Modes.Tuple [Modes.Tuple in_s; Modes.Tuple out_s];
    stride = Modes.Tuple [Modes.Tuple in_st; Modes.Tuple out_st] }
