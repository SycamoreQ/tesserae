open Base

type ('elem, 'space) t = {
  layout : Layout.t;
  space  : 'space Memspace.space;
}

let make (layout : Layout.t) (space : 'space Memspace.space) : ('elem, 'space) t =
  {layout ; space}

let layout (t : ('elem, 'space) t) : Layout.t =
  t.layout

let space (t : ('elem, 'space) t) : 'space Memspace.space =
  t.space

let size (t : ('elem, 'space) t) : int =
  Layout.size t.layout

let rank (t : ('elem, 'space) t) : int =
  Layout.rank t.layout

let local_tile (t : ('elem, 'space) t) (tile_shape : Modes.t) : ('elem, 'space) t =
  let tiled_layout = Compose.tile t.layout tile_shape in
  {
    layout = tiled_layout;
    space  = t.space
  }

let transfer
    ~(src : ('elem, 'src_space) t)
    ~(dst_space : 'dst_space Memspace.space)
  : ('elem, 'dst_space) t =
  let src_space = src.space in

  if Memspace.can_transfer ~src:src_space ~dst:dst_space then
    { layout = src.layout; space = dst_space }
  else
    invalid_arg "invalid transfer between memory spaces"

let pp (fmt : Stdlib.Format.formatter) (t : ('elem, 'space) t) : unit =
  Stdlib.Format.fprintf fmt "@[%a:%a@]"
    Memspace.pp t.space
    Layout.pp t.layout
