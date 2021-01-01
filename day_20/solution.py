#!/usr/bin/env python

# # #
#
# TODO
# 1) use numpy for Tile, Edge, PuzzleBoard
# 2) represent a Tile as a matrix of zeroes and ones from the very beginning.
# 3) there are some TODOs scattered over the script.
#    Those in _arrange_tiles() can be interesting to implement.

import os
import sys
from typing import List, Tuple, Optional, Union, Callable
import math
from functools import reduce
from collections import defaultdict
from devel import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False

Position = Tuple[int, int]
Shape = Tuple[int, int]


class Tile(object):

    SIDES = ('n', 'e', 's', 'w')  # north, east, south, west

    @classmethod
    def from_lines(cls, lines: List[str]) -> 'Tile':
        lines = [line.strip() for line in lines]
        id = lines.pop(0).strip(':').split()[1]
        obj = cls(lines, int(id))
        return obj

    @classmethod
    def hstack(cls, tile1, tile2, sep=''):
        """concatenate two tiles side by side with given separator"""
        rows = [sep.join(rs) for rs in zip(tile1.rows, tile2.rows)]
        return cls(rows)

    @classmethod
    def clockwise_distance(cls, a: str, b: str) -> int:
        """
        How many cloclwise rotation steps between side <a> and <b>.
        For example:
        * s, s --> 0
        * s, w --> 1
        * s, n --> 2
        * s, e --> 3
        """
        assert a in cls.SIDES, \
            f"Invalid side spec: {a}. Must be one of {cls.SIDES}"
        assert b in cls.SIDES, \
            f"Invalid side spec: {b}. Must be one of {cls.SIDES}"
        stops = list(cls.SIDES * 2)
        ai = stops.index(a)
        bi = stops.index(b, ai)
        return bi - ai

    def __init__(self, rows: List[str], id=None):
        self.id = id
        self._set_rows(rows)
        self.orig_rows = None
        self.debug = False

    @property
    def size(self) -> Shape:
        """Return (height, width) of the tile"""
        return (len(self.rows), len(self.rows[0]))

    def is_square(self) -> bool:
        """Test if the current Tile is square."""
        h, w = self.size
        return h == w and h > 0

    def crop(self, topleft: Position, height: int, width: int) -> 'Tile':
        """Return a new Tile that is built by cropping the current Tile
        starting at the position <topleft> and going down <height> rows and
        going right <width> columns."""
        rows = []
        r0, c0 = topleft
        for ri in range(r0, r0+height):
            row = self.rows[ri][c0:c0+width]
            rows.append(row)
        return self.__class__(rows)

    def count(self, char: Optional[str] = None) -> int:
        """Count how many times given character appears on the Tile."""
        if char is None:
            h, w = self.size
            return h * w
        else:
            return sum(map(lambda row: row.count(char), self.rows))

    def to_onehot(self):
        """Return 2D matrix that represents current Tile such that # is encoded
        with 1 and any other character, for example ., is encoded with 0."""
        chars = {'#': 1, '.': 0}
        return [[chars.get(ch, 0) for ch in list(line)] for line in self.rows]

    # TODO
    # 1. memoize transformations
    # 2. there is no need to transform the whole Tile. Suffice it to transform
    #    edges. However, the whole Tile needs to be transformed when str(tile)
    #    Therefore for now, just transforming the whole tile.

    def hflip(self):
        """Flip the current tile horizontally (swap left and right) in place."""
        if self.debug:
            print("::flipping horizontally")
        rows = [r[::-1] for r in self.rows]
        self._set_rows(rows)
        return self

    def vflip(self):
        """Flip the current time vertically (swap top and bottom) in place."""
        if self.debug:
            print("::flipping vertically")
        rows = list(reversed(self.rows))
        self._set_rows(rows)
        return self

    def rotate(self, n_steps=1):
        """Rotate the current tile clockwise given number of steps, one step
        being equal to 90 degrees."""
        if self.debug:
            print(f"::rotating {n_steps} step(s)")
        assert n_steps > -1, \
            f"Number of rotation steps cannot be negative: {n_steps}"
        n_steps = n_steps % 4
        if n_steps in {1, 3}:
            newrows = []
            for c in range(self.size[1]):
                newrows.append(self._col(c)[::-1])
            self._set_rows(newrows)
            n_steps -= 1
        if n_steps == 2:
            self.hflip().vflip()
        return self

    def transform(self, code: str) -> None:
        """Perform several transformations (flips and/or rotations) encoded as
        one command. For example, rrh means rotate, rotate, flip horizontally.
        """
        for ch in code.lower():
            if ch == '_':   pass
            elif ch == 'r': self.rotate()
            elif ch == 'h': self.hflip()
            elif ch == 'v': self.vflip()
            else:
                raise ValueError(f"Invalid transformation '{ch}' in '{code}'")

    def _set_rows(self, lines: List[str]):
        self.rows = list(lines)
        self.edges = self._cut_edges()

    def _cut_edges(self):
        return [
            self.Edge(self.rows[0],  'n'),
            self.Edge(self._col(-1), 'e'),
            self.Edge(self.rows[-1], 's'),
            self.Edge(self._col(0),  'w'),
        ]

    def _col(self, idx: int):
        chars = [r[idx] for r in self.rows]
        return "".join(chars)

    def edge(self, side: str) -> 'Edge':
        """Return the edge of the Tile that is at the side <side>.
        Side is identified by one of the letters n, s, e or w."""
        assert side in self.SIDES, \
            f"Invalid side spec: {side}. Must be one of {self.SIDES}"
        for edge in self.edges:
            if edge.side == side:
                return edge

    def __str__(self):
        return "\n".join(self.rows)

    def __repr__(self):
        name = str(self.__class__.__name__)
        lines = ["<{} id={}".format(name, self.id)]
        for edge in self.edges:
            lines.append("  {}".format(repr(edge)))
        lines.append(f"/{name}>")
        return "\n".join(lines)

    class Edge(object):

        def __init__(self, value: str, side: str):
            self.value = value
            self.side = side.lower()
            assert self.side in Tile.SIDES, \
                f"Wrong side spec: {side}, must be one of {Tile.SIDES}"

        def __repr__(self):
            return "<{} side={} '{}'>".format(
                self.__class__.__name__, self.side, self.value)

        def __eq__(self, other: Union['Edge', str]) -> bool:
            if isinstance(other, type(self)):
                other = other.value
            return self.value == other

        def compute_transform_to(self, other: 'Edge') -> str:
            """Compute a transform (in terms of `Tile.transform()`) that will
            convert the current Edge to given <other> Edge.
            """
            trn, flips = None, None
            if other == self:
                flips = ['', '', 'h', 'v']
            elif other == self.value[::-1]:
                flips = ['h', 'v', '', '']
            if flips:
                if self.side in 'we':
                    flips.append(flips.pop(0))
                n_steps = Tile.clockwise_distance(self.side, other.side)
                trn = 'r' * n_steps + flips[n_steps]
                trn = trn or '_'   # to have _ instead of ''
            return trn


class Deck(object):

    @classmethod
    def from_lines(cls, lines: List[str]) -> 'Deck':
        tiles = []
        for grp in utils.group_lines(lines):
            if grp:
                # print("\n".join(grp))
                tile = Tile.from_lines(grp)
                assert tile.is_square(), \
                    f"Tile must be square:\n{tile}"
                assert tile.id is not None, \
                    f"Tile must have an id:\n{tile}"
                # print(tile)
                tiles.append(tile)
        return cls(tiles)

    def __getitem__(self, idx):
        return self.tiles[idx]

    def __init__(self, tiles):
        self.tiles = list(tiles)

    def __len__(self):
        return len(self.tiles)

    def __iter__(self):
        return iter(self.tiles)

    def find(self, id: int):
        """Find a tile in the deck by tile id"""
        idx = self._find_index(id)
        if idx is not None:
            return self.tiles[idx]
        raise ValueError(f"Tile {id} not found")

    def take(self, tile: Tile) -> Tile:
        if isinstance(tile, Tile):
            idx = self._find_index(tile.id)
        else:
            idx = tile
        if idx is not None:
            return self.tiles.pop(idx)
        raise ValueError(f"Tile {tile.id} not found")

    def _find_index(self, id: int) -> Optional[int]:
        for idx, deck_tile in enumerate(self.tiles):
            if deck_tile.id == id:
                return idx
        return None

    def add(self, tile: Tile):
        assert isinstance(tile, Tile), "Wrong object type, must be Tile"
        idx = self._find_index(tile.id)
        assert idx is None, "Will not add a duplicate Tile"
        self.tiles.append(tile)


class PuzzleBoard(object):

    def __init__(self, size: Shape):
        assert len(size) == 2, \
            f"Size must be a list/tuple of two ints but got {size}"
        self.size = tuple(size)
        self.cells = {}

    def __getitem__(self, pos: Position) -> Optional[Tile]:
        """Return Tile that is located at the position <pos>."""
        assert len(pos) == 2, f"Wrong position: {pos}"
        return self.cells.get(tuple(pos), None)

    def places(self, is_suitable: Callable = None) -> List[Position]:
        """
        Return a list of places on the board, each place being represented
        as a tuple (row, column) (aka Position).
        An additional selection criterion can be supplied by passing a callable
        (function) to the method.
        """
        is_suitable = is_suitable or (lambda x: True)
        places = []
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                pos = (r, c)
                if is_suitable(pos):
                    places.append(pos)
        return places

    def free_places(self) -> List[Position]:
        """Return a list of positions that are not occupied by a Tile."""
        def if_free(pos): return self[pos] is None
        return self.places(if_free)

    def place(self, tile: Tile, pos: Position) -> Tile:
        """Add given <tile> to the given position on the board."""
        self._check_tile_already_on_board(tile, pos)
        # print("Tile id={} goes to {}".format(tile.id, pos))
        old = self.cells.get(pos, None)
        self.cells[pos] = tile
        return old  # needed?

    def _check_tile_already_on_board(self, other_tile, other_pos):
        for pos, tile in self.cells.items():
            if tile.id == other_tile.id and pos != other_pos:
                raise RuntimeError(f"Tile {tile.id} is already at {pos}")

    def take(self, pos: Position) -> Optional[Tile]:
        """Remove a tile (if any) from given position returning the tile."""
        return self.cells.pop(pos, None)

    def _corners(self) -> List[Position]:
        """Return a list of positions at the corners of the board"""
        h, w = self.size
        return [(0, 0), (0, w-1), (h-1, w-1), (h-1, 0)]

    def corners(self) -> List[Optional[Tile]]:
        """Return tiles that are at the four corners."""
        return [self[p] for p in self._corners()]

    def is_corner(self, pos: Position) -> bool:
        """Test whether given position is at a corner of the board."""
        return pos in self._corners()

    def is_side(self, pos: Position) -> bool:
        """Test whether given position is located along the side (including at
        the corner) of the board."""
        h, w = self.size
        return (pos[0] in (0, w-1) or pos[1] in (0, h-1))

    def __str__(self):
        return self.to_s(' ')

    def to_s(self, sep=''):
        nrows, ncols = self.size
        rows = [None] * nrows
        for r in range(nrows):
            for c in range(ncols):
                tile = self[(r, c)]
                if rows[r] is None:
                    rows[r] = tile
                else:
                    rows[r] = tile.hstack(rows[r], tile, sep)
        if sep:
            lines = []
            seps = [sep * rows[0].size[1]] * len(rows)
            for r, s in zip(rows, seps):
                lines.append(str(r))
                lines.append(s)
            lines.pop()
        else:
            lines = [str(r) for r in rows]
        return "\n".join(lines)


class TileFitter(object):
    """Iterator that will transform given tile to fit it into the given
    position on the board."""

    OFFSETS = [
        (-1, 0, "n"), (+1, 0, "s"),
        (0, +1, "e"), (0, -1, "w")
    ]

    OPPOSITE_SIDES = {'n': 's', 's': 'n', 'e': 'w', 'w': 'e'}

    TRANSFORMS = [
        '_', 'r', 'r', 'r', 'rh', 'r', 'r', 'r', 'hv', 'r', 'r', 'r',  # 'v'
    ]

    NUM_TRANSFORMS = 0

    @classmethod
    def count_transforms(cls):
        cls.NUM_TRANSFORMS += 1

    def __init__(self, board: PuzzleBoard, pos: Position, tile: Tile):
        self.board = board
        self.pos = pos
        self.tile = tile
        self.debug = False
        self.transform_ = None  # the most recent transformation
        self._set_surrounding_edges()
        self._set_transforms()

    def _set_transforms(self):
        """Set a list of transforms that are relevant for fitting the current
        tile to the current position on the board.
        """
        if self._has_all_expected_edges():
            self.transforms = list(self.TRANSFORMS)

            # Decide what transformations are actually relevant and place them
            # at the beginning of the list of transforms.
            # NOTE: we still want to use the whole list of transforms to cover
            # our asses in case our suggested transform(s) is/are incorrect.
            # After extensive testing of the algorithm for suggesting
            # transforms, default transforms can be removed.

            # For now we handle the simplest case only:
            # Pick the first suggested transform that is common to all edges.
            transforms = defaultdict(int)
            for _, edges in self.matching_edges.items():
                for edge in edges:
                    trn = edge[1]
                    transforms[trn] += 1
            for trn, count in transforms.items():
                if count == len(self.matching_edges):
                    self.transforms[0] = trn
        else:
            # The tile does not have expected edges
            self.transforms = []

    def _has_all_expected_edges(self):
        """Check that the current tile has all edges that are expected based
        on the adjacent tiles (surrounding edges).

        Additionally, collect information on corresponding edges, that can
        be used to determine what transforms are needed to bring the tile into
        desired orientation.
        """
        self.matching_edges = defaultdict(list)
        for side, surr_edge in self.surrounding_edges.items():
            exp_edge = Tile.Edge(surr_edge.value, side)
            # For every edge of the current tile, suggest a transform
            # to convert the edge to corresponding surrounding edge.
            for myedge in self.tile.edges:
                trn = myedge.compute_transform_to(exp_edge)
                if trn:
                    self.matching_edges[side].append((myedge, trn))

        ok = len(self.matching_edges) >= len(self.surrounding_edges)

        if ok and self.debug:
            print("--- Surrounding/matching edges ---")
            for side, exp_edge in self.surrounding_edges.items():
                print(f"Side: {side}")
                print("..Expected:\t", self.surrounding_edges[side])
                print("..Candidates:\t", self.matching_edges[side])
            print("--- END ---")

        return ok

    def _set_surrounding_edges(self):
        """Inspect the tiles that are adjacent to the current position on
        the board in all 4 directions (n, s, w, e) and from those tiles collect
        edges that are adjacent to the current position.
        Something like this:
        {
           "n": Southern edge of the tile to the North of the current position
           "w": Eastern edge of the tile to the West of the current position
        }
        """
        self.surrounding_edges = {}
        for dh, dw, side in self.OFFSETS:
            pos = (self.pos[0] + dh, self.pos[1] + dw)
            tile = self.board[pos]
            if tile:
                edge = tile.edge(self.OPPOSITE_SIDES[side])
                self.surrounding_edges[side] = edge
        # print("-- surrounding edges ---")
        # print(self.surrounding_edges)

    def __iter__(self):
        return self

    def __next__(self):
        """Manipulate the tile to fit to given place on the board."""
        while self.transforms:
            self.transform_ = self.transforms.pop(0)
            self._verbose(f"applying {self.transform_} to {self.tile.id}")
            self.tile.transform(self.transform_)
            self.count_transforms()
            if self.tile_fits(self.tile):
                return self.tile
        raise StopIteration

    def tile_fits(self, tile: Tile) -> bool:
        """Determine if the tile fits surrounding tiles exactly."""
        res = True
        for side, exp_edge in self.surrounding_edges.items():
            if exp_edge != tile.edge(side):
                res = False
                break
        self._verbose(f"Tile fits -- {res}")
        return res

    def _verbose(self, msg):
        if self.debug:
            print(msg)

class arrange_tiles(object):
    """ Recursive brute-force algorithm """

    def __init__(self, board: PuzzleBoard, deck: Deck):
        self.board = board
        self.deck = deck
        self.level = 0
        self.debug = False

        self._learn()

        TileFitter.NUM_TRANSFORMS = 0  # for dev purposes
        self._arrange_tiles()

    def _learn(self):
        """Inspect the deck and the board and precompute some useful data:
        * for each tile, collect a list of neighbouring tiles. This can be
          easily determined by finding common edges.
        """

        self.tile_neighbours = defaultdict(list)  # tile.id: List[Tile]

        common_edges = {}  # edge: List[Tile]
        for tile in self.deck:
            for edge in tile.edges:
                added = False
                for text in [edge.value, edge.value[::-1]]:
                    if text in common_edges:
                        common_edges[text].append(tile)
                        added = True
                if not added:
                    common_edges.setdefault(edge.value, []).append(tile)
        for tiles in common_edges.values():
            for i in range(len(tiles)):
                for j in range(len(tiles)):
                    if i != j:
                        self.tile_neighbours[tiles[i].id].append(tiles[j])

        assert len(self.deck) == len(self.tile_neighbours), "Unbelievable"

        if self.debug:
            print("{} learnt the following nearby tiles:".format(
                self.__class__.__name__))
            for k, vals in self.tile_neighbours.items():
                print(k, [t.id for t in vals])

    def _select_candidate_tiles(self, pos: Position) -> List[Tile]:
        """Select tiles from the deck that can fit given position <pos> on the
        board given the following heuristics:
        * for every position on the board we know how many tiles can be around:
          - a tile in the board corner can only have 2 neighbouring tiles,
          - a tile along the side of the board can have 3 neighbouring tiles,
          - otherwise a tile can have 4 neighbouring tiles.
        * for all tiles in the deck we have precomputed a list of neighbouring
          tiles. see self._learn() method.
        """
        n_neighbours = 4
        if self.board.is_corner(pos):
            n_neighbours = 2
        elif self.board.is_side(pos):
            n_neighbours = 3
        # NOTE: filter with lambda turned out to be not a good idea, because
        # it did not create an inmutable list. Instead, it produced something
        # that had duplicates, probably coming from changed underlying deck.
        tiles = [tile for tile in self.deck
                 if len(self._neighbours(tile) or []) == n_neighbours]
        return tiles

    def _neighbours(self, tile: Union[Tile, int]) -> List[int]:
        if isinstance(tile, Tile):
            tile = tile.id
        return self.tile_neighbours.get(tile, None)

    def _turn(self, tile: Tile, pos: Position):
        return TileFitter(self.board, pos, tile)

    def _arrange_tiles(self):
        """
        Arrange remaining tiles from the deck into free places on the board.

        Algorithm
        ---------
        * pick the 1st unoccupied place on the board (see Filling Order).
        * place a tile on that place on the board.
        * recursive (with reduced deck and partially filled board)

        Filling Order
        --------------
        By diagonals, that is
        1) fill (0,0);
        2) fill spots adjacent to (0,0) -- (0,1) and (1,0);
        3) fill adjacent to (0,1) and to (1,0) -- (0,2), (1,1), (2,0)
        4) and so on

        It seems that filling by diagonals requires less attemps to fill
        the whole board than filling by rows. TODO: prove it?
        A problem when filling by rows:
        when the 0th row -- (0, 0)...(0, last) -- is filled and we fail to fill
        the place (1,0), the problem is in the tile at (0, 0) rather than in
        the tile at (0,last), hence we should invalidate the *whole* 0th row.
        Honestly, filling by diagonals shows the same issue.
        TODO: this can be improved by passing up from the recursive calls
        information about the place on the board that caused the problem:
        we keep returning from the recursive calls until we are back to that
        place where that place on the board is handled.

        TODO:
        the very first tile should be placed on the board in the orientation
        that leaves both edges common with other tiles facing inside the board.
        However, filling by diagonal helps to accomplish it by detecting wrong
        filling of (0,0) very early.
        """

        self.level += 1

        def by_diagonal(pos: Position):
            """
            Examples:
            (0, 0) -> (0, 0, 0)
            (0, 2) -> (2, 0, 2)
            (1, 1) -> (2, 1, 1)
            (2, 0) -> (2, 2, 0)
            (0, 3) -> (3, 0, 3)
            If this collection is sorted, the order will be like this:
            [(0, 0),    (0, 2), (1, 1), (2, 0),    (0, 3)]
            """
            return [sum(pos)] + list(pos)

        places = sorted(self.board.free_places(), key=by_diagonal)
        # print("Free places", places)

        if not places:
            self.level -= 1
            return True

        assert len(places) == len(self.deck), \
          "Deck size does not match the amount of remaining board space"

        pos = places[0]
        tiles = self._select_candidate_tiles(pos)

        # print(f"Filling in place: {pos}, with {len(tiles)} candidates")
        # reveal_deck(tiles)

        found = False
        for i, tile in enumerate(tiles):
            self.deck.take(tile)
            self.board.place(tile, pos)

            # print("..Trying for {} the tile #{}: {} (level={})".format(
            #         pos, i, tile.id, self.level))

            for t in self._turn(tile, pos):
                if self._arrange_tiles():
                    found = True
                    break

            if found:
                # we found that the current tile matches the current spot on
                # the board. No need to do anything else for the current spot.
                break
            else:
                # the current tile does not match the current spot: we remove
                # the tile from the board and put it back into the deck.
                # print(f"..No fit for {pos} with {tile.id}")
                self.board.take(pos)
                self.deck.add(tile)

        # if not found:
        #     print(f"..No fit for {pos} with any tile. Go to previous step")

        if self.level == 1 and self.debug:
            print("--- resulting deck ---")
            reveal_deck(self.deck)
            print("--- resulting board --=")
            print(self.board)

        if self.level == 1:
            print("Total number of transforms applied to Tiles: {}".format(
                TileFitter.NUM_TRANSFORMS))

        self.level -= 1

        return found


def solve_p1(lines: List[str], do_part=1) -> Union[int, PuzzleBoard]:
    """Solution to the 1st part of the challenge"""
    deck = Deck.from_lines(lines)

    # demo_tile_transforms(deck)
    # demo_deck_capacities(deck)
    # return 0

    h = int(math.sqrt(len(deck)))
    size = (h, h)
    assert size[0] * size[1] == len(deck), \
        f"Expecting a square arrangement but {len(deck)} does not fit into {size}"

    board = PuzzleBoard(size)

    # demo_fill_board_sequentially(board, deck)
    # demo_fill_board_randomly(board, deck)
    # demo_tile_fitting(board, deck)
    # return 0

    arrange_tiles(board, deck)

    if DEBUG:
        print("--- FINAL BOARD ---")
        for pos in board.places():
            tile = board[pos]
            print(pos, tile.id if tile else None)

    if do_part == 1:
        corners = [t.id for t in board.corners()]
        res = reduce(lambda a, b: a * b, corners)

    else:
        res = board

    return res


class scan(object):
    """Scan given <tile> generating smaller tiles of size <kernel>"""

    def __init__(self, tile, kernel=(3, 20)):
        self.tile = tile
        self.th, self.tw = tile.size
        self.kh, self.kw = kernel
        self.ri, self.ci = 0, -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.ci < (self.tw - self.kw):
            self.ci += 1
        else:
            self.ri += 1
            self.ci = 0
        if self.ri <= (self.th - self.kh):
            topleft = (self.ri, self.ci)
            return topleft, self.tile.crop(topleft, self.kh, self.kw)
        else:
            raise StopIteration


def can_see_monster(sea: Tile, monster: Tile) -> bool:
    """Compare two tiles -- the Sea tile and the Monster tile -- and tell if
    the Sea tile contains the Monster.

    Algorithm
    ---------
    To accomplish matching, both the Sea and Monster tiles are represented as
    matrices of 0s and 1s. When elementwise multiplication of said two matrices
    produces the matrix that is elementwise equal to Monster, this means
    the Sea tile contains the Monster.
    Since the Moster can be upside down or point to the left or right, the Sea
    tile is turned (flipped vertically and horizontally) a few times. Mismatch
    in rotation is handled elsewhere.
    """
    assert sea.size == monster.size, "Sizes must be equal"
    sea = sea.to_onehot()
    monster = monster.to_onehot()
    for s in turn(sea):
        # if elementwise multiplication produces a matrix that is equal to
        # the monster, this means the monster has been detected
        if mmult(monster, s) == monster:
            return True
    return False


def mmult(m1, m2):
    """Multiply elementwise two 2d matrices"""
    shape1 = (len(m1), len(m1[0]))
    shape2 = (len(m2), len(m2[0]))
    assert shape1 == shape2, f"Incompatible shapes: {shape1} vs {shape2}"
    m3 = []
    for r in range(shape1[0]):
        m3.append([])
        for c in range(shape1[1]):
            val = m1[r][c] * m2[r][c]
            m3[r].append(val)
    shape3 = (len(m3), len(m3[0]))
    assert shape1 == shape3, \
        f"Input shape must match output shape: {shape1} vs {shape3}"
    return m3


class turn(object):
    """
    Generate 4 views of given 2d matrix by sequentially flipping it vertically
    and horizontally, returning at:
    * step 0: the original matrix
    * step 1: flipped horizontally
    * step 2: further flipped vertically
    * step 3: further flipped horizontally

    An additional vertical flip, that is not performed, would produce
    the original arrangement of the matrix (as in step 0).
    """

    def __init__(self, mat):
        self.mat2d = [list(r) for r in mat]
        self.round = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.round += 1
        if self.round > 3:
            raise StopIteration
        if self.round in {1, 3}:
            # flip horizontally
            self.mat2d = [list(reversed(r)) for r in self.mat2d]
        elif self.round == 2:
            # flip vertically
            self.mat2d = list(reversed(self.mat2d))
        return self.mat2d


def detect_monsters(sea: Tile, monster: Tile) -> List[Position]:
    """
    Find Monster(s) in the See. The monster can be flipped/rotated, returning
    a list of unique points that represent the top left coordinate of the
    bounding box enclosing the monster. The returned points are not really
    used but they help to dismiss possible duplicates.

    Algorithm
    ----------
    Slide the Sea tile with a window of the size equal to the size of Monster
    tile, comparing each generated window with the Monster tile.
    We need to rotate the Sea tile once (90 degrees clockwise).
    Other discrepancies are handled (by flipping the tile) elsewhere.
    """
    # TODO
    # At this point, for optimization purposes, it would be better to have
    # the Sea and the Monster represented as a matrix of zeroes and ones.
    monsters = set()
    for _ in range(2):
        for topleft, tile in scan(sea, monster.size):
            # print(f"Sea fragment at {topleft}:\n{tile}")
            if can_see_monster(tile, monster):
                monsters.add(topleft)
                # print(topleft)
                # print(tile)
        sea.rotate()
    return list(monsters)


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    board = solve_p1(lines, do_part=2)

    # Build sea -- a new board with the same tiles slightly cropped
    # (the edges must be removed from all sides of the tiles).
    sea = PuzzleBoard(board.size)
    for pos in board.places():
        tile = board[pos]
        h, w = tile.size
        t = tile.crop((1, 1), h-2, w-2)
        t.id = tile.id
        sea.place(t, pos)

    # It is easier to manipulate sea as a Tile, not as a PuzzleBoard.
    sea = Tile(sea.to_s().split('\n'))
    if DEBUG:
        print(f"--- Tile: Sea {sea.size} ---")
        print(sea)

    monster = Tile(SEA_MONSTER.split('\n'))
    if DEBUG:
        print(f"--- Tile: Monster {monster.size} ---")
        print(monster)

    found_monsters = detect_monsters(sea, monster)

    roughness = sea.count('#')
    roughness -= monster.count('#') * len(found_monsters)

    return roughness


# Expected arrangement of the testcase tiles.
# A transposed variant is also valid
# 1951    2311    3079
# 2729    1427    2473
# 2971    1489    1171

tests = [
    (utils.load_input("test_input.0.txt"), 1951 * 3079 * 2971 * 1171, 273),
]


FAKE_SEA_MONSTER = """\
.####....####....##.
...##.#..##.##.##.#.
...#..##.##..###.##."""


REAL_SEA_MONSTER = """\
..................#.
#....##....##....###
.#..#..#..#..#..#..."""


SEA_MONSTER = REAL_SEA_MONSTER


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        if exp1 is not None:
            res1 = solve_p1(inp)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '20'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 83775126454273
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 1993
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
