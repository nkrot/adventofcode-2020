#!/usr/bin/env python

# # #
#
# TODO
# use numpy for Tile, Edge, PuzzleBoard

import re
import os
import sys
from typing import List, Tuple, Optional, Union
import math
from functools import reduce
import random

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False

class Tile(object):

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

    def __init__(self, rows, id=None):
        self.id = id
        self._set_rows(rows)
        self.orig_rows = None

    @property
    def size(self) -> Tuple[int, int]:
        """Return (height, width) of the tile"""
        return (len(self.rows), len(self.rows[0]))

    # TODO
    # 1. memoize transformations
    # 2. there is no need to transform the whole Tile. Suffice it to transform
    #    edges. However, the whole Tile needs to be transformed when str(tile)
    #    Therefore for now, just transforming the whole tile.

    def hflip(self):
        """Flip horizontally (swap left and right)"""
        print("::flipping horizontally")
        rows = [r[::-1] for r in self.rows]
        self._set_rows(rows)
        return self

    def vflip(self):
        """Flip vertically (swap top and bottom)"""
        print("::flipping vertically")
        rows = list(reversed(self.rows))
        self._set_rows(rows)
        return self

    def rotate(self, n_steps=1):
        """Rotate clockwise given number of steps, one step being equal to 90
        degrees."""
        print("::rotating")
        assert n_steps > -1, \
            f"Number pf rotation steps cannot be negative: {n_steps}"
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

    def _col(self, idx):
        chars = [r[idx] for r in self.rows]
        return "".join(chars)

    def edge(self, side: str):
        for edge in self.edges:
            if edge.side == side:
                return edge
        return None

    def __str__(self):
        lines = [] # [f"Tile: {self.id}"]
        lines.extend(self.rows)
        return "\n".join(lines)

    def __repr__(self):
        name = str(self.__class__.__name__)
        lines = ["<{} id={}".format(name, self.id)]
        for edge in self.edges:
            lines.append("  {}".format(repr(edge)))
        lines.append(f"/{name}>")
        return "\n".join(lines)

    class Edge(object):

        def __init__(self, value, side):
            self.value = value
            self.side = side.lower()
            assert self.side in 'nesw', f"Wrong side name: {side}"

        def __repr__(self):
            return "<{} side={} '{}'>".format(
                self.__class__.__name__, self.side, self.value)

        def fits(self, other: Union['Edge', str]) -> bool:
            print(f"Comparing edges: {self} vs {other}")
            if isinstance(other, type(self)):
                other = other.value
            return self.value == other


class Deck(object):

    @classmethod
    def from_lines(cls, lines: List[str]) -> 'Deck':
        tiles = []
        for grp in utils.group_lines(lines):
            if grp:
                # print("\n".join(grp))
                tile = Tile.from_lines(grp)
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
        return iter(list(self.tiles))

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

    def __init__(self, size: Tuple[int, int]):
        assert len(size) == 2, \
            f"Size must be a list/tuple of two ints but is {size}"
        self.size = tuple(size)
        self.cells = {}

    def __getitem__(self, pos: Tuple[int, int]) -> Optional[Tile]:
        assert len(pos) == 2, f"Wrong position: {pos}"
        return self.cells.get(tuple(pos), None)

    def places(self, is_suitable=None):
        is_suitable = is_suitable or (lambda x: True)
        places = []
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                pos = (r, c)
                if is_suitable(pos):
                    places.append(pos)
        return places

    def free_places(self):
        if_free = lambda pos: self[pos] is None
        return self.places(if_free)

    # def fit(self, pos: Tuple[int, int], tile: Tile) -> bool:
    #     """Add given <tile> to the position <pos> on the board if the tile
    #     matches."""
    #     # get constraints for the current position:
    #     # what edges are expected on what sides
    #     # check that given tile complies with the constraints and add if yes
    #     added = False
    #     return added

    def place(self, tile: Tile, pos: Tuple[int, int]):
        """Add the tile at the given position on the board"""
        print("Tile id={} goes to {}".format(tile.id, pos))
        old = self.cells.get(pos, None)
        self.cells[pos] = tile
        return old  # needed?

    def take(self, pos: Tuple[int, int]) -> Optional[Tile]:
        """Remove a tile (if any) from given position"""
        return self.cells.pop(pos, None)

    def corners(self) -> List[Tile]:
        """Return 4 corner tiles"""
        h, w = self.size
        positions = [(0, 0), (0, w-1), (h-1, w-1), (h-1, 0)]
        tiles = [self[p] for p in positions]
        return tiles

    def __str__(self):
        nrows, ncols = self.size
        rows = [None] * nrows
        for r in range(nrows):
            for c in range(ncols):
                tile = self[(r, c)]
                if rows[r] is None:
                    rows[r] = tile
                else:
                    rows[r] = tile.hstack(rows[r], tile, ' ')
        seps = [" " * rows[0].size[1]] * len(rows)
        lines = []
        for r, s in zip(rows, seps):
            lines.append(str(r))
            lines.append(s)
        lines.pop()
        return "\n".join(lines)


class TileFitter(object):

    OFFSETS = [
        (-1, 0, "s"), (+1, 0, "n"),
        (0, +1, "w"), (0, -1, "e")
    ]

    OPPOSITE_SIDES = {'n': 's', 's': 'n', 'e': 'w', 'w': 'e'}

    def __init__(self, board: PuzzleBoard, pos: Tuple[int, int], tile: Tile):
        self.board = board
        self.pos = pos
        self.tile = tile
        self.surrounding_edges = self._collect_surrounding_edges()
        self._set_transforms()

    def _set_transforms(self):
        self.transform_ = None  # the most recent transformation
        checks = []
        for exp_edge in self.surrounding_edges:
            side = self.OPPOSITE_SIDES[exp_edge.side]
            for myedge in self.tile.edges:
                if exp_edge.fits(myedge) or exp_edge.fits(myedge.value[::-1]):
                    checks.append(True)
                    break
        # Optimization:
        # decide what transformations are relevant
        if checks.count(True) >= len(self.surrounding_edges):
            self.transforms = [
                '_', 'r', 'r', 'r', 'rh', 'r', 'r', 'r', 'hv',
                'r', 'r', 'r' #, 'v' this brings the tile into original state
            ]
        else:
            self.transforms = []

    def _collect_surrounding_edges(self):
        edges = []
        for dh, dw, opside in self.OFFSETS:
            h = self.pos[0] + dh
            w = self.pos[1] + dw
            tile = self.board[(h, w)]
            if tile:
                edge = tile.edge(opside)
                if edge:
                    edges.append(edge)
        print("-- surrounding edges ---")
        print(edges)
        return edges

    def __iter__(self):
        return self

    def __next__(self):
        """Manipulate the tile to fit to given place at the board."""
        while self.transforms:
            tr = self.transforms.pop(0)
            self.tile.transform(tr)
            self.transform_ = tr
            if self.tile_fits(self.tile):
                return self.tile
        raise StopIteration

    def tile_fits(self, tile):
        checks = []
        for exp_edge in self.surrounding_edges:
            print(f"Expected edge: {exp_edge}")
            side = self.OPPOSITE_SIDES[exp_edge.side]
            check = exp_edge.fits(tile.edge(side))
            checks.append(check)
            if not check:
                break
        # print(checks)
        return checks.count(False) == 0


def demo_tile_transforms(deck):
    print("--- Demo Tile Transforms ---")
    tile = deck[0]

    print("Initial tile:")
    print(tile)

    print(">> flipped horizontally")
    print(tile.hflip())

    print(">> flipped horizontally again")
    print(tile.hflip())

    print(">> flipped vertically")
    print(tile.vflip())

    print(">> flipped vertically again")
    print(tile.vflip())

    print(">> flipped vertically 2 times")
    print(tile.vflip().vflip())

    print(">> rotate 2 steps")
    print(tile.rotate(6))

    print(">> rotate 1 step")
    print(tile.rotate(1))

    print(">> rotate 3 steps")
    print(tile.rotate(3))


def demo_random_fill_board(board, deck):
    print("--- Demo Random Board Filling ---")

    places = board.places()
    random.shuffle(places)
    for pos, tile in zip(places, deck):
        board.place(tile, pos)

    print(board)


def reveal_deck(deck):
    tiles = []
    for idx, tile in enumerate(deck):
        tiles.append(tile.id)
    print(f"Size: {len(deck)}, Tiles: {tiles}")


def demo_deck_capacities(deck):
    print("-- Demo deck functionality ---")
    print("Size:", len(deck))

    print("State of the deck:")
    reveal_deck(deck)

    tiles2remove = []
    for i in {1, 3}:
        print(f"Tile at [{i}] : {deck[i].id}")
        tiles2remove.append(deck[i])

    print("Taking (removing) the above tiles from the deck")
    for tile in tiles2remove:
        deck.take(tile)

    print("State of the deck:")
    reveal_deck(deck)

    print("Re-adding the tiles back to the deck")
    for tile in tiles2remove:
        deck.add(tile)

    print("State of the deck:")
    reveal_deck(deck)

    id = 2311
    print(f"Searching the deck for a Tile by tile id: {id}")
    tile = deck.find(id)
    print(tile)
    print("Found?", tile.id == id)


def demo_tile_fitting(board, deck):

    tests = [
        ((1951, (0, 0)), (2311, (0, 1)),
         "should match side by side w/o transformations"),
        ((None, None),   (3079, (0, 2)),
         "Needs transforms"),
        ((None, None),   (2729, (1, 0)),
         "Needs transforms"),
        ((None, None),   (1427, (1, 1)),
         "Needs transforms"),
        ((None, None),   (2473, (1, 2)),
         "Needs transforms"),
        ((None, None),   (2971, (2, 0)),
         "Needs transforms"),
        ((None, None),   (1489, (2, 1)),
         "Needs transforms"),
        ((None, None),   (1171, (2, 2)),
         "Needs transforms"),
    ]

    for (id1, pos1), (id2, pos2), msg in tests:
        if id1:
            base = deck.find(id1)
            base.transform('v')

            base_fitter = TileFitter(board, pos1, base)
            for i, _ in enumerate(base_fitter):
                print(i)
            board.place(base, pos1)

            print(f"--- Base Tile {base.id} ---")
            print(base)

        tile = deck.find(id2)
        print(f"\n--- Target: {pos2}, fitting tile: id={tile.id}")
        print(tile)

        fitter = TileFitter(board, pos2, tile)
        for i, _ in enumerate(fitter):
            print(f"State of the tile after {i}th '{fitter.transform_}':")
            print(tile)
            board.place(tile, pos2)
            print("=== Placed !!!")
            break

    # print("--- Final Deck ---")
    # reveal_deck(deck)
    print("--- Final Board ---")
    print(board)


def arrange_tiles(board, deck, level=0):
    """
    """
    if not deck:
        return True

    pos = board.free_places()[0]
    print(f"Filling in place: {pos}, with {len(deck)} candidates")
    reveal_deck(deck)

    found = False
    tiles = list(deck.tiles)
    for i, tile in enumerate(tiles):
        deck.take(tile)
        print(f"Trying for {pos} tile at={i}: {tile.id}")
        fitter = TileFitter(board, pos, tile)
        board.place(tile, pos)
        for _ in fitter:
            if arrange_tiles(board, deck, level+1):
                found = True
                break
        if found:
            break
        else:
            board.take(pos)
            deck.add(tile)

    if level == 0 and DEBUG:
        print("--- resulting deck ---")
        reveal_deck(deck)
        print("--- resulting board --=")
        # print(board.cells)
        print(board)

    return found


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    deck = Deck.from_lines(lines)

    # demo_tile_transforms(deck)
    # demo_deck_capacities(deck)
    # return 0

    size  = [int(math.sqrt(len(deck)))] * 2
    assert size[0] * size[1] == len(deck), \
        f"Expecting a square arrangement but {len(deck)} fits into {size}"
    board = PuzzleBoard(size)

    # demo_random_fill_board(board, deck)
    # demo_tile_fitting(board, deck)
    # return 0

    arrange_tiles(board, deck)

    corners = [t.id for t in board.corners()]
    return reduce(lambda a, b: a * b, corners)


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    # TODO
    return 0



tests = [
    (utils.load_input("test_input.0.txt"), 1951 * 3079 * 2971 * 1171, None),
]


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
    day = '20'  # TODO
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = -1
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = -1
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    # run_tests()
    run_real()
