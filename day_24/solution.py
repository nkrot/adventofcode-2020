#!/usr/bin/env python

# # #
#
# Useful reading on how to implement coordinates in hexagonal grids:
#   https://www.redblobgames.com/grids/hexagons/
#
# This script implements
# - EvenRCoord (supports plotting). A sort of offset coordinates
# - CubeCoord (does not support plotting)
# - AxialCoord (does not support plotting)
# - DoubledCoord (does not suppoert plotting)
# A specific implementation can be chosen by setting COORD variable to one
# of the above classes.
#
# Plotting (set DO_PLOT to enable)
# ========
# Supported for EvenRCoord only.
#
#  / \ / \ / \
# | B |   | B |
#  \ / \ / \ /
#   | B |   |
#  / \ / \ / \
# | B | B |   |
#  \ / \ / \ /
#

import re
import os
import sys
from typing import List, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False
DO_PLOT = not True


class HexagonalCoord(object):

    DIRECTIONS = ['w', 'e', 'ne', 'nw', 'se', 'sw']
    OFFSETS = {}  # a subclass must define a valid map

    @classmethod
    def parse_directions(cls, line: str) -> List[str]:
        line = line.strip()
        reg = r'({})'.format("|".join(cls.DIRECTIONS))
        tokens = re.findall(reg, line)
        assert line == "".join(tokens), \
            "Parser line does not correspond to the input line"
        return tokens

    def neighbours(self):
        for d in self.DIRECTIONS:
            yield(self.neighbour(d))

    @staticmethod
    def spans(coords: list) -> List[list]:
        """Inspect given list of coordinates and return spans for every
        dimension. A span is two values [min, max] -- minimal and maximal value
        in a dimension.
        """
        mins, maxs = [], []
        for i, coord in enumerate(coords):
            if i == 0:
                for v in coord:
                    mins.append(v)
                    maxs.append(v)
            else:
                for j, v in enumerate(coord):
                    mins[j] = min(mins[j], v)
                    maxs[j] = max(maxs[j], v)
        return list(zip(mins, maxs))

    def __iter__(self):
        raise NotImplementedError("Subclass must implement __iter__()")

    def __eq__(self, other):
        return list(self) == list(other)

    def __hash__(self):
        return hash(tuple(self))

    def __str__(self):
        return str(tuple(self))

    def __add__(self, other):
        if not isinstance(other, (type(self), tuple, list)):
            raise ValueError(f"Unsupported type: {type(other)}")
        args = [sum(vs) for vs in zip(list(self), list(other))]
        return self.__class__(*args)

    def neighbour(self, direction):
        """Return the coordinate that is located in the given <direction> with
        respect to the current coordinate.
        """
        assert direction in self.DIRECTIONS, \
            f"Invalid neighbour direction: {direction}"
        return self + self.OFFSETS[direction]


class EvenRCoord(HexagonalCoord):
    """Even-rows system as explained in the article
    https://www.redblobgames.com/grids/hexagons/
    """

    OFFSETS = { 'n': -1, 's': +1, 'w': -1, 'e': +1 }

    @classmethod
    def origin(cls):
        return cls(0, 0)

    def __init__(self, *args):
        self.x, self.y = args

    def __iter__(self):
        return iter([self.x, self.y])

    def neighbour(self, direction):
        """Return the coordinate that is located in the given <direction> with
        respect to the current coordinate.
        TODO: here an easier and nicer method
        https://www.redblobgames.com/grids/hexagons/#neighbors-offset
        """
        assert direction in self.DIRECTIONS + ['s', 'n'], \
            f"Invalid neighbour direction: {direction}"

        if direction in 'ns':
            # Special case to assist plotting:
            # Coordinate N or S of the current one *in the same column*
            direction = direction + ('e', 'w')[self.x % 2]
            return self.neighbour(direction)

        if direction in 'we':
            dx, dy = 0, self.OFFSETS[direction]
        else:
            r, c = list(direction)
            dx = self.OFFSETS[r]
            if c == 'w':
                # k must be 1 in even rows and 0 otherwise
                k = (abs(self.x) + 1) % 2
            if c == 'e':
                k = self.x % 2
            dy = self.OFFSETS[c] * k
        return self + (dx, dy)


class CubeCoord(HexagonalCoord):

    OFFSETS = {
        "w"  : (-1, +1, 0), "e"  : (+1, -1, 0),
        "nw" : (0, +1, -1), "se" : (0, -1, +1),
        "ne" : (+1, 0, -1), "sw" : (-1, 0, +1),
    }

    @classmethod
    def origin(cls):
        return cls(0, 0, 0)

    def __init__(self, *args):
        self.x, self.y, self.z = args

    def __iter__(self):
        return iter([self.x, self.y, self.z])


class AxialCoord(HexagonalCoord):
    # the column is skewed and goes diagonally from top left to bottom right

    OFFSETS = {
        # row, column
        "w"  : (0, -1),  "e"  : (0, +1),
        "nw" : (-1, 0),  "se" : (+1, 0),
        "ne" : (-1, +1), "sw" : (+1, -1),
    }

    @classmethod
    def origin(cls):
        return cls(0, 0)

    def __init__(self, *args):
        self.x, self.y = args

    def __iter__(self):
        return iter([self.x, self.y])


class DoubledCoord(HexagonalCoord):
    # the step in a row is 2 (instead of 1).

    OFFSETS = {
        # row, column
        "w"  : (0, -2),  "e"  : (0, +2),
        "nw" : (-1, -1), "se" : (+1, +1),
        "ne" : (-1, +1), "sw" : (+1, -1),
    }

    @classmethod
    def origin(cls):
        return cls(0, 0)

    def __init__(self, *args):
        self.x, self.y = args

    def __iter__(self):
        return iter([self.x, self.y])


class Tile(object):

    STATES = ['white', 'black']

    def __init__(self, coord, state=0):
        self.coord = coord
        self.states = []  # history of states
        self.state = state

    # def neighbours(self):
    #     """Return all exisiting neighbouring (adjacent) tiles"""
    #     pass

    # def neighbour(self, direction):
    #     """Return the tile that is in the given direction from the current tile
    #     """
    #     pass

    def is_black(self):
        return self.state == 1

    def is_white(self):
        return self.state == 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, newstate):
        self.states.append(newstate)
        self._state = newstate

    @property
    def position(self):
        return self.coord

    def flip(self):
        self.state = (self.state + 1) % 2
        return self

    def __repr__(self):
        return "<{}: {} state={} ({}) history={}>".format(
            self.__class__.__name__, self.position,
            self.state, self.STATES[self.state],
            self.states)


COORD = EvenRCoord
# COORD = CubeCoord
# COORD = AxialCoord
# COORD = DoubledCoord


def demo_coord():
    # works for EvenRCoord, does not work for CubeCoord
    x, y = 0, 1

    xy = COORD(x, y)
    print(xy, type(xy))
    tile = Tile(xy)
    print(tile)

    tiles = {}
    tiles[tile.position] = tile
    print(tiles)

    if xy in tiles:
        print("..found 1")

    if COORD(x, y) in tiles:
        print("..found 2")
    else:
        raise RuntimeError("OOps. This should not happen")

    print("Demo Coord + Coord:")
    a = xy + xy
    print(type(a), a)
    b = xy + (10, 2)
    print(type(b), b)
    c = xy + [20, 1]
    print(type(c), c)

    # d = xy + [20, 1, 3]
    # print(type(d), d)


def demo_tile():
    t = Tile(COORD.origin())
    print(t)
    for _ in range(2):
        t.flip()
        print(t)


# demo_coord()
# demo_tile()
# exit(100)


def find_tile(path: tuple) -> Tile:
    """Return tile reached from the reference tile"""

    if DEBUG:
        print(f"\nPath: {path}")

    tiles = {}

    xy = COORD.origin()
    tile = tiles.setdefault(xy, Tile(xy))

    if DEBUG:
        print(f"Startng tile: {tile}")

    for code in path:
        xy = tile.coord.neighbour(code)
        tile = tiles.setdefault(xy, Tile(xy))
        if DEBUG:
            print(f"Current tile: {tile}")

    if DEBUG:
        print(f"Reached tile: {tile}")

    return tile



def flip_tiles(tiles: dict) -> None:

    flippable = []

    # Any black tile with zero or more than 2 black tiles immediately
    # adjacent to it is flipped to white.
    for pos in list(tiles.keys()):
        tile = tiles[pos]
        # print(pos, tile)

        if tile.is_black():
            neighbours = []
            for xy in tile.coord.neighbours():
                # create surrounding tile if does not exist
                neigh = tiles.setdefault(xy, Tile(xy))
                neighbours.append(neigh)
            n_blacks = count_black_tiles(neighbours)
            if n_blacks == 0 or n_blacks > 2:
                flippable.append(tile)

    # Any white tile with exactly 2 black tiles immediately adjacent to it
    # is flipped to black.
    for pos, tile in tiles.items():
        if tile.is_white():
            neighbours = []
            for xy in tile.coord.neighbours():
                if xy in tiles:
                    neighbours.append(tiles[xy])
            n_blacks = count_black_tiles(neighbours)
            if n_blacks == 2:
                flippable.append(tile)

    # TODO: what if a tile matches both above conditions?

    for tile in flippable:
        tile.flip()

    pass


def demo_find_tile():
    paths = [COORD.parse_directions(line) for line in ["esew", "nwwswee"]]
    for path in paths:
        tile = find_tile(path)
        print(tile)


# demo_find_tile()
# exit(100)


def count_black_tiles(tiles: Union[dict, list]) -> int:
    if isinstance(tiles, dict):
        return sum(t.is_black() for t in tiles.values())
    else:
        return sum(t.is_black() for t in tiles)


def plot(tiles):
    tile = list(tiles.keys())[0]
    if not isinstance(tile, EvenRCoord):
        print(f"Cannot plot coordinates of type {type(tile)}")
        return False

    # TODO: ask for height and width from COORD. these values do not
    # obligatorily come from xspan and yspan.
    xspan, yspan = COORD.spans(tiles.keys())

    height = xspan[1] - xspan[0]
    width = yspan[1] - yspan[0]

    stripes = [
         ' \ /' * (1+width) + ' \\',
        ' / \\' * (1+width) + ' /' ]
    indents = ["| ", "  | "]

    def tile(coord, chars=['B', ' ']):
        # chars is for debug purposes
        t = tiles.get(coord, Tile(coord))
        return chars[0] if t.is_black() else chars[1]

    # top left corner in the grid
    # TODO: ask COORD for top left corner of the grid
    xy = COORD(xspan[0], yspan[0])

    print(stripes[1])
    for r in range(height+1):
        first_in_row = xy
        row = [tile(xy)]
        for _ in range(width):
            xy = xy.neighbour("e")
            row.append(tile(xy))
        print("{}{} |".format(indents[r%2], " | ".join(row)))
        print(stripes[r%2])
        # get read for processing the next row
        xy = first_in_row.neighbour('s')


def solve_p1(lines: List[str], do_part=1) -> int:
    """Solution to the 1st part of the challenge"""
    paths = [COORD.parse_directions(line) for line in lines]

    tiles = {}
    for path in paths:
        tile = find_tile(path)
        tile = tiles.setdefault(tile.position, tile)
        tile.flip()

    if DEBUG:
        print(len(tiles))
        for tile in tiles.values():
            print(tile)

    if do_part == 1:
        if DO_PLOT:
            plot(tiles)
        return count_black_tiles(tiles)
    else:
        return tiles


def solve_p2(lines: List[str], n_days=100) -> int:
    """Solution to the 2nd part of the challenge"""

    tiles = solve_p1(lines, do_part=2)

    for n in range(n_days):
        flip_tiles(tiles)
        cnt = count_black_tiles(tiles)

        if DEBUG:
            print(f"Day {n+1}: {cnt}")
        if DO_PLOT and n < 20:  # with larger n does not fit into term width
            if not DEBUG:
                print(f"Day {n+1}: {cnt}")
            plot(tiles)
            print()

    return cnt


text_1 = """sesenwnenenewseeswwswswwnenewsewsw
neeenesenwnwwswnenewnwwsewnenwseswesw
seswneswswsenwwnwse
nwnwneseeswswnenewneswwnewseswneseene
swweswneswnenwsewnwneneseenw
eesenwseswswnenwswnwnwsewwnwsene
sewnenenenesenwsewnenwwwse
wenwwweseeeweswwwnwwe
wsweesenenewnwwnwsenewsenwwsesesenwne
neeswseenwwswnwswswnw
nenwswwsewswnenenewsenwsenwnesesenew
enewnwewneswsewnwswenweswnenwsenwsw
sweneswneswneneenwnewenewwneswswnese
swwesenesewenwneswnwwneseswwne
enesenwswwswneneswsenwnewswseenwsese
wnwnesenesenenwwnenwsewesewsesesew
nenewswnwewswnenesenwnesewesw
eneswnwswnwsenenwnwnwwseeswneewsenese
neswnwewnwnwseenwseesewsenwsweewe
wseweeenwnesenwwwswnew"""


tests = [
    (text_1.split('\n'), 10+5-5, 2208),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        if exp1 is not None:
            res1 = solve_p1(inp)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp, n_days=100)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    global DO_PLOT
    DO_PLOT = False

    day = '24'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 420
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 4206
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
