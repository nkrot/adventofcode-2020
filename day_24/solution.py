#!/usr/bin/env python

# # #
#
# https://www.redblobgames.com/grids/hexagons/
#
#
# | B |   | B |
#  \ / \ / \ /
#   | B |   |
#  / \ / \ / \
# | B | B |   |
#

import re
import os
import sys
from typing import List, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False
DO_PLOT = True


class HexagonalCoord(object):

    DIRECTIONS = ['w', 'e', 'ne', 'nw', 'se', 'sw']

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
        """
        assert direction in self.DIRECTIONS, \
            f"Invalid neighbour direction: {direction}"
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


COORD = EvenRCoord


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


def demo_tile():
    t = Tile(COORD(0, 0))
    print(t)
    for _ in range(2):
        t.flip()
        print(t)


# demo_tile()
# exit(100)

def demo_coord():
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


# demo_coord()
# exit(100)


def find_tile(path: tuple) -> Tile:
    """Return tile reached from the reference tile"""

    if DEBUG:
        print(f"\nPath: {path}")

    tiles = {}

    xy = COORD(0, 0)
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
    assert isinstance(tile, EvenRCoord), \
        f"Cannot plot coordinates of type {type(tile)}"

    xspan, yspan = COORD.spans(tiles.keys())

    xspan = (xspan[0], xspan[1]+1)
    yspan = (yspan[0], yspan[1]+1)

    ywidth = (yspan[1] - yspan[0])
    stripes = [ ' \ /'  * ywidth + ' \\', ' / \\' * ywidth + ' /' ]
    indents = ["| ", "  | "]

    print(stripes[1])
    for i, x in enumerate(range(xspan[0], xspan[1])):
        row = []
        for y in range(yspan[0], yspan[1]):
            xy = COORD(x, y)
            tile = tiles.get(xy, Tile(xy))
            ch = 'B' if tile.is_black() else ' '
            row.append(ch)
        print(indents[i%2] + " | ".join(row) + " |")
        #if x+1 < xspan[1]:
        print(stripes[i%2])


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
