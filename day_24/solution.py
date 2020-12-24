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


class Tile(object):

    DIRECTIONS = [ 'w', 'e', 'ne', 'nw', 'se', 'sw']
    OFFSETS = { 'n': -1, 's': +1, 'w': -1, 'e': +1 }

    STATES = ['white', 'black']

    def __init__(self, xy, state=0):
        self.x, self.y = xy
        self.states = []  # history of states
        self.state = state

    def neighbour(self, code):
        assert code in self.DIRECTIONS, f"Invalid neighbour: {code}"
        if code in 'we':
            dx, dy = 0, self.OFFSETS[code]
        else:
            r, c = list(code)
            dx = self.OFFSETS[r]
            if c == 'w':
                # k must be 1 in even rows and 0 otherwise
                k = (abs(self.x) + 1) % 2
            if c == 'e':
                k = self.x % 2
            dy = self.OFFSETS[c] * k
        return (self.x + dx, self.y + dy)

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
        return (self.x, self.y)

    def flip(self):
        self.state = (self.state + 1) % 2
        return self

    # def __hash__(self):
    #     return self.position

    def __repr__(self):
        return "<{}: {} state={} ({}) history={}>".format(
            self.__class__.__name__, self.position,
            self.state, self.STATES[self.state],
            self.states)


def parse_line(line: str) -> tuple:
    line = str(line)
    tokens = []
    while line:
        for l in (2, 1, 0):
            assert l > 0, f"Wrong code of len {l} at the beginning: {line}"
            if line[:l] in Tile.DIRECTIONS:
                tokens.append(line[:l])
                line = line[l:]
                break

    return tuple(tokens)


def demo_tile():
    t = Tile((0, 0))
    print(t)
    for _ in range(2):
        t.flip()
        print(t)


def find_tile(path: tuple) -> Tile:
    """Return tile reached from the reference tile"""

    if DEBUG:
        print(f"\nPath: {path}")

    tiles = {}

    xy = (0, 0)
    tile = tiles.setdefault(xy, Tile(xy))

    if DEBUG:
        print(f"Startng tile: {tile}")

    for code in path:
        xy = tile.neighbour(code)
        tile = tiles.setdefault(xy, Tile(xy))
        if DEBUG:
            print(f"Current tile: {tile}")

    if DEBUG:
        print(f"Reached tile: {tile}")

    return tile



def flip_tiles(tiles: dict) -> None:

    flippable = []

    # first check rule 1 and create new surrounding white tiles
    # Any black tile with zero or more than 2 black tiles immediately adjacent to it is flipped to white.
    for pos in list(tiles.keys()):
        tile = tiles[pos]

        if tile.is_black():
            neighbours = []
            for d in tile.DIRECTIONS:
                xy = tile.neighbour(d)
                neigh = tiles.setdefault(xy, Tile(xy))
                neighbours.append(neigh)
            n_blacks = count_black_tiles(neighbours)
            if n_blacks == 0 or n_blacks > 2:
                flippable.append(tile)

    # Any white tile with exactly 2 black tiles immediately adjacent to it is flipped to black.
    for pos, tile in tiles.items():
        if tile.is_white():
            neighbours = []
            for d in tile.DIRECTIONS:
                xy = tile.neighbour(d)
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
    paths = [parse_line(line) for line in ["esew", "nwwswee"]]
    for path in paths:
        tile = find_tile(path)


#demo_find_tile()
#exit(100)


def count_black_tiles(tiles: Union[dict, list]) -> int:
    if isinstance(tiles, dict):
        return sum(t.is_black() for t in tiles.values())
    else:
        return sum(t.is_black() for t in tiles)


def plot(tiles):
    xs, ys = [], []
    for (x, y) in tiles.keys():
        xs.append(x)
        ys.append(y)
    xspan = utils.minmax(xs)
    yspan = utils.minmax(ys)

    xspan = (xspan[0], xspan[1]+1)
    yspan = (yspan[0], yspan[1]+1)

    ywidth = (yspan[1] - yspan[0])
    stripes = [ ' \ /'  * ywidth + ' \\', ' / \\' * ywidth + ' /' ]
    indents = ["| ", "  | "]

    print(stripes[1])
    for i, x in enumerate(range(xspan[0], xspan[1])):
        row = []
        for y in range(yspan[0], yspan[1]):
            xy = (x, y)
            tile = tiles.get(xy, Tile(xy))
            ch = 'B' if tile.is_black() else ' '
            row.append(ch)
        print(indents[i%2] + " | ".join(row) + " |")
        #if x+1 < xspan[1]:
        print(stripes[i%2])

    pass


def solve_p1(lines: List[str], do_part=1) -> int:
    """Solution to the 1st part of the challenge"""
    paths = [parse_line(line) for line in lines]

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
