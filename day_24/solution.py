#!/usr/bin/env python

# # #
#
#

import re
import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = not True


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


# def flip_tiles(paths: List[tuple]) -> List[Tile]:
#     tiles = []
#     return tiles


def demo_find_tile():
    paths = [parse_line(line) for line in ["esew", "nwwswee"]]
    for path in paths:
        tile = find_tile(path)


#demo_find_tile()
#exit(100)


def solve_p1(lines: List[str]) -> int:
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

    return sum(t.state for t in tiles.values())


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    # TODO
    return 0


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
    (text_1.split('\n'), 10+5-5, None),
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
    day = '24'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 420
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = -1
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
