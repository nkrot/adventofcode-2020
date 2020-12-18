#!/usr/bin/env python

# # #
# similar to day_11 but the data structure from day_11 is not appropriate
#

import os
import sys
from typing import List, Union
from collections.abc import Iterable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils
from aoc import Command


DEBUG = False


class activate(Command):

    def __init__(self, obj):
        super().__init__(obj, 'make_active')


class deactivate(Command):

    def __init__(self, obj):
        super().__init__(obj, 'make_inactive')


class Cube(object):

    VALUES = ('#', '.')

    def __init__(self, value=None):
        self.value = value or '.'
        self.position = None

    def is_active(self):
        return self.value == '#'

    def is_inactive(self):
        return self.value == '.'

    def make_active(self):
        self.value = '#'

    def make_inactive(self):
        self.value = '.'

    def __str__(self):
        return str(self.value)


class Coord(object):

    OFFSETS_AROUND = [None, None, None, None, None]

    @staticmethod
    def generate_combinations(ndims):
        # print(f"Generating combinations for {ndims}")
        queue = [[]]
        for ndim in range(ndims):
            for _ in range(len(queue)):
                item = queue.pop(0)
                for n in {-1, 0, 1}:
                    queue.append(list(item) + [n])
        return queue

    @classmethod
    def offsets_around(cls, ndims):
        if cls.OFFSETS_AROUND[ndims] is None:
            combinations = cls.generate_combinations(ndims)
            removed = combinations.pop(0)  # to remove all zeroes
            # print(f"removed: {removed}")
            cls.OFFSETS_AROUND[ndims] = [cls(xs) for xs in combinations]
        return cls.OFFSETS_AROUND[ndims]

    def __init__(self, *args):
        if isinstance(args[0], Iterable):
            self.values = tuple(args[0])
        else:
            self.values = tuple(args)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, pos: int) -> int:
        return self.values[pos]

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other):
        return (len(self) == len(other)
                and all(v1 == v2 for v1, v2 in zip(self, other)))

    def __hash__(self):
        return hash(tuple(self.values))

    def __add__(self, other):
        assert len(self) == len(other), "Mismatching dimensions"
        return self.__class__([v1+v2 for v1, v2 in zip(self, other)])

    def __str__(self):
        return ",".join([str(v) for v in self.values])

    def neighbours(self):
        coords = []
        for offset in self.offsets_around(len(self)):
            coords.append(self + offset)
        return coords


class EnergySource(object):
    PIECE = Cube

    @classmethod
    def from_text(cls, text: str, ndims=3):
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return cls.from_lines(lines, ndims)

    @classmethod
    def from_lines(cls, lines: List[str], ndims=3):
        board = cls()
        for x, line in enumerate(lines):
            for y, char in enumerate(line):
                piece = cls.PIECE(char)
                if ndims == 3:
                    piece.position = Coord(x, y, 0)
                elif ndims == 4:
                    piece.position = Coord(x, y, 0, 0)
                else:
                    raise ValueError(
                        f"Wrong values of ndims argument: {ndims}")
                board.add(piece)
        return board

    def __init__(self):
        self.items = {}
        self.expandable = False

    def __getitem__(self, coord: Coord) -> PIECE:
        item = self.items.get(coord, None)
        if item is None and self.expandable:
            item = self.PIECE()
            item.position = coord
            self.add(item)
        return item

    def __setitem__(self, xyz, piece):
        assert isinstance(piece, self.PIECE), "Wrong type"
        self.items[xyz] = piece

    def spans(self):
        """Compute min and max values in every dimension
        Return: [(minx, maxx), (miny, maxy), (minz, maxz)]
        """
        dims = None
        for coord in self.items.keys():
            if dims is None:
                dims = [[] for _ in range(len(coord))]
            for pos, val in enumerate(coord):
                dims[pos].append(val)
        spans = [utils.minmax(xs) for xs in dims]
        return spans

    def inflate(self):
        """Extend in every direction 1 layer on each side"""
        # collect coordinates that are at any of the faces (either at edge
        # or in the middle)
        spans = self.spans()
        edge_points = []
        for coord in self.items.keys():
            if any(x in spans[pos] for pos, x in enumerate(coord)):
                edge_points.append(coord)
        # Checking neighbors of the edge points will create missing neighbours.
        oldvalue, self.expandable = self.expandable, True
        for coord in edge_points:
            self.neighbours(coord)
        self.expandable = oldvalue

    def add(self, item: PIECE):
        assert isinstance(item, self.PIECE), "Wrong type"
        self[item.position] = item

    def neighbours(self, obj: Union[PIECE, tuple]) -> List[PIECE]:
        if isinstance(obj, self.PIECE):
            coord = obj.position
        else:
            coord = obj
        neighbours = []
        for ncoord in coord.neighbours():
            neighbour = self[ncoord]
            if neighbour:
                neighbours.append(neighbour)
        return neighbours

    def cubes(self):
        return list(self.items.values())

    def __str__(self):
        # TODO: how to make it work for 4d case?
        lines = []
        spans = self.spans()
        assert len(spans) == 3, "Can handle 3 dimensions only"
        xs, ys, zs = spans
        for z in range(zs[0], zs[1]+1):
            lines.append(f"-- z={z} --")
            for x in range(xs[0], xs[1]+1):
                cells = [str(self[Coord(x, y, z)])
                         for y in range(ys[0], ys[1]+1)]
                line = "".join(cells)
                lines.append(line)
        return "\n".join(lines)


def run_one_cycle(source: EnergySource):
    if DEBUG:
        print("--- Cycle ---")

    source.inflate()
    if DEBUG:
        print(f"Number cubes before inflation: {len(source.cubes())}")
        print(" ++ inflated (1 layer added around) ++ ")
        print(source)
        print(f"Number cubes after inflation: {len(source.cubes())}")

    actions = []
    for cube in source.cubes():
        c_active = sum([int(nc.is_active()) for nc in source.neighbours(cube)])
        if cube.is_active() and c_active not in {2, 3}:
            actions.append(deactivate(cube))
        elif cube.is_inactive() and c_active == 3:
            actions.append(activate(cube))

    for action in actions:
        action.execute()

    if DEBUG:
        print(" ++ result of this cycle ++")
        print(source)


def solve(source: EnergySource) -> int:
    if DEBUG:
        print("--- Initial State ---")
        print(source)

    n_cycles = 6
    for i in range(n_cycles):
        run_one_cycle(source)

    c_active = 0
    for cube in source.cubes():
        if cube.is_active():
            c_active += 1

    return c_active


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    lines = [ln.strip() for ln in lines]
    lines = [ln for ln in lines if ln]
    source = EnergySource.from_lines(lines, 3)
    return solve(source)


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    source = EnergySource.from_lines(lines, 4)
    return solve(source)


text_1 = """.#.
..#
###"""


tests = [
    (text_1, 112, 848),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        lines = inp.split('\n')

        if exp1 is not None:
            res1 = solve_p1(lines)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(lines)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '17'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 359
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 2228
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
