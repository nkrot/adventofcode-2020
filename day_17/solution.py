#!/usr/bin/env python

# # #
# similar to day_11
#

import re
import os
import sys
from typing import List, Tuple, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils
from aoc import Board, Square, Command


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


class EnergySource(object):  # TODO: rename to Board3D
    PIECE = Cube

    OFFSETS_AROUND = [
        (-1, 0, 0), (-1, +1, 0), (0, +1, 0), (+1, +1, 0),
        (+1, 0, 0), (+1, -1, 0), (0, -1, 0), (-1, -1, 0),
        (0,  0, -1),
        (-1, 0, -1), (-1, +1, -1), (0, +1, -1), (+1, +1, -1),
        (+1, 0, -1), (+1, -1, -1), (0, -1, -1), (-1, -1, -1),
        (0,  0, +1),
        (-1, 0, +1), (-1, +1, +1), (0, +1, +1), (+1, +1, +1),
        (+1, 0, +1), (+1, -1, +1), (0, -1, +1), (-1, -1, +1)
    ]

    @classmethod
    def from_text(cls, text: str):
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return cls.from_lines(lines)

    @classmethod
    def from_lines(cls, lines: List[str]):
        board = cls()
        z = 0
        for x, line in enumerate(lines):
            for y, char in enumerate(line):
                piece = cls.PIECE(char)
                piece.position = (x, y, z)
                board.add(piece)
        return board

    def __init__(self):
        self.items = {}
        self.extendable = True

    def __getitem__(self, xyz: Tuple[int, int, int]) -> PIECE:
        if xyz not in self.items and self.extendable:
            item = self.PIECE()
            item.position = xyz
            self.add(item)
        return self.items.get(xyz, None)

    def __setitem__(self, xyz, piece):
        assert isinstance(piece, self.PIECE), "Wrong type"
        self.items[xyz] = piece

    def spans(self):
        """Compute min and max values in every dimension
        Return: [(minx, maxx), (miny, maxy), (minz, maxz)]
        """
        xs, ys, zs = [], [], []
        for x, y, z in self.items.keys():
            xs.append(x)
            ys.append(y)
            zs.append(z)
        spans = [utils.minmax(xs), utils.minmax(ys), utils.minmax(zs)]
        return spans

    def inflate(self):
        """Extend in every direction 1 layer on each side"""
        spans = self.spans()
        # collect coordinates that are at the edges
        edge_points = []
        for x, y, z in self.items.keys():
            if x in spans[0] or y in spans[1] or z in spans[1]:
                edge_points.append((x,y,z))
        # checking neighbors of the edge points has a side effect that new
        # neighbors will be created.
        for xyz in edge_points:
            self.neighbours(xyz)

    def add(self, item: PIECE):
        assert isinstance(item, self.PIECE), "Wrong type"
        self[item.position] = item

    def neighbours(self, obj: Union[PIECE, tuple]) -> List[PIECE]:
        offsets = self.OFFSETS_AROUND
        if isinstance(obj, self.PIECE):
            x, y, z = obj.position
        else:
            x, y, z = obj
        neighbours = []
        for dx, dy, dz in offsets:
            nxyz = (x + dx, y + dy, z + dz)
            neighbour = self[nxyz]
            if neighbour:
                neighbours.append(neighbour)
        return neighbours

    def cubes(self):
        return list(self.items.values())

    def __str__(self):
        lines = []
        xs, ys, zs = self.spans()
        for z in range(zs[0], zs[1]+1):
            lines.append(f"-- z={z} --")
            for x in range(xs[0], xs[1]+1):
                cells = [str(self[(x,y,z)]) for y in range(ys[0], ys[1]+1)]
                line = "".join(cells)
                lines.append(line)
        return "\n".join(lines)


# If a cube is active and exactly 2 or 3 of its neighbors are also active, the cube remains active. Otherwise, the cube becomes inactive.
# If a cube is inactive but exactly 3 of its neighbors are active, the cube becomes active. Otherwise, the cube remains inactive.

def run_one_cycle(source: EnergySource):
    if DEBUG:
        print("--- Cycle ---")

    source.inflate()
    if DEBUG:
        print(source)

    actions = []
    for cube in source.cubes():
        c_active = sum([int(nc.is_active()) for nc in source.neighbours(cube)])
        if cube.is_active() and c_active not in {2,3}:
            actions.append(deactivate(cube))
        elif cube.is_inactive() and c_active == 3:
            actions.append(activate(cube))

    for action in actions:
        action.execute()

    if DEBUG:
        print(source)


def solve_p1(source: EnergySource) -> int:
    """Solution to the 1st part of the challenge"""

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


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    # TODO
    return 0


text_1 = """.#.
..#
###"""


tests = [
    (text_1, 112, 848),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        if exp1 is not None:
            res1 = solve_p1(EnergySource.from_text(inp))
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '17'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 359
    res1 = solve_p1(EnergySource.from_lines(lines))
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = -1
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
