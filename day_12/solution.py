#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List, Tuple
import math
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def parse_commands(lines: List[str]) -> List[Tuple[str, int]]:
    commands = []
    for line in lines:
        line = line.strip()
        cmd, arg = line[:1], int(line[1:])
        commands.append((cmd, arg))
    return commands


class Vektor(object):

    @classmethod
    def base(cls, d):
        if d == 'N':
            x, y = [0, 1]
        elif d == 'S':
            x, y = [0, -1]
        elif d == 'E':
            x, y = [1, 0]
        elif d == 'W':
            x, y = [-1, 0]
        return cls(x, y)

    def __init__(self, x, y, d=None):
        self.x = x
        self.y = y
        self.d = d
        if d:
            assert d in 'NSWE', f"Wrong value of direction: {d}"

    def __add__(self, other):
        return self.__class__(self.x+other.x, self.y+other.y, self.d)

    def __mul__(self, n):
        return self.__class__(self.x*n, self.y*n, self.d)

    def __rmul__(self, n):
        return self * n

    def __rmatmul__(self, other):
        res = other @ [self.x, self.y]
        # surprise! have to use rint(), because int(1.) outputs 0.
        # this happens because 1. is actually 0.9999999
        res = np.rint(res)
        return self.__class__(int(res[0]), int(res[1]), self.d)

    def __repr__(self):
        return "<{}: x={}, y={}, d={}>".format(
            self.__class__.__name__, self.x, self.y, self.d)

    def l1_distance(self):
        """L1/Manhattan distance to the origin (0,0)"""
        return abs(self.x) + abs(self.y)

    __array_priority__ = 10000


def solve_p1(data: List[str], ship: dict = None) -> int:
    """Solution to the 1st part of the challenge"""
    ship = Vektor(0, 0, 'E')

    commands = parse_commands(data)

    for cmd, arg in commands:

        if cmd == 'F':
            # moving forward can be expressed as moving in the direction
            # in where the ship points.
            cmd = ship.d

        if cmd in 'NSEW':
            ship = ship + Vektor.base(cmd) * arg

        # rotation
        if cmd == 'L':
            cmd, arg = 'R', 360 - arg
        if cmd == 'R':
            ribbon = 'NESW' * 2
            i = ribbon.index(ship.d) + arg//90
            ship.d = ribbon[i]

    return ship.l1_distance()


def solve_p2(data: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    commands = parse_commands(data)

    ship = Vektor(0, 0, 'E')
    waypoint = Vektor(10, 1)

    if DEBUG:
        print("Ship:", ship)
        print("Wayp:", waypoint)

    for cmd, arg in commands:
        if DEBUG:
            print("\nCommand:", (cmd, arg))

        # move ship towards the waypoint
        if cmd in 'F':
            ship = ship + arg * waypoint

        if cmd in 'NSEW':
            waypoint = waypoint + arg * Vektor.base(cmd)

        if cmd in 'LR':
            # rotation is counter-clockwise, therefore we express rotation
            # to the right in terms of rotation to the left.
            arg = 360 - arg if cmd == 'R' else arg

            # Rotation by any angle alpha means
            # x2 = cos(alpha)*x1 - sin(alpha) * y1
            # y2 = sin(alpha)*x1 + cos(alpha) * y1
            # which is equivalent in the linear form to matrix multiplcation
            # [[cos(alpha), sin(alpha)]      [[x1],
            #  [sin(alpha), cos(alpha)]]      [y1]]

            degr = math.radians(arg)  # surprise! trig in math speaks radians
            rotm = np.array([[math.cos(degr), - math.sin(degr)],
                             [math.sin(degr), math.cos(degr)]])
            waypoint = rotm @ waypoint

        if DEBUG:
            print("Ship:", ship)
            print("Wayp:", waypoint)

    return ship.l1_distance()


text_1 = """F10
N3
F7
R90
F11"""


tests = [
    (text_1.split('\n'), 17 + 8,  214 + 72),
    ("R90\nF1\nR90\nR90\nR90".split('\n'), 1, 11),
    ("L90\nF1\nL90\nR90\nR90".split('\n'), 1, 11),
    ("R180\nF1\nL180".split('\n'), 1, 11),
    ("R270\nF1\nL270".split('\n'), 1, 11),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(inp)
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(inp)
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '12'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 1482
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 48739
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
