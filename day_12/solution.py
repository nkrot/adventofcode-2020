#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def solve_p1(data: List[str], ship: dict = None) -> int:
    """Solution to the 1st part of the challenge"""
    if not ship:
        ship = {'d': 'E', 'ns': 0, 'we': 0}
    for line in data:
        line = line.strip()
        cmd, arg = line[:1], int(line[1:])
        # print(line, cmd, arg)

        # movement
        if cmd == 'F':
            if ship['d'] == 'N':
                ship['ns'] += arg
            elif ship['d'] == 'S':
                ship['ns'] -= arg
            elif ship['d'] == 'E':
                ship['we'] += arg
            elif ship['d'] == 'W':
                ship['we'] -= arg

        # rotation
        if cmd == 'L':
            cmd, arg = 'R', 360 - arg
        if cmd == 'R':
            steps = ['N', 'E', 'S', 'W'] * 2
            while steps[0] != ship['d']:
                steps.pop(0)
            ship['d'] = steps[arg//90]

        # transfer
        if cmd == 'N':
            ship['ns'] += arg
        elif cmd == 'S':
            ship['ns'] -= arg
        elif cmd == 'E':
            ship['we'] += arg
        elif cmd == 'W':
            ship['we'] -= arg

    return abs(ship['ns']) + abs(ship['we'])


def solve_p2(data):
    """Solution to the 2nd part of the challenge"""
    # TODO
    waypoint = {'ns': 1, 'we': 10}
    pass


text_1 = """F10
N3
F7
R90
F11"""


tests = [
    (text_1.split('\n'), 17 + 8,  214 + 72),
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

    # print(f"--- Day {day} p.2 ---")
    # exp2 = 360
    # res2 = solve_p2(lines)
    # print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
