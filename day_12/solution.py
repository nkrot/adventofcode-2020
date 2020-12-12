#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = not True


def move(operation: Tuple[str, int], obj: dict):
    """Update the coordinates of obj according to the the instruction"""
    name, arg = operation

    k = 1 if name in 'NE' else -1

    if name in 'NS':
        obj['ns'] += arg * k
    if name in 'WE':
        obj['we'] += arg * k


def parse_commands(lines: List[str]) -> List[Tuple[str, int]]:
    commands = []
    for line in lines:
        line = line.strip()
        cmd, arg = line[:1], int(line[1:])
        commands.append((cmd, arg))
    return commands


def solve_p1(data: List[str], ship: dict = None) -> int:
    """Solution to the 1st part of the challenge"""
    ship = {'d': 'E', 'ns': 0, 'we': 0}

    commands = parse_commands(data)

    for cmd, arg in commands:

        if cmd == 'F':
            # moving forward can be expressed as moving in the direction
            # where the ship points.
            cmd = ship['d']

        if cmd in 'NSEW':
            move((cmd, arg), ship)

        # rotation
        if cmd == 'L':
            cmd, arg = 'R', 360 - arg
        if cmd == 'R':
            steps = ['N', 'E', 'S', 'W'] * 2
            while steps[0] != ship['d']:
                steps.pop(0)
            ship['d'] = steps[arg//90]

    return abs(ship['ns']) + abs(ship['we'])


def solve_p2(data: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    commands = parse_commands(data)

    ship = {'we': 0, 'ns': 0, 'd': 'E'}
    # waypoint coordinates are always relative to the ship
    waypoint = {'we': 10, 'ns': 1}

    if DEBUG:
        print("Ship:", ship)
        print("Wayp:", waypoint)

    for cmd, arg in commands:
        if DEBUG:
            print("\nCommand:", (cmd, arg))

        # move ship towards the waypoint
        if cmd in 'F':
            ship['ns'] += arg * waypoint['ns']
            ship['we'] += arg * waypoint['we']

        # rotate waypoint around the ship
        if cmd == 'L':
            cmd, arg = 'R', 360 - arg
        if cmd == 'R':
            if arg == 90:
                waypoint['we'], waypoint['ns'] = waypoint['ns'], -waypoint['we']
            elif arg == 90*2:
                waypoint['we'], waypoint['ns'] = -waypoint['we'], -waypoint['ns']
            elif arg == 90*3:
                waypoint['we'], waypoint['ns'] = -waypoint['ns'], waypoint['we']
            else:
                raise ValueError(f"Wrong angle: {arg}")

        # transfer the waypoint relative to the ship
        if cmd in 'NSEW':
            move((cmd, arg), waypoint)

        if DEBUG:
            print("Ship:", ship)
            print("Wayp:", waypoint)

    return abs(ship['ns']) + abs(ship['we'])


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
