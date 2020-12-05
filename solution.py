#!/usr/bin/env python

# # #
# template script
# 1) copy it under a day-related directory
# 2) fill in TODOs
#

import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def solve_p1():
    """Solution to the 1st part of the challenge"""
    # TODO
    pass


def solve_p2():
    """Solution to the 2nd part of the challenge"""
    # TODO
    pass


tests = [
    # (input, exp1, exp2),
    # TODO
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(inp)
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(inp)
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = 'DD'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 542
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 360
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
