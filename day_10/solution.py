#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def solve_p1(numbers: List[int]) -> int:
    """Solution to the 1st part of the challenge"""
    nums = sorted(numbers + [0])
    counts = defaultdict(int)
    for i in range(1, len(nums)):
        diff = nums[i] - nums[i-1]
        if DEBUG:
            print(nums[i-1], nums[i], diff)
        counts[diff] += 1
    counts[3] += 1  # device's built-in adapter
    if DEBUG:
        print(counts)
    return counts[1] * counts[3]


def solve_p2(data):
    """Solution to the 2nd part of the challenge"""
    # TODO
    pass


text1 = """16
10
15
5
1
11
7
19
6
12
4"""


tests = [
    (text1.split('\n'), 7*5, -1),
    (utils.load_input('test_input.1.txt'), 22*10, -1)
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        inp = utils.to_numbers(inp)

        res1 = solve_p1(inp)
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        # res2 = solve_p2(inp)
        # print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '10'
    lines = utils.to_numbers(utils.load_input())

    print(f"--- Day {day} p.1 ---")
    exp1 = 1700
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    # print(f"--- Day {day} p.2 ---")
    # exp2 = 360
    # res2 = solve_p2(lines)
    # print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
