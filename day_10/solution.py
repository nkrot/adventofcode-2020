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


def solve_p2(numbers: List[int]) -> int:
    """Solution to the 2nd part of the challenge"""
    nums = sorted(numbers)
    if nums[0] > 0:
        nums.insert(0, 0)
    nums.append(nums[-1]+3)
    if DEBUG:
        print(nums)
    mx = nums[-1]
    mat = [[0] * (1+mx) for i in range(mx+1)]

    # connect adapters according to rules
    for r, nr in enumerate(nums):
        if r == 0:
            continue
        for c in {max(r-3, 0), max(r-2, 0), max(r-1, 0)}:
            nc = nums[c]
            diff = nr - nc
            assert diff != 0, "Shit happened"
            if diff < 4:
                # print(nc, nr, diff)
                mat[nr][nc] += 1

    # update the number of previous adapters
    for ri, row in enumerate(mat):
        n_ins = sum(row)
        if DEBUG:
            print(n_ins, ri, row)
        if n_ins > 1:
            # check next 3 rows
            for rj in range(ri+1, ri+4):
                if rj < len(mat) and mat[rj][ri] > 0:
                    mat[rj][ri] += (n_ins-1)

    if DEBUG:
        show(mat)

    return sum(mat[-1])


def show(m):
    lines = []
    for ri, r in enumerate(m):
        row = [f'{n:>2}' for n in r]
        lines.append(f"{ri:<3}:" + ", ".join(row))
    print("\n".join(lines))


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
    (text1.split('\n'), 7*5, 8),
    (utils.load_input('test_input.1.txt'), 22*10, 19208)
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        inp = utils.to_numbers(inp)

        res1 = solve_p1(inp)
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(inp)
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '10'
    lines = utils.to_numbers(utils.load_input())

    print(f"--- Day {day} p.1 ---")
    exp1 = 1700
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 12401793332096
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
