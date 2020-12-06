#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List, Union
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def group_lines(data: Union[str,List[str]]) -> List[List[str]]:
    groups = [[]]
    if isinstance(data, str):
        data = [ln.strip() for ln in data.split('\n')]
    for ln in data:
        if ln:
            groups[-1].append(ln)
        else:
            groups.append([])
    return groups


def solve_p1(data: Union[str,List[str]]) -> int:
    """Solution to the 1st part of the challenge"""
    cnt = 0
    groups = group_lines(data)
    if DEBUG:
        print(groups)
    for grp in groups:
        questions = set()
        for line in grp:
            questions.update(list(line))
        if DEBUG:
            print(questions)
        cnt += len(questions)
    return cnt


def solve_p2(data):
    """Solution to the 2nd part of the challenge"""
    cnt = 0
    groups = group_lines(data)
    if DEBUG:
        print(groups)
    for grp in groups:
        counts = defaultdict(int)
        for line in grp:
            for char in line:
                counts[char] += 1
        if DEBUG:
            print(counts)
        for k, v in counts.items():
            if v == len(grp):
                cnt += 1
    return cnt


text_1 = \
"""abc

a
b
c

ab
ac

a
a
a
a

b"""

tests = [
    (text_1, 11, 6),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(inp)
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(inp)
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '06'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 6596
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 3219
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
