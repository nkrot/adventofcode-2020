#!/usr/bin/env python

# # #
#
#

import re
import os
import sys
from typing import List
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def play_one_round(cups, curridx=0):
    # print("Cups", cups)
    n_out = 3

    curr = cups.pop(0)
    taken = []
    for _ in range(n_out):
        taken.append(cups.pop(0))
    # print("Taken:", taken)
    # print("cups shortened:", cups)

    # find destination cup
    destidx = None
    for dest in range(curr-1, max(curr-2-n_out, 0), -1):
        if dest not in taken:
            destidx = cups.index(dest)
            break

    if destidx is None:
        dest = max(cups)
        destidx = cups.index(dest)

    # print(destidx, dest)

    # place taken out cups after the destination cup
    while taken:
        cups.insert(destidx+1, taken.pop())

    # print(cups)

    cups.append(curr)


def solve_p1(line, n_moves=100) -> str:
    """Solution to the 1st part of the challenge"""
    cups = [int(ch) for ch in list(line)]

    for n in range(n_moves):
        play_one_round(cups)

    while cups[0] != 1:
        cups.append(cups.pop(0))

    return "".join(str(n) for n in cups[1:])


def solve_p2(line, n_moves=10000000) -> int:
    """Solution to the 2nd part of the challenge"""
    return 0

    print("This algorithm will take a day to complete")

    size = 1000000
    cups = [0] * size

    _cups = [int(ch) for ch in list(line)]
    for idx, n in enumerate(_cups):
        cups[idx] = n

    n = max(cups)
    for idx in range(len(line), size):
        n += 1
        cups[idx] = n

    for n in range(n_moves):
        play_one_round(cups)

    idx = cups.index(1)
    idx1 = idx+1 % size
    idx2 = idx+2 % size

    return cups[idx1] * cups[idx2]


tests = [
    ("389125467", "92658374", None),  # 10 moves
    ("389125467", "67384529", None),  # 100 moves
    ("389125467", None, 934001 * 159792)  # 10 mln moves
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        n_moves = 10 if tid == 0 else 100
        if exp1 is not None:
            res1 = solve_p1(inp, n_moves)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '23'
    # lines = utils.load_input()
    inp = "598162734"

    print(f"--- Day {day} p.1 ---")
    exp1 = "32658947"
    res1 = solve_p1(inp)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = -1
    res2 = solve_p2(inp)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    # run_real()
