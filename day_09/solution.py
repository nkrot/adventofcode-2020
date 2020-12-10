#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def sum_of_two(target: int,
               numbers: List[int], sidx, eidx) -> Optional[Tuple[int, int]]:
    for i in range(sidx, eidx):
        for j in range(i+1, eidx):
            if numbers[i] + numbers[j] == target:
                return (numbers[i], numbers[j])
    return None


def sum_to_target1(target: int,
                   numbers: List[int], eidx) -> Optional[List[int]]:
    """Find contiguous sequence of numbers from <numbers> between [0,eidx)
    that sum up to given <target>. The sequence must contains at least 2 items.
    Return the sequence found.
    Naive approach."""
    for li in range(0, eidx):
        t = numbers[li]
        for ri in range(li+1, eidx):
            t += numbers[ri]
            if t == target:
                return numbers[li:1+ri]
            elif t > target:
                break
    return None


def sum_to_target(target: int,
                  numbers: List[int], eidx) -> Optional[List[int]]:
    """Find contiguous sequence of numbers from <numbers> between [0,eidx)
    that sum up to given <target>. The sequence must contains at least 2 items.
    Return the sequence found.
    Sliding window approach."""

    li, ri = 0, 0
    t = numbers[li]
    while ri < eidx:
        if t < target or li == ri:
            ri += 1
            t += numbers[ri]
        elif t > target:
            t -= numbers[li]
            li += 1
        elif t == target:
            return numbers[li:ri+1]
    return None


def solve_p1(numbers: List[int],
             preamble_length: int = 25) -> Optional[Tuple[int, int]]:
    """Solution to the 1st part of the challenge"""
    for idx in range(preamble_length, len(numbers)):
        num = numbers[idx]
        args = sum_of_two(num, numbers, idx-preamble_length, idx)
        if DEBUG:
            print("[{}]={} is sum of {}".format(idx, num, args))
        if not args:
            return (idx, num)
    return None


def solve_p2(numbers: List[str], preamble_length: int = 25) -> Optional[int]:
    """Solution to the 2nd part of the challenge"""
    inv_idx, inv_num = solve_p1(numbers, preamble_length)
    seq = sum_to_target(inv_num, numbers, inv_idx)
    return sum(utils.minmax(seq))


text_1 = """35
20
15
25
47
40
62
55
65
95
102
117
150
182
127
219
299
277
309
576"""


tests = [
    (text_1.split('\n'), 127, 15+47),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        inp = utils.to_numbers(inp)

        res1 = solve_p1(inp, 5)
        print(f"T1.{tid}:", res1[1] == exp1, exp1, res1)

        res2 = solve_p2(inp, 5)
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '09'
    lines = utils.to_numbers(utils.load_input())

    print(f"--- Day {day} p.1 ---")
    exp1 = 88311122
    res1 = solve_p1(lines)
    print(exp1 == res1[1], exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 13549369
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
