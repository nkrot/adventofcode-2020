#!/usr/bin/env python

# # #
#
#

import re
import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


class Mask(object):

    def __init__(self, mask):
        self.value = mask

    def __call__(self, num):
        bnum = f'{int(num):036b}'
        newnums = [n if m == 'X' else m
                   for n, m in zip(list(bnum), list(self.value))]
        newnum = "".join(newnums)
        return self.to_int(newnum)

    def to_int(self, bnum):
        return int(bnum, 2)


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    mem = {}

    for line in lines:
        tokens = line.strip().split()

        if tokens[0] == 'mask':
            mask = Mask(tokens[2])

        elif tokens[0].startswith('mem['):
            m = re.search(r'mem\[(\d+)\]', line)
            key = int(m.group(1))
            mem[key] = mask(tokens[2])

    s = sum(mem.values())

    return s


def solve_p2(data):
    """Solution to the 2nd part of the challenge"""
    # TODO
    pass


text_1 = """mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X
mem[8] = 11
mem[7] = 101
mem[8] = 0"""

tests = [
    (text_1.split('\n'), 101+64, -1),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(inp)
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(inp)
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '14'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 6513443633260
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    # print(f"--- Day {day} p.2 ---")
    # exp2 = -1
    # res2 = solve_p2(lines)
    # print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
