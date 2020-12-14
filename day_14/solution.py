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

    def to_bin(self, num) -> str:
        return f'{int(num):036b}'

    def to_int(self, bnum: str) -> int:
        return int(bnum, 2)

    def v2(self, num) -> List[int]:
        bnum = self.to_bin(num)
        newnums = [n if m == '0' else m
                   for n, m in zip(list(bnum), list(self.value))]

        # replace X with any posisble value
        addrs = [newnums]
        for _ in range(newnums.count('X')):
            for _ in range(len(addrs)):
                curr = addrs.pop(0)
                pos = curr.index('X')
                for i in {"0", "1"}:
                    upd = list(curr)
                    upd[pos] = i
                    addrs.append(upd)

        # print("--- final ---")
        addrs = [self.to_int("".join(addr)) for addr in addrs]

        return addrs

    def v1(self, num):
        bnum = self.to_bin(num)
        newnums = [n if m == 'X' else m
                   for n, m in zip(list(bnum), list(self.value))]
        newnum = "".join(newnums)
        return self.to_int(newnum)


def parse_lines(lines: List[str]) -> List[tuple]:
    commands = []
    for line in lines:
        tokens = line.strip().split()

        if tokens[0] == 'mask':
            commands.append(('mask', tokens[2]))

        elif tokens[0].startswith('mem['):
            m = re.search(r'mem\[(\d+)\]', line)
            commands.append(("mem", int(m.group(1)), tokens[2]))

    return commands


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    mem = {}
    commands = parse_lines(lines)
    for cmd in commands:
        if cmd[0] == 'mask':
            mask = Mask(cmd[1])
        elif cmd[0] == 'mem':
            mem[cmd[1]] = mask.v1(cmd[2])
    return sum(mem.values())


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    mem = {}
    commands = parse_lines(lines)
    for cmd in commands:
        if cmd[0] == 'mask':
            mask = Mask(cmd[1])
        elif cmd[0] == 'mem':
            for addr in mask.v2(cmd[1]):
                mem[addr] = int(cmd[2])
    return sum(mem.values())


text_1 = """mask = XXXXXXXXXXXXXXXXXXXXXXXXXXXXX1XXXX0X
mem[8] = 11
mem[7] = 101
mem[8] = 0"""


text_2 = """mask = 000000000000000000000000000000X1001X
mem[42] = 100
mask = 00000000000000000000000000000000X0XX
mem[26] = 1"""


tests = [
    (text_1.split('\n'), 101+64, None),
    (text_2.split('\n'), None, 208),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        if exp1 is not None:
            res1 = solve_p1(inp)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '14'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 6513443633260
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 3442819875191
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
