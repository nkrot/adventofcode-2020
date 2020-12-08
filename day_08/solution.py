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


class Command(object):

    NAMES = ['nop', 'jmp', 'acc']

    @classmethod
    def from_text(cls, text: str):
        fields = text.strip().split()
        assert len(fields) == 2, f"Wrong format: {text}"
        obj = cls(fields[0], int(fields[1]))
        return obj

    def __init__(self, name, arg):
        assert name in self.NAMES, f"Wrong instruction name: {name}"
        self.name = name
        self.argument = arg
        self.times = 0  # number of envocations of this command
        self.debug = DEBUG

    def __call__(self, argument: int, address: int):
        self.times += 1
        if self.name == 'acc':
            argument += self.argument
            address += 1
        elif self.name == 'jmp':
            address += self.argument
        elif self.name == 'nop':
            address += 1
        if self.debug:
            print("{} --> {}".format(repr(self), (argument, address)))
        return argument, address

    def __repr__(self):
        return "{}".format((self.name, self.argument, self.times))


def solve_p1(data: List[str]):
    """Solution to the 1st part of the challenge"""
    res, accum = 0, 0
    commands = [Command.from_text(line.strip()) for line in data if line]
    if DEBUG:
        print(commands)
    addr = 0
    while True:
        cmd = commands[addr]
        accum, addr = cmd(accum, addr)
        if cmd.times > 1:
            break
        res = accum
    return res


def solve_p2(data):
    """Solution to the 2nd part of the challenge"""
    # TODO
    pass


text1 = \
"""nop +0
acc +1
jmp +4
acc +3
jmp -3
acc -99
acc +1
jmp -4
acc +6"""


tests = [
    (text1.split('\n'), 5, -1),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(inp)
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        # res2 = solve_p2(inp)
        # print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '08'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 2014
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    # print(f"--- Day {day} p.2 ---")
    # exp2 = 360
    # res2 = solve_p2(lines)
    # print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
