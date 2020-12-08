#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List, Tuple
from copy import copy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


class Command(object):

    NAMES = {'nop', 'jmp', 'acc'}

    @classmethod
    def from_lines(cls, lines: List[str]) -> List['Command']:
        lines = [line.strip() for line in lines if line]
        commands = [cls.from_line(line) for line in lines]
        return commands

    @classmethod
    def from_line(cls, text: str):
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

    def __copy__(self):
        """make a copy of self, resetting some fields"""
        return self.__class__(self.name, self.argument)

    def __call__(self, argument: int, address: int) -> Tuple[int, int]:
        """Execute current command with given argument and start address.
        Return
        ------
        A tuple of two values:
          1) new value of the argument
          2) address of the next command to execute
        """
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


def solve_p1(commands: List[Command]) -> Tuple[int, int]:
    """Solution to the 1st part of the challenge"""
    res, ecode = 0, 0
    accum, addr = 0, 0
    while addr < len(commands):
        cmd = commands[addr]
        accum, addr = cmd(accum, addr)
        if cmd.times > 1:
            ecode = 1
            break
        res = accum
    return res, ecode


def solve_p2(commands: List[Command]) -> int:
    """Solution to the 2nd part of the challenge"""
    res = None

    def fix(cmd: Command):
        """Fix given command in place"""
        if cmd.name == 'jmp':
            cmd.name = 'nop'
        elif cmd.name == 'nop':
            cmd.name = 'jmp'

    # indices of potentially corrupted commands
    cmd_idxs = [idx for idx, cmd in enumerate(commands)
                if cmd.name in {'nop', 'jmp'}]

    for cmd_idx in cmd_idxs:
        if DEBUG:
            print(f"--- fixing command at #{cmd_idx} --")
        fixed_commands = [copy(cmd) for cmd in commands]
        fix(fixed_commands[cmd_idx])
        val, ecode = solve_p1(fixed_commands)
        if ecode == 0:
            res = val
            break

    return res


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
    (text1.split('\n'), 5, 8),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(Command.from_lines(inp))[0]
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(Command.from_lines(inp))
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '08'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 2014
    res1 = solve_p1(Command.from_lines(lines))
    print(exp1 == res1[0], exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 2251
    res2 = solve_p2(Command.from_lines(lines))
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
