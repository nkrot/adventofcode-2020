#!/usr/bin/env python

# # #
#
# TODO:
# Shunting-Yard algorithm
# Reverse Polish Notation

import re
import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


class Operator(object):

    def __init__(self, name):
        self.name = name
        self.args = []
        self.arity = 2

    def set_args(self, arg1, arg2):
        self.args.append(arg1)
        self.args.append(arg2)

    def is_full(self):
        return len(self.args) == self.arity

    def compute(self, *args):
        if args:
            lhs, rhs = args
        else:
            lhs, rhs = self.args
        lhs = lhs.compute()
        rhs = rhs.compute()
        if self.name == '+':
            res = lhs + rhs
        elif self.name == '-':
            res = lhs - rhs
        elif self.name == '*':
            res = lhs * rhs
        return Value(res)

    def to_s(self, level=0):
        offset = ' ' * level
        offset = '\n' + offset
        lhs = self.args[0].to_s(level+1) if len(self.args) > 0 else ''
        rhs = self.args[1].to_s(level+1) if len(self.args) > 1 else ''
        return "<{}: {} {}LEFT={} {}RIGHT={}>".format(
            self.__class__.__name__,
            self.name, offset, lhs, offset, rhs)

    def __repr__(self):
        return self.to_s()


class Value(object):

    def __init__(self, val):
        self.value = int(val)

    def to_s(self, level=0):
        return "<{}: {}>".format(self.__class__.__name__,
                                 self.value)

    def compute(self):
        return self.value

    def __repr__(self):
        return self.to_s()


class OpenParen(object):

    def __repr__(self):
        return self.to_s()

    def to_s(self, level=0):
        return "<{}: ( >".format(self.__class__.__name__)


class CloseParen(object):

    def to_s(self, level=0):
        return "<{}: ) >".format(self.__class__.__name__)

    def __repr__(self):
        return self.to_s()


class NaiveCalculator(object):
    """ Recursive calculator of math expressions w/o precedence rules
    that does not build any computational graph/ast

    """

    def __init__(self, tokens, level=0):
        self.tokens = tokens

    def execute(self):

        while len(self.tokens) > 1:
            self.tokens = [t for t in self.tokens if t]

            # ex: 1 + 2 * 3 + 4 * 5 + 6
            for i, op in enumerate(self.tokens):
                if DEBUG:
                    print(f"Looking at [{i}]: {op}")

                if isinstance(op, Operator):
                    lhs = self.tokens[i-1]
                    rhs = self.tokens[i+1]

                    if isinstance(lhs, Value) and isinstance(rhs, Value):
                        self.tokens[i] = op.compute(lhs, rhs)
                        self.tokens[i-1] = None
                        self.tokens[i+1] = None
                        break

                if (isinstance(op, CloseParen)
                      and isinstance(self.tokens[i-2], OpenParen)):
                    self.tokens[i] = None
                    self.tokens[i-2] = None
                    break

        if DEBUG:
            print(self.tokens)

        return self.tokens.pop(0)


def lexeme(token: str):
    if token in {'+', '-', '*'}:
        res = Operator(token)
    elif re.match(r'\d+$', token):
        res = Value(token)
    elif token == '(':
        res = OpenParen()
    elif token == ')':
        res = CloseParen()
    else:
        raise ValueError(f"Unrecognized token: {token}")
    return res


def lexemes(line: str) -> list:
    line = re.sub(r'([()*+-])', ' \g<1> ', line.strip())
    return [lexeme(t) for t in line.split()]


def evaluate_math_expression(line: str) -> int:
    calculator = NaiveCalculator(lexemes(line))
    res = calculator.execute()
    return res.value


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    results = [evaluate_math_expression(line) for line in lines]
    return sum(results)


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    # TODO
    return 0


tests = [
    ("1 + 2 * 3 + 4 * 5 + 6", 71, None),
    ("(1 + 2 * 3 + 4 * 5 + 6)", 71, None),
    ("(1 + 2 * 3 + 4 * 5) + 6", 71, None),
    ("(1 + 2 * 3 + 4) * 5 + 6", 71, None),
    ("1 + (2 * 3) + (4 * (5 + 6))", 51, None),

    ("2 * 3 + (4 * 5)", 26, None),
    ("5 + (8 * 3 + 9 + 3 * 4 * 3)", 437, None),
    ("5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))", 12240, None),

    ("((2 + 4 * 9))", 54, None),
    ("( 2 + 3 + 4) * ( 2 + 3 )", 45, None),

    ("((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6)", 6810, None),

    ("((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2", 13632, None)

]

def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        if exp1 is not None:
            res1 = solve_p1([inp])
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2([inp])
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '18'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 31142189909908
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = -1
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
