#!/usr/bin/env python

# # #
#
#

import re
import os
import sys
from typing import List, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


class Operator(object):

    precedence = {'+': 2, '*': 1}  # weird, eh?

    def __init__(self, name):
        self.name = name
        self.args = []
        self.arity = 2

    def __lt__(self, other):
        return self.precedence[self.name] < self.precedence[other.name]

    def __gt__(self, other):
        return self.precedence[self.name] > self.precedence[other.name]

    def compute(self, *args):
        if args:
            lhs, rhs = args
        else:
            lhs, rhs = self.args
        lhs = lhs.compute()
        rhs = rhs.compute()
        if self.name == '+':
            res = lhs + rhs
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

    @property
    def value(self):
        # to unify iterfaces of Operator and Valuex
        return self.name


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

    @property
    def value(self):
        return "("


class CloseParen(object):

    def to_s(self, level=0):
        return "<{}: ) >".format(self.__class__.__name__)

    def __repr__(self):
        return self.to_s()

    @property
    def value(self):
        return ")"


class Stack(object):

    def __init__(self):
        self.items = []

    def __iter__(self):
        return reversed(self.items)

    def __len__(self):
        return len(self.items)

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.items:
            return self.items.pop()
        return None

    def peek(self):
        if self.items:
            return self.items[-1]
        return None


class Queue(object):

    def __init__(self):
        self.items = []

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def enq(self, item):
        self.items.append(item)

    def deq(self, item):
        if self.items:
            return self.items.pop(0)
        return None

    def peek(self):
        if self.items:
            return self.items[0]
        return None


class AdvancedCalculator(object):

    def __init__(self, tokens):
        self.tokens = tokens  # in infix form

    def execute(self) -> Value:
        postfix = self.to_postfix_form()
        result = self.evaluate_postfix_form(postfix)
        return result

    def evaluate_postfix_form(self,
                              terms: List[Union[Value, Operator]]) -> Value:

        operands = Stack()
        for term in terms:
            if isinstance(term, Operator):
                rhs = operands.pop()
                lhs = operands.pop()
                res = term.compute(lhs, rhs)
                operands.push(res)
            else:
                operands.push(term)

        assert len(operands) == 1, "Must have 1 element only"

        return operands.pop()

    def to_postfix_form(self):
        """
        Convert to postfix notation (aka Reverse Polish Notation)

        "1 + 2 * 3 + 4 * 5 + 6" => "1 2 + 3 4 + 5 6 + * *"
        "2 * 3 + (4 * 5)"       => "2 3 4 5 * + *"

        TODO: can produce AST directly while building RPN
        """

        operators = Stack()
        terms = Queue()  # output in postfix form

        if DEBUG:
            print("-- Tokens --")
            print([t.value for t in self.tokens])

        while self.tokens:
            token = self.tokens.pop(0)

            if DEBUG:
                print(f"Looking at: {token.value}")

            if isinstance(token, Value):
                terms.enq(token)

            if isinstance(token, Operator):
                # move all operators with higher precedence to the output queue
                if DEBUG:
                    print("..Terms: ", [t.value for t in terms])
                    print("..Operators:", [t.value for t in operators])

                while (operators
                       and not isinstance(operators.peek(), OpenParen)
                       and token < operators.peek()):
                    if DEBUG:
                        print(f"..Moving: {operators.peek().name}")
                    terms.enq(operators.pop())
                operators.push(token)
                if DEBUG:
                    print("..Updated:", [t.value for t in operators])

            if isinstance(token, OpenParen):
                operators.push(token)

            if isinstance(token, CloseParen):
                while not isinstance(operators.peek(), OpenParen):
                    terms.enq(operators.pop())
                operators.pop()  # remove OpenParen

        while operators:
            op = operators.pop()
            if not isinstance(op, OpenParen):
                terms.enq(op)

        if DEBUG:
            print([t.value for t in terms])

        return terms


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
    if token in {'+', '*'}:
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


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    summa = 0
    for line in lines:
        calculator = NaiveCalculator(lexemes(line))
        res = calculator.execute()
        summa += res.value
    return summa


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    summa = 0
    for line in lines:
        calculator = AdvancedCalculator(lexemes(line))
        res = calculator.execute()
        summa += res.value
    return summa


tests = [
    ("1 + 2 * 3 + 4 * 5 + 6", 71, 231),
    ("(1 + 2 * 3 + 4 * 5 + 6)", 71, None),
    ("(1 + 2 * 3 + 4 * 5) + 6", 71, None),
    ("(1 + 2 * 3 + 4) * 5 + 6", 71, None),
    ("1 + (2 * 3) + (4 * (5 + 6))", 51, 51),
    ("2 * 3 + (4 * 5)", 26, 46),
    ("5 + (8 * 3 + 9 + 3 * 4 * 3)", 437, 1445),
    ("5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))", 12240, 669060),
    ("((2 + 4 * 9))", 54, None),
    ("( 2 + 3 + 4) * ( 2 + 3 )", 45, None),
    ("((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6)", 6810, None),
    ("((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2", 13632, 23340)
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
    exp2 = 323912478287549
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
