#!/usr/bin/env python

# # #
# template script
# 1) copy it under a day-related directory
# 2) fill in TODOs
#

import re
import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def read_rules(lines):
    rules = {}
    for line in lines:
        line = line.strip()
        if not line:
            break
        # route: 31-453 or 465-971
        m = re.match(r'([^:]+): (\d+)-(\d+) or (\d+)-(\d+)', line)
        assert m, f"Unexpected line: {line}"
        name = m.group(1)
        a1, a2 = int(m.group(2)), int(m.group(3))
        b1, b2 = int(m.group(4)), int(m.group(5))
        assert name not in rules, f"Duplicate rule: {name}"
        rules[name] = [(a1, a2), (b1, b2)]
    # print(rules)
    return rules


def read_tickets(lines):
    doit = False
    tickets = []
    for line in lines:
        line = line.strip()
        if re.search(r'(your|nearby) tickets?:', line):
            doit = True
            continue
        if doit and line:
            ticket = utils.to_numbers(line.split(','))
            tickets.append(ticket)

    return tickets.pop(0), tickets


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    rules = read_rules(lines)
    my_ticket, tickets = read_tickets(lines)

    print(rules)

    # fill an array with 0s and 1s to mark which numbers are valid
    # TODO: can make it better by merging rules and making less intervals
    maxn = 0
    for name, intervals in rules.items():
        for interval in intervals:
            maxn = max(maxn, max(interval))

    tape = [0] * (1+maxn)
    for name, intervals in rules.items():
        for mn, mx in intervals:
            for i in range(mn, mx+1):
                tape[i] = 1

    # print(tape)

    invalid_codes = []
    for ticket in tickets:
        for code in ticket:
            if code > len(tape) or tape[code] == 0:
                invalid_codes.append(code)

    return sum(invalid_codes)


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    # TODO
    return 0


text_1 = """class: 1-3 or 5-7
row: 6-11 or 33-44
seat: 13-40 or 45-50

your ticket:
7,1,14

nearby tickets:
7,3,47
40,4,50
55,2,20
38,6,12"""


tests = [
    (text_1.split('\n'), 4 + 55 + 12, None),
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
    day = 'DD'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 27850
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 360
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
