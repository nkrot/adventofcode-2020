#!/usr/bin/env python

# # #
# appox 1 hr
#

import re
import os
import sys
from typing import List
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def parse_rules(lines: List[str]) -> dict:
    rules = defaultdict(list)
    for line in lines:
        ln = re.sub(r'\sno\s', '0 ', line)
        ln = re.sub(r' bags?[,.]?( contains?)? ?', '@', ln)
        chunks = [ch for ch in ln.split('@') if ch]
        # print(chunks)
        head = chunks.pop(0)
        for chunk in chunks:
            m = re.match(r'(\d+)\s(.+)', chunk)
            num, color = int(m.group(1)), m.group(2)
            rules[head].append((num, color))
    if DEBUG:
        print(rules)
    return rules


def solve_p1(data: List[str]):
    """Solution to the 1st part of the challenge"""
    rules = parse_rules(data)

    inverted = defaultdict(list)
    for parent, children in rules.items():
        for num, color in children:
            inverted[color].append(parent)
    if DEBUG:
        print(inverted)

    color = 'shiny gold'
    visiting, visited = [color], []
    while visiting:
        color = visiting.pop(0)
        if color in inverted:
            visiting.extend(inverted[color])
        visited.append(color)

    visited.pop(0)  # delete starting bag
    visited = set(visited)

    return len(visited)


def solve_p2(data: List[str]):
    """Solution to the 2nd part of the challenge"""
    rules = parse_rules(data)
    color = 'shiny gold'

    def count_inner_bags(color):
        cnt = 0
        if color in rules:
            for num, clr in rules[color]:
                if clr != 'other':
                    cnt += num
                    cnt += num * count_inner_bags(clr)
        return cnt

    return count_inner_bags(color)


tests = [
    (utils.load_input("test_input.1.txt"), 4, 32),
    (utils.load_input("test_input.2.txt"), 0, 126),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(inp)
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(inp)
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = 'DD'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 112
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 6260
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
