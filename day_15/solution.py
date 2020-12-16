#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def solve_p1(numbers: List[int], n_turns: int = 2020) -> int:
    game = defaultdict(list)

    def age(num):
        return game[num][-1] - game[num][0]

    def stores(num, t):
        # TODO: why storing two numbers? age can receive relevant t
        # as a parameter.
        # this method can replace the old value with the new one
        # returning the old value
        if DEBUG:
            print(f"Turn {1+t}:\n  heard {num}")
        game[num].append(t)
        if len(game[num]) > 2:
            game[num].pop(0)
        return num

    def speaks(num):
        if DEBUG:
            print(f" speaks {num}")
        return num

    # 0,3,6, 0,3,3,1,0,4,0
    # n_turns = 10

    last = 0
    for r in range(1, len(numbers)):
        last = stores(numbers[r-1], r)  # bad name, bad reporting
    last = numbers[-1]

    for r in range(len(numbers), n_turns):
        if DEBUG:
            print("")
            print(f"Turn {1+r}:", last, dict(game))
        if last in game:
            stores(last, r)
            last = speaks(age(last))
        else:
            stores(last, r)
            last = speaks(0)

    # print(dict(game))
    return last


def old_solve_p1(numbers: List[int], n_turns: int = 2020) -> int:
    """Solution to the 1st part of the challenge"""
    game = list(reversed(numbers))
    while len(game) < n_turns:
        try:
            idx = game.index(game[0])
            try:
                jdx = game.index(game[0], idx+1)
            except ValueError:
                jdx = 0
            newvalue = abs(jdx - idx)
        except ValueError:
            newvalue = 0
        game.insert(0, newvalue)
    return game[0]


def solve_p2(numbers: List[int]) -> int:
    """Solution to the 2nd part of the challenge"""
    return solve_p1(list(numbers), 30000000)


# text_1 = """hello
# yellow
# melon"""


tests = [
    ("0,3,6", 436, 175594),
    ("1,3,2", 1, 2578),
    ("2,1,3", 10, 3544142),
    ("1,2,3", 27, 261214),
    ("2,3,1", 78, 6895259),
    ("3,2,1", 438, 18),
    ("3,1,2", 1836, 362)
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        inp = utils.to_numbers(inp.split(','))

        if exp1 is not None:
            res1 = solve_p1(inp)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '15'
    inp = utils.to_numbers("2,0,1,7,4,14,18".split(','))

    print(f"--- Day {day} p.1 ---")
    exp1 = 496
    res1 = solve_p1(inp)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 883
    res2 = solve_p2(inp)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
