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


DEBUG = not True


def parse_lines(lines):
    decks = []
    for grp in utils.group_lines(lines):
        if grp:
            grp.pop(0)
            decks.append([int(n) for n in grp])
    if DEBUG:
        print(decks)
    return decks


def winner_in_combat(c1, c2):
    if c1 > c2:
        return 1
    elif c2 > c1:
        return 2
    else:
        raise ValueError("Oops")


def play_combat_game(player1, player2):
    while player1 and player2:
        c1 = player1.pop(0)
        c2 = player2.pop(0)
        wid = winner_in_combat(c1, c2)
        if wid == 1:
            player1.extend([c1, c2])
        else:
            player2.extend([c2, c1])

    if DEBUG:
        print(player1)
        print(player2)

    return (player1 or player2)


def show_state(p1, p2, level=0):
    indent = "  " * level
    print("{} [{}] Player 1: {}".format(indent, level, p1))
    print("{} [{}] Player 2: {}".format(indent, level, p2))


def play_recursive_combat_game(player1, player2, level=0):
    winner = 1

    gamelog = [[], []]

    while player1 and player2:
        if DEBUG:
            show_state(player1, player2, level)

        if seen(gamelog, player1, player2):
            winner = 1
            if DEBUG:
                print(f"Bad idea. Player {winner} wins")
            break

        c1, c2 = player1.pop(0), player2.pop(0)

        if len(player1) >= c1 and len(player2) >= c2:
            winner = play_recursive_combat_game(
                list(player1[:c1]), list(player2[:c2]), level+1)
        else:
            winner = winner_in_combat(c1, c2)

        if winner == 1:
            player1.extend([c1, c2])
        else:
            player2.extend([c2, c1])

        memoize(gamelog, player1, player2)

    return winner


def seen(memory, player1, player2):
    p1, p2 = tuple(player1), tuple(player2)
    if DEBUG:
        print("Checking {} in: {}".format(p1, memory[0]))
        print("Checking {} in: {}".format(p2, memory[1]))
    return (p1 in memory[0][:-1]) or (p2 in memory[1][:-1])


def memoize(memory, player1, player2):
    memory[0].append(tuple(player1))
    memory[1].append(tuple(player2))


def score(player):
    i, sc = 0, 0
    while player:
        i += 1
        sc += i * player.pop()
    return sc


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    player1, player2 = parse_lines(lines)
    winner = play_combat_game(player1, player2)
    return score(winner)


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    player1, player2 = parse_lines(lines)
    winner_id = play_recursive_combat_game(player1, player2)
    winner = [None, player1, player2][winner_id]
    return score(winner)


text_1 = """Player 1:
9
2
6
3
1

Player 2:
5
8
4
7
10
"""


tests = [
    (text_1.split('\n'), 306, 291)
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
    day = '22'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 33421
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 33651
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
