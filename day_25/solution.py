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


def demo(lines: List[str]):
    print("--- demo ---")
    card_pk, door_pk = [int(ln) for ln in lines]
    print("Public keys:", card_pk, door_pk)

    card_loop_size = 8
    door_loop_size = 11

    card_pk_ = compute_key(7, card_loop_size)
    print(card_pk == card_pk)

    door_pk_ = compute_key(7, door_loop_size)
    print(door_pk == door_pk)

    expected_encryption_key = 14897079
    card_encryption_key = compute_key(door_pk, card_loop_size)
    print("Expected encryption key {}, computed {}: {}".format(
        expected_encryption_key, card_encryption_key,
        card_encryption_key == expected_encryption_key))

    door_encryption_key = compute_key(card_pk, door_loop_size)
    print("Expected encryption key {}, computed {}: {}".format(
        expected_encryption_key, door_encryption_key,
        door_encryption_key == expected_encryption_key))


def compute_key(subject_number, loop_size):
    constant = 20201227
    pk = 1
    for i in range(loop_size):
        pk = (pk * subject_number) % constant
    return pk


def find_loop_size(subject_number, expected_key) -> int:
    constant = 20201227

    key, loop_size = 1, 0
    while True:
        loop_size += 1
        key = ( key * subject_number ) % constant
        if DEBUG:
            print("Expected: {}, computed: {}".format(expected_key, key))
        if key == expected_key:
            break

    return loop_size


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""

    card_pk, door_pk = [int(ln) for ln in lines]

    if DEBUG:
        print("Public keys: card={}, door={}".format(card_pk, door_pk))

    card_loop_size = find_loop_size(7, card_pk)
    door_loop_size = find_loop_size(7, door_pk)

    card_encryption_key = compute_key(door_pk, card_loop_size)
    door_encryption_key = compute_key(card_pk, door_loop_size)

    equal = card_encryption_key == door_encryption_key
    if DEBUG:
        print("Encryption keys: card={}, door={}, equal?={}".format(
            card_encryption_key, door_encryption_key, equal))

    assert equal, "Encryption keys must match"

    return card_encryption_key


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    # TODO
    return 0


tests = [
    ("5764801\n17807724".split('\n'), 14897079, None),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        # demo(inp)
        if exp1 is not None:
            res1 = solve_p1(inp)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '25'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 6198540
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    # print(f"--- Day {day} p.2 ---")
    # exp2 = -1
    # res2 = solve_p2(lines)
    # print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
