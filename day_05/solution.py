#!/usr/bin/env python

# # #
#
#

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def decode(code: str, mn: int, mx: int, msg: str = ''):
    if DEBUG:
        print(f"{msg}:", code, mn, mx)
    for ch in code:
        if ch in {'F', 'L'}:
            mx -= (mx-mn+1)/2
        elif ch in {'B', 'R'}:
            mn += (mx-mn+1)/2
        else:
            raise ValueError("Wrong")
    if DEBUG:
        print("DECODED:", code, mn, mx)
    assert mn == mx, "Did not converge"
    return mn


def seat_id(code):
    row = decode(code[0:7], 0, 127, "ROW")
    col = decode(code[7:],  0,   7, "COL")
    return int(row * 8 + col)


def solve_p1(boarding_passes):
    """Find max seat id"""
    seat_ids = [seat_id(bpass) for bpass in boarding_passes]
    max_seat_id = max(seat_ids)
    return max_seat_id


def solve_p2(boarding_passes):
    """Find an empty seat that is surrounded by non-empty seats
    on both sides"""

    seats = [0] * 1024  # mark all seats as empty

    for bpass in boarding_passes:
        sid = seat_id(bpass)
        seats[sid] = 1  # mark the seat as occcupied
        # print(bpass, sid, file=sys.stderr)

    candidate_seat_ids = []
    for sid in range(1, len(seats)-1):
        if not seats[sid] and seats[sid-1] and seats[sid+1]:
            candidate_seat_ids.append(sid)

    assert len(candidate_seat_ids) == 1, "Must be exactly one solution"

    return candidate_seat_ids[0]


tests = [
    ('FBFBBFFRLR',  357, -1),
    ('BFFFBBFRRR',  567, -1),  # row 70, column 7, seat ID 567.
    ('FFFBBBFRRR',  119, -1),  # row 14, column 7, seat ID 119.
    ('BBFFBBFRLL',  820, -1),  # row 102, column 4, seat ID 820.
    ('FFFFFFFLLL',    0, -1),  # minimal seat id
    ('BBBBBBBRRR', 1023, -1),  # max seat id
]


def run_tests():
    print("--- Tests ---")
    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = seat_id(inp)
        print("T1.{}: {} {} {}".format(tid, res1 == exp1, exp1, res1))


def run_real():
    lines = utils.load_input()  # loads ./input.txt

    print("--- Day 5 p.1 ---")
    exp1 = 901
    res1 = solve_p1(lines)
    print("D.1: {} {} {}".format(res1 == exp1, exp1, res1))

    print("--- Day 5 p.2 ---")
    exp2 = 661
    res2 = solve_p2(lines)
    print("D.2: {} {} {}".format(res2 == exp2, exp2, res2))


def main():
    run_tests()
    run_real()


if __name__ == '__main__':
    main()
