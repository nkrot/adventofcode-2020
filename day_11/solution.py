#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils
import aoc.board


DEBUG = False


class Seat(aoc.Square):
    # Q: a seat must have a reference to the Board?

    VALUES = ('L', '#', '.')

    def __init__(self, *args):
        super().__init__(*args)

    def is_empty(self):
        return not self.is_floor() and self.value == 'L'

    def is_occupied(self):
        return not self.is_floor() and self.value == '#'

    def is_floor(self):
        return self.value == '.'

    def leave(self):
        # print(f"{self.__class__.__name__}.leave() is called")
        assert not self.is_floor(), "Wrong operation applied to a Floor"
        self.value = 'L'

    def occupy(self):
        # print(f"{self.__class__.__name__}.occupy() is called")
        assert not self.is_floor(), "Wrong operation applied to a Floor"
        self.value = '#'

    def __str__(self):
        return self.value

    def __repr__(self):
        msg = "<{}: position={}, value={}, is_floor={}, is_empty={}>".format(
            self.__class__.__name__, self.position, self.value,
            self.is_floor(), self.is_empty())
        return msg


class WaitingArea(aoc.Board):
    PIECE = Seat

    def __init__(self):
        super().__init__()

    def adjacent_seats(self, seat):
        """Return a list of seats that are adjacent to the given seat"""
        x, y = seat.position

        nseats = []
        for dx, dy in self.OFFSETS_AROUND:
            nx, ny = x + dx, y + dy
            nseat = self[nx, ny]
            if nseat and not nseat.is_floor():
                nseats.append(nseat)

        return nseats

    def empty_seats(self) -> List[Seat]:
        """List all empty seats"""
        return [seat for seat in self if seat.is_empty()]

    def occupied_seats(self) -> List[Seat]:
        """List all occupied seats"""
        return [seat for seat in self if seat.is_occupied()]


class Command(object):

    @classmethod
    def occupy(cls, seat):
        obj = cls(seat, 'occupy')
        return obj

    @classmethod
    def leave(cls, seat):
        obj = cls(seat, 'leave')
        return obj

    def __init__(self, receiver: object, meth: str):
        self.receiver = receiver
        self.method = meth
        self.debug = False

    def execute(self):
        """Execute the command"""
        assert hasattr(self.receiver, self.method), \
            f"Object {self.receiver} does not have attribute {self.method}"
        if self.debug:
            print(f"Before command: {repr(self.receiver)}")
        ret = getattr(self.receiver, self.method)()
        if self.debug:
            print(f"After command : {repr(self.receiver)}")
        return ret

    def __call__(self):
        return self.execute()


def run_one_round(wa):
    debug = False

    commands = []

    for seat in wa:
        if debug:
            print(repr(seat))

        if seat.is_floor():
            continue

        adj_seats = wa.adjacent_seats(seat)
        if debug:
            for s in adj_seats:
                print("  ", repr(s))

        empty_adj_seats = [s.is_empty() for s in adj_seats]
        if debug:
            print(empty_adj_seats)

        if seat.is_empty():
            if all(empty_adj_seats):
                commands.append(Command.occupy(seat))
        else:
            if empty_adj_seats.count(False) > 3:
                commands.append(Command.leave(seat))

    for cmd in commands:
        cmd.execute()

    changed = len(commands) > 0

    return changed


def solve_p1(wa: WaitingArea):
    """Solution to the 1st part of the challenge"""
    if DEBUG:
        print("--- Before ---")
        print(wa)

    c_rounds = 0
    while True:
        c_rounds += 1
        changed = run_one_round(wa)
        if DEBUG:
            print(f"--- After round {c_rounds} ---")
            print(wa)
        if not changed:
            break

    if DEBUG:
        print("Number of rounds:", c_rounds)

    return len(wa.occupied_seats())


def solve_p2(data):
    """Solution to the 2nd part of the challenge"""
    # TODO
    pass


text_1 = """L.LL.LL.LL
LLLLLLL.LL
L.L.L..L..
LLLL.LL.LL
L.LL.LL.LL
L.LLLLL.LL
..L.L.....
LLLLLLLLLL
L.LLLLLL.L
L.LLLLL.LL"""

tests = [
    (text_1, 37, -1),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(WaitingArea.from_text(inp))
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        # res2 = solve_p2(WaitingArea.from_text(inp))
        # print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '11'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 2418
    res1 = solve_p1(WaitingArea.from_lines(lines))
    print(exp1 == res1, exp1, res1)

    # print(f"--- Day {day} p.2 ---")
    # exp2 = 360
    # res2 = solve_p2(lines)
    # print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
