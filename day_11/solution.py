#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import List, Callable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils
import aoc.board
import aoc.command

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

    def adjacent_seats(self, seat: Seat) -> List[Seat]:
        """Return a list of seats that are adjacent to the given seat"""
        nseats = [nseat for nseat in self.neighbours(seat)
                  if not nseat.is_floor()]
        return nseats

    def visible_seats(self, seat: Seat) -> List[Seat]:
        """Return a list of seats that are reachable from the given
        seat in all 8 directions"""
        x, y = seat.position
        nseats = []
        for dx, dy in self.OFFSETS_AROUND:
            i = 0
            while True:
                i += 1
                nx, ny = x + i*dx, y + i*dy
                if nx < 0 or ny < 0:
                    break
                nseat = self[nx, ny]
                # print(nx, ny, repr(nseat))
                if nseat:
                    if not nseat.is_floor():
                        nseats.append(nseat)
                        break
                else:
                    break
        return nseats

    def empty_seats(self) -> List[Seat]:
        """List all empty seats"""
        return [seat for seat in self if seat.is_empty()]

    def occupied_seats(self) -> List[Seat]:
        """List all occupied seats"""
        return [seat for seat in self if seat.is_occupied()]


class Action(aoc.Command):

    @classmethod
    def occupy(cls, seat):
        obj = cls(seat, 'occupy')
        return obj

    @classmethod
    def leave(cls, seat):
        obj = cls(seat, 'leave')
        return obj

    def __init__(self, *args):
        super().__init__(*args)


def run_one_round(wa: WaitingArea,
                  select_seats_to_check: Callable,
                  acceptable_number_of_occupied_seats: Callable) -> bool:
    debug = False
    commands = []

    for seat in wa:
        if debug:
            print(repr(seat))

        if seat.is_floor():
            continue

        adj_seats = select_seats_to_check(wa, seat)

        if debug:
            for s in adj_seats:
                print("  ", repr(s))

        empty_adj_seats = [s.is_empty() for s in adj_seats]
        if debug:
            print(empty_adj_seats)

        if seat.is_empty():
            if all(empty_adj_seats):
                commands.append(Action.occupy(seat))
        else:
            # number of occupied seats that forces people to leave their seat
            c_occupied_seats = empty_adj_seats.count(False)
            if not acceptable_number_of_occupied_seats(c_occupied_seats):
                commands.append(Action.leave(seat))

    for cmd in commands:
        cmd.execute()

    changed = len(commands) > 0

    return changed


def solve(wa: WaitingArea,
          select_seats_to_check: Callable,
          check_number_of_occupied_seats: Callable) -> int:

    if DEBUG:
        print("--- Before ---")
        print(wa)

    c_rounds = 0
    while True:
        c_rounds += 1
        changed = run_one_round(wa,
                                select_seats_to_check,
                                check_number_of_occupied_seats)
        if DEBUG:
            print(f"--- After round {c_rounds} ---")
            print(wa)
        if not changed:
            break

    if DEBUG:
        print("Number of rounds:", c_rounds)

    return len(wa.occupied_seats())


def solve_p1(wa: WaitingArea) -> int:
    """Solution to the 1st part of the challenge"""

    def select_adjacent(area, seat):
        return area.adjacent_seats(seat)

    def tolerate_n_neighbours(num_occupied_seats):
        return num_occupied_seats < 4

    return solve(wa, select_adjacent, tolerate_n_neighbours)


def solve_p2(wa: WaitingArea) -> int:
    """Solution to the 2nd part of the challenge"""

    def select_visible(area, seat):
        return area.visible_seats(seat)

    def tolerate_n_neighbours(num_occupied_seats):
        return num_occupied_seats < 5

    return solve(wa, select_visible, tolerate_n_neighbours)


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
    (text_1, 37, 26),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(WaitingArea.from_text(inp))
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(WaitingArea.from_text(inp))
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '11'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 2418
    res1 = solve_p1(WaitingArea.from_lines(lines))
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 2144
    res2 = solve_p2(WaitingArea.from_lines(lines))
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
