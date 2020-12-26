#!/usr/bin/env python

# # #
#
#

#import os
#import sys
from typing import List, Union, Optional
#from collections import deque

#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from aoc import utils


DEBUG = False


class GameOfCups(object):
    """ implemented as a dict
    {
        current : (prev,    next),
        value_1 : (value_3  value_2),
        value_2 : (value_2, value_3),
        value_3 : (value_2, value_1)
    }

    Constraints:
    Items are unique (not checked)
    """

    def __init__(self, items: Optional[List[int]] = None):
        self.items = {}
        self._last = None
        self._first = None
        self.removed = []  # temporarily removed items
        self.maxvalue = 0  # global max value
        if items:
            for item in items:
                self.append(item)

    def lowest(self):
        return 1

    def highest(self) -> int:
        maxval = self.maxvalue
        while maxval in self.removed:
            maxval -= 1
        return maxval

    @property
    def first(self):
        return self._first

    @first.setter
    def first(self, newval):
        self._first = newval

    def append(self, value: int):
        if self.first is None:
            self.first = value
        if self._last is not None:
            self.items[self._last] = value
        self.items[value] = self.first
        self._last = value
        if value > self.maxvalue:
            self.maxvalue = value

    def pop3after(self, value: int) -> List[int]:
        """Remove 3 items that immediately follow given item <value>"""
        self.removed.clear()
        nxt = self.items[value]
        for _ in range(3):
            self.removed.append(nxt)
            nxt = self.items[nxt]
        self.items[value] = nxt
        return list(self.removed)

    def insert(self, after: int, new_values: Union[int, List[int]]):
        """Insert given value(s) <new_values> after the value <after>"""
        if not isinstance(new_values, list):
            new_values = [new_values]
        for nval in new_values:
            if nval in self.removed:
                self.removed.remove(nval)
            self.items[nval] = self.items[after]
            self.items[after] = nval
            after = nval

    def __contains__(self, value: int) -> Optional[int]:
        if value in self.items and value not in self.removed:
            return value
        return None

    def after(self, value: int) -> int:
        return self.items[value]

    def listitems(self, start: Optional[int] = None) -> List[int]:
        curr = start or self.first
        # print(f"starting with: {curr}")
        items = [curr]
        nxt = self.after(curr)
        while nxt != curr:
            items.append(nxt)
            nxt = self.after(nxt)
        return items

    def __repr__(self):
        return "<{}: {} {} {}>".format(self.__class__.__name__,
            (self.first,), self.highest(), self.listitems())


def play_one_round(game: GameOfCups, current: int) -> int:
    # print(game)

    # The crab removes 3 cups immediately after the current cup
    removed = game.pop3after(current)

    # The crab finds a destination cup
    found = False
    dest = current
    while dest >= game.lowest():
        dest -= 1
        if dest in game:
            found = True
            break
    if not found:
        dest = game.highest()

    # The crab inserts removed cups immediately after the destination cup
    game.insert(dest, removed)

    # The crab selects a new current cup that is immediately after the current
    # one.
    current = game.after(current)

    return current


def solve_p1(line, n_moves=100) -> str:
    """Solution to the 1st part of the challenge"""
    cups = [int(ch) for ch in list(line)]

    game = GameOfCups(cups)

    current = game.first
    for n in range(n_moves):
        current = play_one_round(game, current)

    cups = game.listitems(1)
    cups.pop(0)
    #print(cups)

    return "".join([str(n) for n in cups])


def solve_p2(line, n_moves=10000000) -> int:
    """Solution to the 2nd part of the challenge"""

    cups = [int(ch) for ch in list(line)]
    game = GameOfCups(cups)

    maxval = 1000000
    for n in range(game.highest()+1, maxval+1):
        game.append(n)

    print("Game has been setup")

    current = game.first
    for n in range(n_moves):
        if n % 1000000 == 0:
            print(f"..rounds played: {n}")
        current = play_one_round(game, current)

    cup1 = game.after(1)
    cup2 = game.after(cup1)

    #print(cup1, cup2)

    return cup1 * cup2


tests = [
    ("389125467", "92658374", None),  # 10 moves
    ("389125467", "67384529", None),  # 100 moves
    ("389125467", None, 934001 * 159792)  # 10 mln moves
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        n_moves = 10 if tid == 0 else 100
        if exp1 is not None:
            res1 = solve_p1(inp, n_moves)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '23'
    inp = "598162734"

    print(f"--- Day {day} p.1 ---")
    exp1 = "32658947"
    res1 = solve_p1(inp)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 683486010900
    res2 = solve_p2(inp)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
