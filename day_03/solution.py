#!/usr/bin/env python

# # #
#
#

import os
import sys
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils
import aoc.board


DEBUG = False


class Park(aoc.Board):

    def __init__(self):
        super().__init__()
        self.repeats = True

    def __getitem__(self, *args):
        if len(args) == 1:
            x, y = args[0]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError(f"Wrong number of arguments: {args}")
        row = self.squares.get(x, {})
        if self.repeats:
            maxy = sorted(self.squares[0].keys()).pop()
            _y = y % (1+maxy)
            sq = row.get(_y, None)
            if sq:
                sq.y = y  # unused. for consistency :)
        else:
            sq = row.get(y, None)
        return sq


class Slope(object):

    @classmethod
    def default(cls):
        return cls(3, 1)

    def __init__(self, right: int, down: int):
        self.right = right
        self.down = down

    def apply(self, x: int, y: int):
        return (x + self.down, y + self.right)

    def __call__(self, x: int, y: int):
        return self.apply(x, y)


def solve_p1(park: Park, slope: Optional[Slope] = None) -> int:
    slope = slope or Slope.default()
    if DEBUG:
        print(park)
        print(park.size())
    x, y, cnt = 0, 0, 0
    sq = park[x, y]
    while x < park.size()[0]:
        if sq.value == '#':
            cnt += 1
        if DEBUG:
            print(sq)
            if sq.value == '#':
                sq.value = 'X'
            else:
                sq.value = 'O'
        x, y = slope(x, y)
        sq = park[x, y]
    if DEBUG:
        print(park)
    return cnt


def solve_p2(park: Park):
    slopes = [
        Slope(1, 1),  # Right 1, down 1.
        Slope(3, 1),  # Right 3, down 1.
        Slope(5, 1),  # Right 5, down 1.
        Slope(7, 1),  # Right 7, down 1.
        Slope(1, 2),  # Right 1, down 2.
    ]
    prod = 1
    for slope in slopes:
        prod *= solve_p1(park, slope)
    return prod


text1 = \
"""..##.......
#...#...#..
.#....#..#.
..#.#...#.#
.#...##..#.
..#.##.....
.#.#.#....#
.#........#
#.##...#...
#...##....#
.#..#...#.#"""


tests = [
    (text1.split('\n'), 7, 336),
    (utils.load_input("test_input.0.txt"), 7, 336),  # stacked text1
    (utils.load_input("test_input.1.txt"), 7, 336),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(Park.from_lines(inp))
        print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        res2 = solve_p2(Park.from_lines(inp))
        print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():

    print("--- Task 3 p.1 ---")
    exp1 = 176
    res1 = solve_p1(Park.load_from_file())  # 323x31
    print(res1 == exp1, exp1, res1)

    print("--- Task 3 p.2 ---")
    exp2 = 5872458240
    res2 = solve_p2(Park.load_from_file())
    print(res2 == exp2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
