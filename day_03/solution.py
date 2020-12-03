#!/usr/bin/env python

# # #
#
# TODO
# 1) there is no need to store the whole board. only squares with Trees
#    can be stored.

from typing import Optional, Tuple, Union


class Board(object):

    @classmethod
    def load_from_file(cls, fname):
        with open(fname) as fd:
            return cls.from_text(fd.read())

    @classmethod
    def from_text(cls, text):
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return cls.from_lines(lines)

    @classmethod
    def from_lines(cls, lines):
        board = cls()
        for rowi,line in enumerate(lines):
            for coli,char in enumerate(line):
                sq = cls.Square(char, (rowi, coli))
                board.add(sq)
        return board

    def __init__(self):
        self.data = {}
        self.repeats = True

    @property
    def shape(self):
        """Return (height,width) """
        maxx = sorted(self.data.keys()).pop()
        maxy = sorted(self.data[0].keys()).pop()
        return (1+maxx, 1+maxy)

    def add(self, square: 'Square'):
        x, y = square.coord
        self.data.setdefault(x, {})[y] = square

    def __getitem__(self, *args):
        x, y = args[0]
        row = self.data.get(x, {})
        if self.repeats:
            maxy = sorted(self.data[0].keys()).pop()
            _y = y % (1+maxy)
            sq = row.get(_y, None)
            if sq:
                sq.y = y
        else:
            sq = row.get(y, None)
        return sq

    def __str__(self):
        lines = []
        for rowi in sorted(self.data.keys()):
            rowd = self.data[rowi]
            row = [rowd[coli].value for coli in sorted(rowd.keys())]
            lines.append("".join(row))
        return "\n".join(lines)


    class Square(object):

        def __init__(self, value: str, xy: Optional[Tuple[int,int]] = None):
            length = len(value)
            assert length == 1, "Must be of length 1 but is {length}"
            self.value = value
            self.x, self.y = list(xy or [-1,-1])

        @property
        def coord(self):
            """Return (row, column)"""
            return (self.x, self.y)

        def __repr__(self):
            return "{}: value='{}', coord={}".format(
                self.__class__.__name__, self.value, self.coord)


class Slope(object):

    @classmethod
    def default(cls):
        return cls(3,1)

    def __init__(self, right, down):
        self.right = right
        self.down = down

    def apply(self, x, y):
        return (x + self.down, y + self.right)

    def __call__(self, x, y):
        return self.apply(x, y)


def solve_p1(text: Union[str, Board],
            slope: Optional[Slope] = Slope.default()) -> int:
    board = text
    if isinstance(text, str):
        board = Board.from_text(text)
    if DEBUG:
        print(board)
        print(board.shape)
    x, y, cnt = 0, 0, 0
    sq = board[x,y]
    while x < board.shape[0]:
        if sq.value == '#':
            cnt += 1
        if DEBUG:
            print(sq)
            if sq.value == '#':
                sq.value = 'X'
            else:
                sq.value = 'O'
        x, y = slope(x, y)
        sq = board[x,y]
    if DEBUG:
        print(board)
    return cnt


def solve_p2(text: Union[str, Board]):
    slopes = [
        Slope(1, 1), # Right 1, down 1.
        Slope(3, 1), # Right 3, down 1.
        Slope(5, 1), # Right 5, down 1.
        Slope(7, 1), # Right 7, down 1.
        Slope(1, 2), # Right 1, down 2.
    ]
    prod = 1
    for slope in slopes:
        prod *= solve_p1(text, slope)
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
    (text1, 7, 336),
    (Board.load_from_file("test_input.0.txt"), 7, 336), # stacked text1
    (Board.load_from_file("test_input.1.txt"), 7, 336),
]


def run_tests():
    for inp,exp1,exp2 in tests:

        print("--- Test, p.1 --- ")
        res = solve_p1(inp)
        print(res == exp1, exp1, res)

        print("--- Test, p.2 --- ")
        res = solve_p2(inp)
        print(res == exp2, exp2, res)


def run_real():
    board = Board.load_from_file("input.txt") # 323x31

    print("--- Task 3 p.1 ---")
    exp1 = 176
    res = solve_p1(board)
    print(res == exp1, exp1, res)

    print("--- Task 3 p.2 ---")
    exp2 = 5872458240
    res = solve_p2(board)
    print(res == exp2, exp2, res)


if __name__ == '__main__':
    DEBUG = False
    run_tests()
    run_real()

