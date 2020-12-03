#!/usr/bin/env python

# # #
#
#

import re
from typing import List, Callable, Tuple

debug = False


def parse_line(line: str) -> Tuple[int, int, str, str]:
    m = re.match(r'(\d+)-(\d+)\s+([^:]+):\s+(\S+)', line)
    mn, mx, char, password = m.groups()
    mn, mx = int(mn), int(mx)
    res = (mn, mx, char, password)
    if debug:
        print("PARSED:", res)
    return res


def solve(lines: List[str], is_valid: Callable) -> int:
    '''return number of valid lines'''
    hits = [1 for line in lines if is_valid(*parse_line(line))]
    return sum(hits)


def is_valid_p1(mn: int, mx: int, char: str, password: str) -> bool:
    """Return True if the number of occurrences of given character <char>
    in <password> is at least <mn> and at most <mx>"""
    hits = [1 for ch in password if char == ch]
    return mn <= sum(hits) <= mx


def is_valid_p2(mn: int, mx: int, char: str, password: str) -> bool:
    """Return True if given character <char> occurs in <password>
    either at position <mn> or at position <mx> but not both.
    Positions <mn> and <mx> are 1-based."""
    counts = [1 for pos in [mn, mx] if password[pos-1] == char]
    return sum(counts) == 1


tests = [
    (['1-3 a: abcde',
      '1-3 b: cdefg',
      '2-9 c: ccccccccc'], 2, 1)
]


def run_tests():
    for lines, exp1, exp2 in tests:
        res = solve(lines, is_valid_p1)
        print(res == exp1)

        res = solve(lines, is_valid_p2)
        print(res == exp2)


def load_input() -> List[str]:
    lines = []
    with open('input.txt') as fd:
        for line in fd:
            lines.append(line.strip())
    return lines


def run_real():
    lines = load_input()
    print('--- p.1 ---')
    res = solve(lines, is_valid_p1)
    print(res)

    print('--- p.2 ---')
    res = solve(lines, is_valid_p2)
    print(res)


if __name__ == '__main__':
    run_tests()
    run_real()
