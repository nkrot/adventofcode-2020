#!/usr/bin/env python

# # #
#
# TODO
# CNF, CYK parser (bottom up?) and pushdown automaton
#

import re
import os
import sys
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def read_rules_and_messages(lines: List[str]) -> List[List[str]]:
    sections = [[]]
    for line in lines:
        line = line.strip()
        if line:
            sections[-1].append(line)
        else:
            sections.append([])

    return sections


class Parser(object):

    # 3: 4 5 | 5 4
    # 4: "a"
    @classmethod
    def from_lines(cls, lines: List[str]) -> 'Parser':
        rules = {}
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                continue
            # print(f"PARSING: {line}")
            m = re.match(r'(\d+):\s+(.+)', line)
            assert m, f"Invalid rule line: {line}"

            rule = cls.Rule()
            rule.id = int(m.group(1))
            rules[rule.id] = rule

            for sub_id, part in enumerate(m.group(2).split('|')):
                subr = cls.Rule.from_text(part)
                subr.id = (rule.id, sub_id)
                rule.subrules.append(subr)
                rules[subr.id] = subr

            # print("Main rule:", rule)
            # for subr in rule.subrules:
            #     print("  subrule:", subr)

        return cls(rules)

    def __init__(self, rules):
        self.rules = rules

    def is_valid(self, text: str) -> bool:
        """Check if given text string <text> can be fully described by
        the parsing rules.
        """
        self.sequence = self.Input(text)

        matches = self.match_rule(0, 0)
        full_matches = [m for m in matches if m.end == len(self.sequence)]

        if DEBUG:
            print("--- Matches ---")
            for m in matches:
                print(m)

        return len(full_matches) > 0

    def match_rule(self, rule_id, offset: int) -> List['Match']:
        """Check that rule <rule_id> matches <self.sequence> starting at the
        offset <offset>.
        Return:
          List[Match], can be empty.
        TODO: for now, only end offset is correct. will fix the start offset
        when necessary.
        """
        matches = []

        if isinstance(rule_id, self.Rule):
            rule = rule_id
        else:
            rule = self.rules[rule_id]

        if rule.is_terminal():
            text = self.sequence[offset, len(rule)]
            if rule == text:
                m = self.Match(True, offset, offset + len(rule))
                matches.append(m)

        elif rule.is_compound():
            for subrule in rule.subrules:
                submatches = self.match_rule(subrule, offset)
                if submatches:
                    matches.extend(submatches)

        else:
            # 0: 4 1 5
            submatches = [self.Match(True, 0, offset)]
            for term in rule:
                for _ in range(len(submatches)):
                    # continue matching after successful matches
                    subm = submatches.pop(0)
                    subms = self.match_rule(term, subm.end)
                    if subms:
                        # TODO: update the start offset of new matches
                        # to be real start offset at the very beginning
                        # of matching process?
                        submatches.extend(subms)
            if submatches:
                matches.extend(submatches)
        if DEBUG:
            print(len(matches), matches)

        return matches
    class Rule(object):

        @classmethod
        def from_text(cls, text: str) -> 'Rule':
            """Create a rule from RHS of the atomic (non OR) rule
                3: 4 5       -> create a rule from 3 5 (NONTERMINAL)
                4: "a"       -> create a rule from "a" (TERMINAL)
            """
            assert '|' not in text, \
                f"Cannot parse compound rules: {text}"
            text = text.strip()
            rule = cls()
            if re.match(r'\d+(\s+\d+)*$', text):
                rule.terms = [int(t) for t in text.split()]
            else:
                rule.literal = text.strip('" ')
            return rule

        def __init__(self):
            self.literal = None
            self.id = None      # int or Tuple[int, int]
            self.subrules = []  # [subrule1_id, subrule2_id, ...]
            self.terms = []     # [3, 2] from "1: 3 2"

        def __eq__(self, other):
            return self.is_terminal() and self.literal == other

        def __len__(self):
            assert self.is_terminal(), "Only terminal symbols have length"
            return len(self.literal)

        def __iter__(self):
            return iter(self.terms)

        def is_terminal(self) -> bool:
            return (self.literal is not None)

        def is_compound(self) -> bool:
            """Return True of the rule has several parts separated by |"""
            return len(self.subrules) > 0

        def __repr__(self):
            val = self.literal or self.terms
            return "<{} {}: {} {}>".format(
                self.__class__.__name__, str(self.id), self._type(), str(val))

        def _type(self):
            if self.is_terminal():
                return "TERMINAL"
            if self.is_compound():
                return "COMPOUND"
            return "NONTERMINAL"


    class Match(object):

        def __init__(self, status, start_offset=None, end_offset=None):
            self.status = status
            self.start = start_offset
            self.end = end_offset

        def __bool__(self):
            return self.status

        def __repr__(self):
            return "<{} {} {}>".format(self.__class__.__name__,
                                       self.status, (self.start, self.end))

    class Input(object):

        def __init__(self, text: str):
            self.data = list(text)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, start, length=1) -> Optional[str]:
            if isinstance(start, tuple):
                start, length = start
            end = start + length
            if end > len(self):
                return None
            return "".join(self.data[start:end])

        def __repr__(self):
            return "<{}: length={} {}>".format(self.__class__.__name__,
                                               len(self), str(self))

        def __str__(self):
            return "".join(self.data)


# s = "abcd"
# text = Parser.Input(s)
# print(text[0] == "a")
# print(text[1,3] == "bcd")
# print(text[2,2] == "cd")
# print(text[2,3] == None)
# print(text[3] == "d")
# print(text[3,1] == "d")
# print(text[3,2] == None)
# print(text[4] == None)


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    lines_with_rules, messages = read_rules_and_messages(lines)
    parser = Parser.from_lines(lines_with_rules)

    if DEBUG:
        print("--- All Rules ---")
        for rid, rule in parser.rules.items():
            print(rule)
        print()

    checks = [parser.is_valid(msg) for msg in messages]
    return checks.count(True)


def solve_p2(lines: List[str]) -> int:
    """Solution to the 2nd part of the challenge"""
    patch = [
        "8: 42 | 42 8",
        "11: 42 31 | 42 11 31"
    ]

    lines = [line for line in lines if not re.match(r'(8|11):', line) ]
    patch.extend(lines)

    return solve_p1(patch)


text_1 = """0: 4 1 5
1: 2 3 | 3 2
2: 4 4 | 5 5
3: 4 5 | 5 4
4: "a"
5: "b"

ababbb
bababa
abbbab
aaabbb
aaaabbb"""


tests = [
    (text_1.split('\n'), 2, None),
    (utils.load_input("test_input.0.txt"), 3, 12),
    # test_input.0.txt with manually replaced rules 8 and 11
    (utils.load_input("test_input.1.txt"), 12, None),
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
    day = '19'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 107
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 321
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
