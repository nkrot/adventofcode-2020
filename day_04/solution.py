#!/usr/bin/env python

# # #
#
#

import re
from typing import List


class Passport(object):

    fields = {
        'byr': 'Birth Year',
        'iyr': 'Issue Year',
        'eyr': 'Expiration Year',
        'hgt': 'Height',
        'hcl': 'Hair Color',
        'ecl': 'Eye Color',
        'pid': 'Passport ID',
        'cid': 'Country ID'
    }

    required_fields = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}
    # missing: cid (Country ID)

    @classmethod
    def from_text(cls, text: str):
        obj = cls()
        for token in text.split():
            obj.add_field(token)
        return obj

    def __init__(self):
        self.data = {}

    def add_field(self, text: str):
        # print(text)
        m = re.search(r'(\S+):(\S+)', text)
        if m:
            key, val = m.group(1), m.group(2)
            self.data[key] = val
            # print("'{}/{}'".format(key, val))
        else:
            raise ValueError(f"Invalid string: '{text}'")

    def is_valid(self):
        # TODO: what about additional keys that are not allowed
        # print(self.data)

        ok = 0
        for fld in sorted(self.required_fields):
            checker = f"is_valid_{fld}"
            assert hasattr(self, checker), f"Invalid field: {fld}"
            if getattr(self, checker)():
                ok += 1
            #     print(f"{fld}: ok")
            # else:
            #     print(f"{fld}: BAD")

        return ok == len(self.required_fields)

    def is_valid_byr(self):
        val = self.data.get('byr', None)
        return (val
                and re.match(r'\d\d\d\d$', val)
                and 1920 <= int(val) <= 2002)

    def is_valid_ecl(self):
        colors = {'amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth'}
        val = self.data.get('ecl', ' ')
        return val in colors

    def is_valid_eyr(self):
        val = self.data.get('eyr', None)
        return (val
                and re.match(r'\d\d\d\d$', val)
                and 2020 <= int(val) <= 2030)

    def is_valid_hcl(self):
        val = self.data.get('hcl', '')
        return re.match(r'#[0-9a-f]{6}$', val)

    def is_valid_hgt(self):
        val = self.data.get('hgt', '')
        m = re.match(r'(\d\d\d?)((cm|in))$', val)
        if m:
            num, nnu = int(m.group(1)), m.group(2)
            if nnu == 'cm' and 150 <= num <= 193:
                return True
            elif nnu == 'in' and 59 <= num <= 76:
                return True
        return False

    def is_valid_iyr(self):
        val = self.data.get('iyr', '')
        return (re.match(r'\d\d\d\d$', val) and 2010 <= int(val) <= 2020)

    def is_valid_pid(self):
        val = self.data.get('pid', '')
        return re.match(r'\d{9}$', val)


def is_passport_valid(p: Passport) -> bool:
    missing_fields = Passport.required_fields - p.data.keys()
    return False if missing_fields else True


def solve_p1(lines: List[str]) -> int:
    # print(len(lines))
    count = 0
    for lns in lines:
        passport = Passport.from_text(" ".join(lns))
        if is_passport_valid(passport):
            count += 1
    return count


def solve_p2(lines: List[str]) -> int:
    # print(len(lines))
    count = 0
    for lns in lines:
        passport = Passport.from_text(" ".join(lns))
        if passport.is_valid():
            count += 1
        #     print("..is valid")
        # else:
        #     print("..is invalid")
    return count


def load_file(fname: str) -> List[str]:
    with open(fname) as fd:
        lines = [[]]
        for line in fd.readlines():
            line = line.strip()
            if line:
                lines[-1].append(line)
            else:
                lines.append([])
    return lines


tests = [
         (load_file("test_input.1.txt"), 2, 2),
         (load_file("test_input.2.txt"), 4, 0),
         (load_file("test_input.3.txt"), 4, 4)
]


def run_tests():
    print("--- Test 04 p.1 ---")
    for idx, (lines, exp1, exp2) in enumerate(tests):
        res1 = solve_p1(lines)
        print(f"T{idx} P1:", exp1 == res1, exp1, res1)

        res2 = solve_p2(lines)
        print(f"T{idx} P2:", exp2 == res2, exp2, res2)


def run_real():
    lines = load_file('input.txt')

    print("--- Day 04 p.1 ---")
    exp1 = 228
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print("--- Day 04 p.2 ---")
    exp2 = 175
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
