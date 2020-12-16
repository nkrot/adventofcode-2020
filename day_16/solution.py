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


def read_rules(lines):
    rules = {}
    for line in lines:
        line = line.strip()
        if not line:
            break
        # route: 31-453 or 465-971
        m = re.match(r'([^:]+): (\d+)-(\d+) or (\d+)-(\d+)', line)
        assert m, f"Unexpected line: {line}"
        name = m.group(1)
        a1, a2 = int(m.group(2)), int(m.group(3))
        b1, b2 = int(m.group(4)), int(m.group(5))
        assert name not in rules, f"Duplicate rule: {name}"
        rules[name] = [(a1, a2), (b1, b2)]
    # print(rules)
    return rules


def read_tickets(lines):
    doit = False
    tickets = []
    for line in lines:
        line = line.strip()
        if re.search(r'(your|nearby) tickets?:', line):
            doit = True
            continue
        if doit and line:
            ticket = Ticket(line)
            tickets.append(ticket)

    return tickets.pop(0), tickets


class Ticket(object):

    def __init__(self, text):
        self.codes = utils.to_numbers(text.split(','))
        self.valid = True
        self._field_names = None

    def fields(self):
        return list(zip(self.field_names, self.codes))

    @property
    def field_names(self):
        return self._field_names

    @field_names.setter
    def field_names(self, names):
        assert len(names) == len(self.codes), "Size mismatch"
        self._field_names = tuple(names)

    def __len__(self):
        return len(self.codes)

    def __iter__(self):
        return iter(self.codes)

    def __repr__(self):
        text = ",".join(str(c) for c in self.codes)
        return "<{}: {} {}>".format(
            self.__class__.__name__,
            text,
            ",".join(self.field_names or []))


class TicketValidator(object):

    def __init__(self, rules):
        self.rules = rules
        self.tape = self._make_tape()
        self.name = ",".join(self.rules.keys())

    def _make_tape(self):
        # fill an array with 0s and 1s to mark which numbers are valid
        # TODO: can make it better by merging rules and making less intervals
        maxn = 0
        for name, intervals in self.rules.items():
            for interval in intervals:
                maxn = max(maxn, max(interval))

        tape = [0] * (1+maxn)
        for name, intervals in self.rules.items():
            for mn, mx in intervals:
                for i in range(mn, mx+1):
                    tape[i] = 1

        return tape

    def validate(self, ticket: Ticket) -> List[int]:
        errors = []
        for code in ticket:
            if code > len(self.tape) or self.tape[code] == 0:
                errors.append(code)
        # print(errors)
        return errors

    def is_valid(self, ticket: Ticket) -> bool:
        errors = self.validate(ticket)
        return len(errors) == 0

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.rules)


def solve_p1(lines: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    rules = read_rules(lines)
    my_ticket, tickets = read_tickets(lines)

    validator = TicketValidator(rules)

    error_rate = 0
    for ticket in tickets:
        error_rate += sum(validator.validate(ticket))

    return error_rate


def solve_p2(lines: List[str], for_testing=False) -> int:
    """Solution to the 2nd part of the challenge"""
    rules = read_rules(lines)
    my_ticket, tickets = read_tickets(lines)

    validator = TicketValidator(rules)

    assert validator.is_valid(my_ticket), \
        f"Your ticket is not valid: {my_ticket}"

    tickets = [t for t in tickets if validator.is_valid(t)]

    if DEBUG:
        print("--- Valid tickets ---")
        print(tickets)

    tickets.append(my_ticket)

    # Algorithm
    # for each code in the ticket and based on rules for each individual field,
    # determine a list of fields with which the code complies:
    #  the code votes for specific field.
    # select a combination of votes that contains all fields such that every
    # field occurs only once.

    # create a dedicated validator for every field
    validators = {}
    for name, intervals in rules.items():
        val = TicketValidator({name: intervals})
        validators[name] = val
        if DEBUG:
            print(val)

    # data structure:
    #            | ticket 1           |  ticket 2           | ticket 3
    # -----------|--------------------|---------------------|-------------
    # position_1 [ votes_from_ticket1 , votes_from_ticket_2 , ...        ]
    # position_2 [ votes_from_ticket1 , votes_from_ticket_2 , ...        ]
    votes = [[None] * len(tickets) for _ in range(len(my_ticket))]

    for tnum, ticket in enumerate(tickets):
        if DEBUG:
            print(ticket)
        for pos, code in enumerate(ticket):
            if DEBUG:
                print(f"Ticket code[{pos}]: {code}")
            votes_from_ticket = []
            for vnum, (vname, validator) in enumerate(validators.items()):
                if validator.is_valid([code]):
                    votes_from_ticket.append(vname)
            if DEBUG:
                print(votes_from_ticket)
            votes[pos][tnum] = votes_from_ticket
        if DEBUG:
            print("")

    if DEBUG:
        show(votes, "Initial State")

    while disambiguate(votes):
        continue

    # TODO: sanity check after disambiguation: check that all columns
    # have equal values, like this:
    # code[0] | row   | row   | row   | row
    # code[1] | class | class | class | class
    # code[2] | seat  | seat  | seat  | seat

    field_names = transpose(votes)[0]
    field_names = [n[0] for n in field_names]

    my_ticket.field_names = field_names

    if DEBUG or True:
        print(my_ticket)

    if for_testing:
        return my_ticket.field_names

    res = 1
    for fieldname, fieldvalue in my_ticket.fields():
        if fieldname.startswith('departure'):
            print((fieldname, fieldvalue))
            res *= fieldvalue

    return res


def disambiguate(votes):
    if DEBUG:
        print("disambiguating...")
    changed = False
    changed = apply_row_heuristics(votes) or changed
    changed = apply_column_heuristics(votes) or changed
    if DEBUG:
        show(votes, "One step done")
    return changed


def apply_row_heuristics(votes):
    """A row (*one* position in *all* tickets) may contain common codes only"""
    changed = False
    for pos in range(len(votes)):
        common = None
        for tnum, votes_from_ticket in enumerate(votes[pos]):
            if common is None:
                common = set(votes_from_ticket)
            else:
                intersection = common.intersection(votes_from_ticket)
                changed = changed or common != intersection
                common = intersection
        votes[pos] = [list(common) for _ in votes[pos]]
    return changed


def apply_column_heuristics(votes):
    """
    All values in a column must be unique:
      * If a cell in a column has one value only, this value must be deleted
        from other cells in this very column."""
    changed = False
    transposed = transpose(votes)
    # show(transposed)
    for rowi, row in enumerate(transposed):
        # find unique value
        uniques = []
        for coli, col in enumerate(row):
            if len(col) == 1:
                uniques.append((coli, col[0]))
        # print(f"Unique: {uniques}")
        # assert len(uniques) == 1, \
        #     f"TODO: more than one unique value: {uniques}. check conflicts"
        while uniques:
            uniq = uniques.pop(0)
            for coli, col in enumerate(row):
                if coli != uniq[0] and uniq[1] in col:
                    col.remove(uniq[1])
                    changed = True
    return changed


def transpose(rows):
    data = None
    for rowi, row in enumerate(rows):
        if data is None:
            data = [[None]*len(rows) for _ in range(len(row))]
            # print(data)
        for coli, col in enumerate(row):
            # print(col)
            data[coli][rowi] = col
    return data


def show(votes, title: str = None):
    if title:
        print(f"--- {title} ---")
    for pos in range(len(votes)):
        # print()
        # for tnum, votes_from_ticket in enumerate(votes[pos]):
        #     print(f"  Ticket[{tnum}]: {votes_from_ticket}")
        fields = [f"code[{pos}]"]
        for tnum, votes_from_ticket in enumerate(votes[pos]):
            # vts = [v[0].upper() for v in votes_from_ticket]
            vts = votes_from_ticket
            fields.append(",".join(sorted(vts)))
        print(" | ".join(fields))


# Alterntive?
# remove a rule or part of a rule
# which tickets become invalid and why (positions that are no longer valid)


text_1 = """class: 1-3 or 5-7
row: 6-11 or 33-44
seat: 13-40 or 45-50

your ticket:
7,1,14

nearby tickets:
7,3,47
40,4,50
55,2,20
38,6,12"""


text_2 = """class: 0-1 or 4-19
row: 0-5 or 8-19
seat: 0-13 or 16-19

your ticket:
11,12,13

nearby tickets:
3,9,18
15,1,5
5,14,9"""


tests = [
    (text_1.split('\n'), 4 + 55 + 12, None),
    (text_2.split('\n'), None, ('row', 'class', 'seat')),
]


def run_tests():
    print("--- Tests ---")

    for tid, (inp, exp1, exp2) in enumerate(tests):
        if exp1 is not None:
            res1 = solve_p1(inp)
            print(f"T1.{tid}:", res1 == exp1, exp1, res1)

        if exp2 is not None:
            res2 = solve_p2(inp, True)
            print(f"T2.{tid}:", res2 == exp2, exp2, res2)


def run_real():
    day = '16'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 27850
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 491924517533
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
