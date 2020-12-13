#!/usr/bin/env python

# # #
#
# eetd -- earliest Estimated Time of Departure

import os
import sys
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = not True


def solve_p1(data: List[str]) -> int:
    """Solution to the 1st part of the challenge"""
    eetd = int(data[0])
    buses = [int(b) for b in data[1].split(',') if b != 'x']

    def make_pairs():
        for ts in range(eetd, eetd + max(buses)):
            for bus in buses:
                yield ts, bus

    for ts, bus in make_pairs():
        if ts % bus == 0:
            break

    if DEBUG:
        print(f"At time {ts} by bus {bus}")

    return bus * (ts - eetd)


class Bus(object):
    def __init__(self, busid, timeoffset):
        self.busid = busid
        self.cycle = int(busid)
        self.timeoffset = timeoffset

    def __str__(self):
        return "<{}: timeoffset={}, cycle={}>".format(
            self.__class__.__name__,
            self.timeoffset,
            self.cycle)

    def __repr__(self):
        return str(self)

    def departing_at(self, timetick):
        """Based on the testcases, all buses depart at timestamp 0.
        The part of the task that tells about offsets of departures mentions
        timestamp t that is misleading if you assume that t=0.

        Always gives a wrong answer if:
        (timetick - self.timeoffset) % self.cycle == 0
        """
        return (timetick % self.cycle == 0)


def solve_p2(data: List[str], start_at: Optional[int] = None) -> int:
    """Solution to the 2nd part of the challenge"""

    buses = [Bus(b, ts) for ts, b in enumerate(data[1].split(',')) if b != 'x']

    if DEBUG:
        print(data[1])
        for bus in buses:
            print(bus)

    buses = sorted(buses, key=lambda b: -b.cycle)
    # show_buses_in_time(buses)
    # exit(100)

    basebus = buses.pop(0)
    # basebus = buses[0]

    step = basebus.cycle
    now = (start_at or 0) // step * step
    if DEBUG:
        print(f"Starting at: {now}, {len(buses)}, {basebus}")

    c_loops, c_max_loops = 0, 50000
    while True:  # now < 3425:
        c_loops += 1
        if c_loops == c_max_loops:
            print(f"Checking {c_loops}: now={now}")
            c_loops = 0

        c_ok = 0
        t = now - basebus.timeoffset
        for bi, bus in enumerate(buses):
            soon = t + bus.timeoffset
            departs = bus.departing_at(soon)
            if DEBUG:
                print("Time: {}, {}, {}\t{}\t{}".format(
                    now, t, soon, bus, departs))
            if departs:
                c_ok += 1
                step *= buses.pop(bi).cycle
                # print(f"Bus {bus} match. Increasing step {step}")
            else:
                break
        if not len(buses):
            break

        now += step

    return t


def show_buses_in_time(buses):
    for t in range(0, 3423):
        for bus in buses:
            departs = bus.departing_at(t)
            print("Time: {}\t{}\t{}".format(t, bus, departs))

        print()


text_1 = """939
7,13,x,x,59,x,31,19"""


tests = [
    # (text_1.split('\n'), 59*5, 1068781),
    ("-1\n17,x,13,19".split('\n'), None, 3417),
    ("-1\n67,7,59,61".split('\n'), None, 754018),
    ("-1\n67,x,7,59,61".split('\n'), None, 779210),
    ("-1\n67,7,x,59,61".split('\n'), None, 1261476),
    ("-1\n1789,37,47,1889".split('\n'), None, 1202161486)
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
    day = '13'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 171
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 539746751134958
    # start_at = 100000000000000
    start_at = 0
    res2 = solve_p2(lines, start_at)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
