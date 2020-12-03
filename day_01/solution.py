#!/usr/bin/env python

from typing import List

def solve_p1(numbers: List[int], target: int):
    """find two numbers that sum to given number.
    Return their product."""
    numbers = sorted(numbers)
    #print(numbers)
    li = 0
    for ri in range(len(numbers)-1, 0, -1):
        rnum = numbers[ri]
        if rnum > target:
            continue
        while li < ri:
            lnum = numbers[li]
            sum = lnum+rnum
            #print("[{}]={}, [{}]={}, sum={}".format(li, lnum, ri, rnum, sum))
            if sum == target:
                return lnum * rnum
            elif sum > target:
                # move right cursor one step
                break
            else:
                # move left cursor one step
                li += 1
        if li >= ri:
            break
    return None


def solve_p2(numbers: List[int], target):
    """Find 3 numbers that sum to target and return their product"""
    numbers = sorted(numbers)
    for li in range(len(numbers)):
        lnum = numbers[li]
        res = solve_p1(numbers[li+1:], target-lnum)
        if res is not None:
            return res * lnum

tests = [
    ([1721, 979, 366, 299, 675, 1456], 514579, 241861950)
]


def load_input():
    numbers = []
    with open("input.txt") as fd:
        for line in fd:
            numbers.append(int(line.strip()))
    return numbers


def run_tests():
    for inputs,exp1,exp2 in tests:
        res1 = solve_p1(inputs, 2020)
        print(res1 == exp1)
        res2 = solve_p2(inputs, 2020)
        print(res2 == exp2)


def run_real():
    inputs = load_input()
    print("--- Task 1 p.1 ---")
    res = solve_p1(inputs, 2020)
    print(res)

    print("--- Task 1 p.2 ---")
    res = solve_p2(inputs, 2020)
    print(res)


if __name__ == '__main__':
    #run_tests()
    run_real()

