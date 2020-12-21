#!/usr/bin/env python

# # #
#
# similar to day 16
#

import re
import os
import sys
from typing import List, Tuple, Union
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from aoc import utils


DEBUG = False


def parse_menu_line(line: str) -> Tuple[List[str], List[str]]:
    allergens = []
    m = re.search(r'(.+)\s[(]contains\s([^()]+)[)]$', line)
    if m:
        line = m.group(1)
        allergens = m.group(2).replace(',', '').split()
    ingredients = line.split()
    return ingredients, allergens


def solve_p1(lines: List[str], for_part_1=True) -> Union[int, dict]:
    """Solution to the 1st part of the challenge"""

    allergen_in_ingredients = {}
    ingredients = Counter()

    for line in lines:
        ingrs, allergens = parse_menu_line(line)
        ingredients.update(ingrs)
        for allrg in allergens:
            # this is unreadable therfore coprophagic therefore pythonic
            # allergen_in_ingredients.setdefault(
            #     allrg, set(ingrs)).intersection_update(ingrs)

            # a readable alternative
            if allrg in allergen_in_ingredients:
                allergen_in_ingredients[allrg].intersection_update(ingrs)
            else:
                allergen_in_ingredients[allrg] = set(ingrs)

    if DEBUG:
        print("\n--- All ingredients:\n", ingredients)
        print("--- Allergens in ingredients:\n", allergen_in_ingredients)

    allergic_ingredients = set()
    for ingrs in allergen_in_ingredients.values():
        allergic_ingredients.update(ingrs)
    if DEBUG:
        print("--- Allergic ingredients:\n", allergic_ingredients)

    nonallergic_ingredients = ingredients.keys() - allergic_ingredients
    if DEBUG:
        print("--- Nonallergic ingredients:\n", nonallergic_ingredients)

    if for_part_1:
        res = sum(ingredients[ingr] for ingr in nonallergic_ingredients)
    else:
        res = allergen_in_ingredients

    return res


def solve_p2(lines: List[str]) -> str:
    """Solution to the 2nd part of the challenge"""

    allergen_in_ingredients = solve_p1(lines, False)

    if DEBUG:
        print("Resolving ingredients to allergens")

    ingredient_with_allergen = {}  # resolved cases

    changed = True
    while changed:
        changed = False
        # find new unambiguous cases
        for allrg in list(allergen_in_ingredients.keys()):
            ingrs = allergen_in_ingredients[allrg]
            if len(ingrs) == 1:
                allergen_in_ingredients.pop(allrg)
                ingredient_with_allergen[ingrs.pop()] = allrg
                changed = True

        if changed:
            # remove unambiguous cases
            known_ingredients = set(ingredient_with_allergen.keys())
            for ingrs in allergen_in_ingredients.values():
                ingrs.difference_update(known_ingredients)

    assert not allergen_in_ingredients, \
      f"Unresolved allergens remain: {allergen_in_ingredients}"

    if DEBUG:
        print("--- Disambiguated allergens:\n", ingredient_with_allergen)

    ingrs = map(lambda kv: kv[0],
                sorted(ingredient_with_allergen.items(),
                       key=lambda kv: kv[1]))
    res = ",".join(ingrs)

    return res


text_1 = """mxmxvkd kfcds sqjhc nhms (contains dairy, fish)
trh fvjkl sbzzf mxmxvkd (contains dairy)
sqjhc fvjkl (contains soy)
sqjhc mxmxvkd sbzzf (contains fish)"""


tests = [
    (text_1.split('\n'), 5, "mxmxvkd,sqjhc,fvjkl"),
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
    day = '21'
    lines = utils.load_input()

    print(f"--- Day {day} p.1 ---")
    exp1 = 2374
    res1 = solve_p1(lines)
    print(exp1 == res1, exp1, res1)

    print(f"--- Day {day} p.2 ---")
    exp2 = 'fbtqkzc,jbbsjh,cpttmnv,ccrbr,tdmqcl,vnjxjg,nlph,mzqjxq'
    res2 = solve_p2(lines)
    print(exp2 == res2, exp2, res2)


if __name__ == '__main__':
    run_tests()
    run_real()
