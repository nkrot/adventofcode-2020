
from typing import List


def load_input(fname: str = 'input.txt') -> List[str]:
    lines = []
    with open(fname) as fd:
        for line in fd:
            lines.append(line.strip())
    return lines
