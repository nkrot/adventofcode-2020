
from typing import List, Union, Tuple, Optional


def load_input(fname: Optional[str] = None) -> List[str]:
    """Load file, either given or default 'input.txt' and return its content
    as a list of lines. All lines are returned, including empty ones."""
    fname = fname or 'input.txt'
    lines = []
    with open(fname) as fd:
        for line in fd:
            lines.append(line.strip())
    return lines


def group_lines(data: Union[str, List[str]]) -> List[List[str]]:
    """Make groups of lines: a group is a sequence of lines that are separated
    by an empty line from another group.
    Input <data> is either of these:
    1) (str) a string that is the whole content of a file;
    2) (List[str]) a list of lines that 1) already split into individual lines.
    """
    groups = [[]]
    if isinstance(data, str):
        data = [ln.strip() for ln in data.split('\n')]
    for ln in data:
        if ln:
            groups[-1].append(ln)
        else:
            groups.append([])
    return groups


def to_numbers(lines: List[str]) -> List[int]:
    """Convert list of lines (strings) to list of ints"""
    return [int(line) for line in lines]


def minmax(numbers: List[int]) -> Tuple[int, int]:
    """Return min and max values from given list of integers"""
    return (min(numbers), max(numbers))
