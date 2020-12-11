
from typing import List


class Square(object):

    def __init__(self, value=None, pos=None):
        self.position = pos or (-1, -1)
        self.value = value


class Board(object):
    PIECE = Square

    OFFSETS_AROUND = [(-1, 0), (-1, +1), (0, +1), (+1, +1),
                      (+1, 0), (+1, -1), (0, -1), (-1, -1)]

    @classmethod
    def from_text(cls, text: str):
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]
        return cls.from_lines(lines)

    @classmethod
    def from_lines(cls, lines: List[str]):
        board = cls()
        for rid, line in enumerate(lines):
            for cid, char in enumerate(line):
                board._add_piece((rid, cid), char)
        return board

    def __init__(self):
        self.squares = {}

    def __getitem__(self, *args):
        """Return the object located at position (x,y) or None if it does exist
        or is off the board"""
        if len(args) == 1:
            x, y = args[0]
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError(f"Wrong number of arguments: {args}")
        return self.squares.get(x, {}).get(y, None)

    def __iter__(self):
        return self.BoardIterator(self)

    def size(self):
        """Return (height, width)"""
        height, width = len(self.squares), 0
        for k, vals in self.squares.items():
            width = len(vals)
            break
        return (height, width)

    def __str__(self):
        rows = []
        height, width = self.size()
        for x in range(height):
            row = "".join([str(self[(x, y)]) for y in range(width)])
            rows.append(row)
        return "\n".join(rows)

    def _add_piece(self, xy, value):
        piece = self.PIECE(value, xy)
        x, y = xy
        self.squares.setdefault(x, {})[y] = piece

    class BoardIterator(object):

        def __init__(self, board):
            self.board = board
            self.index = 0
            nrows, ncols = self.board.size()
            self.items = []
            for r in range(nrows):
                for c in range(ncols):
                    self.items.append((r,c))

        def __iter__(self):
            return self

        def __next__(self):
            try:
                x, y = self.items[self.index]
            except IndexError:
                raise StopIteration
            self.index += 1
            return self.board[x, y]
