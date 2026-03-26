# file: src/piece.py
import random
from .constants import TETROMINOES, BOARD_COLS, SHAPE_NAMES


def rotate_cw(matrix):
    """시계 방향 90도 회전."""
    return [list(row) for row in zip(*matrix[::-1])]


def rotate_ccw(matrix):
    """반시계 방향 90도 회전."""
    return [list(row) for row in zip(*matrix)][::-1]


class Piece:
    def __init__(self, shape_name=None):
        if shape_name is None:
            shape_name = random.choice(SHAPE_NAMES)
        self.name = shape_name
        self.matrix = [row[:] for row in TETROMINOES[shape_name]]
        self.col = BOARD_COLS // 2 - len(self.matrix[0]) // 2
        self.row = 0

    def rotated_cw(self):
        return rotate_cw(self.matrix)

    def rotated_ccw(self):
        return rotate_ccw(self.matrix)

    def copy(self):
        p = Piece(self.name)
        p.matrix = [row[:] for row in self.matrix]
        p.row = self.row
        p.col = self.col
        return p
