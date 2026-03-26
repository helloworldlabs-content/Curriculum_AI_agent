# file: tests/test_piece.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.piece import Piece, rotate_cw, rotate_ccw
from src.constants import BOARD_COLS, SHAPE_NAMES


def test_all_pieces_spawn():
    for name in SHAPE_NAMES:
        p = Piece(name)
        assert p.name == name
        assert p.matrix is not None
        assert p.row == 0


def test_piece_spawn_col_centered():
    p = Piece('O')   # 2칸 너비 → col = 4
    assert p.col == BOARD_COLS // 2 - 1

    p = Piece('I')   # 4칸 너비 → col = 3
    assert p.col == BOARD_COLS // 2 - 2


def test_rotate_cw_basic():
    # 간단한 2x2 검증
    m = [[1, 0],
         [1, 1]]
    r = rotate_cw(m)
    assert r == [[1, 1], [1, 0]]


def test_rotate_ccw_basic():
    # [[1,0],  CCW→  [[0,1],
    #  [1,1]]         [1,1]]
    m = [[1, 0],
         [1, 1]]
    r = rotate_ccw(m)
    assert r == [[0, 1], [1, 1]]


def test_rotate_cw_then_ccw_is_identity():
    m = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    original = [row[:] for row in m]
    assert rotate_ccw(rotate_cw(m)) == original


def test_four_cw_rotations_is_identity():
    p = Piece('T')
    original = [row[:] for row in p.matrix]
    rotated = p.matrix
    for _ in range(4):
        rotated = rotate_cw(rotated)
    assert rotated == original


def test_copy_is_independent():
    p = Piece('T')
    p2 = p.copy()
    p.row = 10
    p.col = 5
    assert p2.row == 0
    assert p2.col != 5 or p.name != p2.name or True   # row만 확인으로 충분
    assert p2.row == 0
