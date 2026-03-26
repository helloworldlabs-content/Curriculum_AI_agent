# file: tests/test_board.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.board import new_board, fits, lock_piece, clear_lines, ghost_row
from src.piece import Piece
from src.constants import BOARD_COLS, BOARD_ROWS


def test_new_board_dimensions():
    board = new_board()
    assert len(board) == BOARD_ROWS
    assert all(len(row) == BOARD_COLS for row in board)


def test_new_board_all_zeros():
    board = new_board()
    assert all(cell == 0 for row in board for cell in row)


def test_fits_empty_board():
    board = new_board()
    p = Piece('I')
    assert fits(board, p.matrix, 0, 0)


def test_fits_right_oob():
    board = new_board()
    p = Piece('O')
    assert not fits(board, p.matrix, 0, BOARD_COLS)


def test_fits_bottom_oob():
    board = new_board()
    p = Piece('O')
    assert not fits(board, p.matrix, BOARD_ROWS, 0)


def test_fits_left_oob():
    board = new_board()
    p = Piece('O')
    assert not fits(board, p.matrix, 0, -1)


def test_fits_collision():
    board = new_board()
    board[19][0] = 1
    # O piece at (18,0) occupies (18,0)(18,1)(19,0)(19,1) → conflicts (19,0)
    assert not fits(board, [[2, 2], [2, 2]], 18, 0)


def test_lock_piece():
    board = new_board()
    p = Piece('O')
    p.row, p.col = 18, 0
    lock_piece(board, p)
    assert board[18][0] == 2
    assert board[18][1] == 2
    assert board[19][0] == 2
    assert board[19][1] == 2


def test_clear_lines_two_full_rows():
    board = new_board()
    for c in range(BOARD_COLS):
        board[BOARD_ROWS - 1][c] = 1
        board[BOARD_ROWS - 2][c] = 1
    cleared = clear_lines(board)
    assert cleared == 2
    assert all(c == 0 for c in board[0])
    assert all(c == 0 for c in board[1])


def test_clear_lines_no_clear():
    board = new_board()
    board[19][0] = 1   # 한 칸만 채움 → 제거 안 됨
    cleared = clear_lines(board)
    assert cleared == 0
    assert board[19][0] == 1


def test_clear_lines_tetris():
    board = new_board()
    for row in range(BOARD_ROWS - 4, BOARD_ROWS):
        for c in range(BOARD_COLS):
            board[row][c] = 1
    cleared = clear_lines(board)
    assert cleared == 4


def test_ghost_row_lands_at_bottom():
    board = new_board()
    p = Piece('O')
    p.row, p.col = 0, 0
    gr = ghost_row(board, p)
    # O는 2줄 높이 → 마지막 착지는 BOARD_ROWS - 2
    assert gr == BOARD_ROWS - 2


def test_ghost_row_blocked_by_piece():
    board = new_board()
    for c in range(BOARD_COLS):
        board[10][c] = 1   # row 10 전체 채움
    p = Piece('O')
    p.row, p.col = 0, 0
    gr = ghost_row(board, p)
    assert gr == 8   # O(2줄)가 row 8-9에 착지
