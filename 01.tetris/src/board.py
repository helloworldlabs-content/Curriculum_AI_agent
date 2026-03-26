# file: src/board.py
from .constants import BOARD_COLS, BOARD_ROWS


def new_board():
    return [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]


def fits(board, matrix, row, col):
    """matrix를 (row, col) 위치에 놓을 수 있는지 확인."""
    for r, line in enumerate(matrix):
        for c, cell in enumerate(line):
            if cell:
                nr, nc = row + r, col + c
                if nr < 0 or nr >= BOARD_ROWS or nc < 0 or nc >= BOARD_COLS:
                    return False
                if board[nr][nc]:
                    return False
    return True


def lock_piece(board, piece):
    """피스를 보드에 고정."""
    for r, line in enumerate(piece.matrix):
        for c, cell in enumerate(line):
            if cell:
                board[piece.row + r][piece.col + c] = cell


def clear_lines(board):
    """꽉 찬 줄을 제거하고 위에서 빈 줄을 추가. 제거된 줄 수 반환."""
    new_rows = [row for row in board if any(c == 0 for c in row)]
    cleared = BOARD_ROWS - len(new_rows)
    for _ in range(cleared):
        new_rows.insert(0, [0] * BOARD_COLS)
    board[:] = new_rows
    return cleared


def ghost_row(board, piece):
    """고스트 피스가 착지할 행 번호 반환."""
    drop = 0
    while fits(board, piece.matrix, piece.row + drop + 1, piece.col):
        drop += 1
    return piece.row + drop


def is_game_over(board, piece):
    """스폰 위치에서 충돌하면 게임 오버."""
    return not fits(board, piece.matrix, piece.row, piece.col)
