# file: src/game.py
# pygame 비의존 순수 게임 로직 — 단위 테스트 가능
from enum import Enum
from .board import new_board, fits, lock_piece, clear_lines, ghost_row, is_game_over
from .piece import Piece
from .constants import (
    FALL_INTERVAL_BASE, FALL_INTERVAL_DECREMENT, FALL_INTERVAL_MIN,
    LINES_PER_LEVEL, SCORE_TABLE, SOFT_DROP_SCORE, HARD_DROP_SCORE,
    HIGHSCORE_FILE,
)


class GameState(Enum):
    START     = "start"
    PLAYING   = "playing"
    PAUSED    = "paused"
    GAME_OVER = "game_over"


def load_highscore():
    try:
        with open(HIGHSCORE_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def save_highscore(score):
    try:
        with open(HIGHSCORE_FILE, "w") as f:
            f.write(str(score))
    except Exception:
        pass


class Game:
    def __init__(self):
        self.highscore = load_highscore()
        self.state = GameState.START
        self._reset()

    # ── 공개 제어 ──────────────────────────────────────────────────────────

    def start(self):
        self._reset()
        self.state = GameState.PLAYING

    def toggle_pause(self):
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING

    def move_left(self):
        if fits(self.board, self.current.matrix, self.current.row, self.current.col - 1):
            self.current.col -= 1
            return True
        return False

    def move_right(self):
        if fits(self.board, self.current.matrix, self.current.row, self.current.col + 1):
            self.current.col += 1
            return True
        return False

    def rotate(self, clockwise=True):
        rot = self.current.rotated_cw() if clockwise else self.current.rotated_ccw()
        for kick in [0, -1, 1, -2, 2]:
            if fits(self.board, rot, self.current.row, self.current.col + kick):
                self.current.matrix = rot
                self.current.col += kick
                return True
        return False

    def soft_drop(self):
        if fits(self.board, self.current.matrix, self.current.row + 1, self.current.col):
            self.current.row += 1
            self.score += SOFT_DROP_SCORE
            return True
        return False

    def hard_drop(self):
        gr = ghost_row(self.board, self.current)
        self.score += HARD_DROP_SCORE * (gr - self.current.row)
        self.current.row = gr
        self._lock_current()

    def hold(self):
        if not self.can_hold:
            return
        if self.hold_piece is None:
            self.hold_piece = Piece(self.current.name)
            self._spawn_next()
        else:
            held_name = self.hold_piece.name
            self.hold_piece = Piece(self.current.name)
            self.current = Piece(held_name)
        self.can_hold = False

    def update(self, dt):
        """게임 루프마다 호출. dt: 경과 ms."""
        if self.state != GameState.PLAYING:
            return
        fall_speed = max(
            FALL_INTERVAL_MIN,
            FALL_INTERVAL_BASE - (self.level - 1) * FALL_INTERVAL_DECREMENT,
        )
        self.fall_timer += dt
        if self.fall_timer >= fall_speed:
            self.fall_timer = 0
            if fits(self.board, self.current.matrix, self.current.row + 1, self.current.col):
                self.current.row += 1
            else:
                self._lock_current()

    def get_ghost_row(self):
        return ghost_row(self.board, self.current)

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────

    def _reset(self):
        self.board       = new_board()
        self.current     = Piece()
        self.next_piece  = Piece()
        self.hold_piece  = None
        self.can_hold    = True
        self.score       = 0
        self.level       = 1
        self.total_lines = 0
        self.fall_timer  = 0

    def _spawn_next(self):
        self.current    = self.next_piece
        self.next_piece = Piece()
        self.fall_timer = 0
        self.can_hold   = True
        if is_game_over(self.board, self.current):
            self._end_game()

    def _lock_current(self):
        lock_piece(self.board, self.current)
        cleared = clear_lines(self.board)
        if cleared:
            self.score       += SCORE_TABLE.get(cleared, 0) * self.level
            self.total_lines += cleared
            self.level        = self.total_lines // LINES_PER_LEVEL + 1
        self._spawn_next()

    def _end_game(self):
        self.state = GameState.GAME_OVER
        if self.score > self.highscore:
            self.highscore = self.score
            save_highscore(self.highscore)
