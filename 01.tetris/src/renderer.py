# file: src/renderer.py
import pygame
from .constants import (
    CELL, BOARD_X, BOARD_Y, BOARD_W, BOARD_H, BOARD_COLS, BOARD_ROWS,
    SIDEBAR_X, SIDEBAR_W, SCREEN_W, SCREEN_H,
    BG_COLOR, GRID_COLOR, BORDER_COLOR, TEXT_COLOR, LABEL_COLOR,
    GHOST_COLOR, PIECE_COLORS, SIDEBAR_BG_COLOR,
)
from .game import GameState

MINI_CELL = CELL - 6   # 24px — 사이드바 미니 피스용


class Renderer:
    def __init__(self, surface):
        self.surface = surface
        self._init_fonts()

    def _init_fonts(self):
        try:
            self.font       = pygame.font.SysFont("malgun gothic", 16)
            self.small_font = pygame.font.SysFont("malgun gothic", 13)
            self.big_font   = pygame.font.SysFont("malgun gothic", 34, bold=True)
            self.title_font = pygame.font.SysFont("malgun gothic", 52, bold=True)
        except Exception:
            self.font       = pygame.font.SysFont(None, 20)
            self.small_font = pygame.font.SysFont(None, 16)
            self.big_font   = pygame.font.SysFont(None, 40, bold=True)
            self.title_font = pygame.font.SysFont(None, 58, bold=True)

    # ── 메인 진입 ──────────────────────────────────────────────────────────

    def draw(self, game):
        self.surface.fill(BG_COLOR)
        if game.state == GameState.START:
            self._draw_start_screen()
        else:
            self._draw_sidebar(game)
            self._draw_board(game.board)
            if game.state != GameState.GAME_OVER:
                self._draw_piece(game.current, ghost_r=game.get_ghost_row())
            else:
                self._draw_piece(game.current)
            if game.state == GameState.PAUSED:
                self._draw_overlay("PAUSE", "P 키로 계속")
            elif game.state == GameState.GAME_OVER:
                self._draw_overlay("GAME OVER", "R 키로 재시작")
        pygame.display.flip()

    # ── 보드 / 피스 ────────────────────────────────────────────────────────

    def _draw_cell(self, r, c, color, origin_x, origin_y, alpha=255):
        x = origin_x + c * CELL
        y = origin_y + r * CELL
        if alpha < 255:
            s = pygame.Surface((CELL - 2, CELL - 2), pygame.SRCALPHA)
            s.fill((*color, alpha))
            self.surface.blit(s, (x + 1, y + 1))
        else:
            rect = pygame.Rect(x + 1, y + 1, CELL - 2, CELL - 2)
            pygame.draw.rect(self.surface, color, rect, border_radius=3)
            light = tuple(min(255, v + 80) for v in color)
            pygame.draw.rect(self.surface, light,
                             pygame.Rect(x + 2, y + 2, CELL // 3, CELL // 3),
                             border_radius=2)

    def _draw_board(self, board):
        pygame.draw.rect(self.surface, BORDER_COLOR,
                         (BOARD_X - 2, BOARD_Y - 2, BOARD_W + 4, BOARD_H + 4), 2)
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                pygame.draw.rect(self.surface, GRID_COLOR,
                                 pygame.Rect(BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL, CELL), 1)
                if board[r][c]:
                    self._draw_cell(r, c, PIECE_COLORS[board[r][c]], BOARD_X, BOARD_Y)

    def _draw_piece(self, piece, ghost_r=None):
        if ghost_r is not None and ghost_r != piece.row:
            for r, line in enumerate(piece.matrix):
                for c, cell in enumerate(line):
                    if cell:
                        self._draw_cell(ghost_r + r, c, GHOST_COLOR, BOARD_X, BOARD_Y, alpha=120)
        for r, line in enumerate(piece.matrix):
            for c, cell in enumerate(line):
                if cell:
                    self._draw_cell(piece.row + r, piece.col + c,
                                    PIECE_COLORS[cell], BOARD_X, BOARD_Y)

    # ── 사이드바 ───────────────────────────────────────────────────────────

    def _draw_sidebar(self, game):
        pygame.draw.rect(self.surface, SIDEBAR_BG_COLOR,
                         (SIDEBAR_X - 8, BOARD_Y - 2, SIDEBAR_W + 8, BOARD_H + 4))
        y = BOARD_Y

        self._label("NEXT", SIDEBAR_X, y);  y += 20
        self._mini_piece(game.next_piece, SIDEBAR_X, y);  y += 4 * MINI_CELL + 12

        self._label("HOLD", SIDEBAR_X, y);  y += 20
        if game.hold_piece:
            if game.can_hold:
                self._mini_piece(game.hold_piece, SIDEBAR_X, y)
            else:
                self._mini_piece_dimmed(game.hold_piece, SIDEBAR_X, y)
        y += 4 * MINI_CELL + 12

        for lbl, val in [("SCORE", game.score), ("BEST", game.highscore),
                         ("LEVEL", game.level), ("LINES", game.total_lines)]:
            self._stat(lbl, val, SIDEBAR_X, y);  y += 45

        y += 5
        controls = [
            "← → : 이동",
            "↑ : 회전(CW)",
            "Z : 회전(CCW)",
            "↓ : 소프트 드롭",
            "Space : 하드 드롭",
            "C : 홀드",
            "P : 일시정지",
            "R : 재시작",
        ]
        for line in controls:
            s = self.small_font.render(line, True, LABEL_COLOR)
            self.surface.blit(s, (SIDEBAR_X, y));  y += 17

    def _label(self, text, x, y):
        self.surface.blit(self.font.render(text, True, LABEL_COLOR), (x, y))

    def _stat(self, label, value, x, y):
        self.surface.blit(self.font.render(label,      True, LABEL_COLOR), (x, y))
        self.surface.blit(self.font.render(str(value), True, TEXT_COLOR),  (x, y + 18))

    def _mini_piece(self, piece, ox, oy, dimmed=False):
        if piece is None:
            return
        for r, line in enumerate(piece.matrix):
            for c, cell in enumerate(line):
                if cell:
                    color = PIECE_COLORS[cell] if not dimmed else (55, 55, 75)
                    x = ox + c * MINI_CELL
                    y = oy + r * MINI_CELL
                    rect = pygame.Rect(x + 1, y + 1, MINI_CELL - 2, MINI_CELL - 2)
                    pygame.draw.rect(self.surface, color, rect, border_radius=2)
                    if not dimmed:
                        light = tuple(min(255, v + 80) for v in color)
                        pygame.draw.rect(self.surface, light,
                                         pygame.Rect(x + 2, y + 2,
                                                     MINI_CELL // 3, MINI_CELL // 3),
                                         border_radius=1)

    def _mini_piece_dimmed(self, piece, ox, oy):
        self._mini_piece(piece, ox, oy, dimmed=True)

    # ── 오버레이 / 시작 화면 ───────────────────────────────────────────────

    def _draw_overlay(self, title, subtitle=""):
        overlay = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.surface.blit(overlay, (BOARD_X, BOARD_Y))
        cx = BOARD_X + BOARD_W // 2
        cy = BOARD_Y + BOARD_H // 2
        t = self.big_font.render(title, True, (255, 255, 80))
        self.surface.blit(t, t.get_rect(center=(cx, cy - 24)))
        if subtitle:
            s = self.font.render(subtitle, True, TEXT_COLOR)
            self.surface.blit(s, s.get_rect(center=(cx, cy + 16)))

    def _draw_start_screen(self):
        cx = SCREEN_W // 2
        cy = SCREEN_H // 2

        title = self.title_font.render("TETRIS", True, (100, 220, 255))
        self.surface.blit(title, title.get_rect(center=(cx, cy - 140)))

        keys = [
            ("← →",    "이동"),
            ("↑",       "시계 방향 회전"),
            ("Z",        "반시계 방향 회전"),
            ("↓",       "소프트 드롭"),
            ("Space",   "하드 드롭"),
            ("C",        "홀드"),
            ("P",        "일시정지"),
        ]
        y = cy - 60
        for key, desc in keys:
            k = self.font.render(f"{key:7s}", True, (240, 200, 80))
            d = self.font.render(desc,         True, TEXT_COLOR)
            kw = k.get_width()
            total = kw + 8 + d.get_width()
            self.surface.blit(k, (cx - total // 2,         y))
            self.surface.blit(d, (cx - total // 2 + kw + 8, y))
            y += 24

        enter = self.font.render("ENTER 키로 시작", True, (100, 240, 140))
        self.surface.blit(enter, enter.get_rect(center=(cx, y + 20)))
