import pygame
import random
import sys
import math
import time

# ─── Constants ────────────────────────────────────────────────────────────────
CELL = 20                  # pixel size of one cell
COLS, ROWS = 12, 22        # board dimensions
PANEL_W = 22               # right-panel character columns
FPS = 60

WIN_W = CELL * (COLS + PANEL_W)
WIN_H = CELL * ROWS

# Terminal palette
BLACK  = (0,   0,   0)
GREEN  = (0,   255, 70)
DKGRN  = (0,   100, 30)
DIMGRN = (0,   50,  15)
BGGRN  = (0,   8,   2)
WHITE  = (180, 255, 180)
AMBER  = (255, 180, 0)

# ─── Tetrominoes ──────────────────────────────────────────────────────────────
SHAPES = {
    'I': [[(0,1),(1,1),(2,1),(3,1)],
          [(2,0),(2,1),(2,2),(2,3)]],
    'O': [[(1,0),(2,0),(1,1),(2,1)]],
    'T': [[(0,1),(1,1),(2,1),(1,0)],
          [(1,0),(1,1),(1,2),(2,1)],
          [(0,1),(1,1),(2,1),(1,2)],
          [(1,0),(1,1),(1,2),(0,1)]],
    'S': [[(1,0),(2,0),(0,1),(1,1)],
          [(1,0),(1,1),(2,1),(2,2)]],
    'Z': [[(0,0),(1,0),(1,1),(2,1)],
          [(2,0),(1,1),(2,1),(1,2)]],
    'J': [[(0,0),(0,1),(1,1),(2,1)],
          [(1,0),(2,0),(1,1),(1,2)],
          [(0,1),(1,1),(2,1),(2,2)],
          [(1,0),(1,1),(0,2),(1,2)]],
    'L': [[(2,0),(0,1),(1,1),(2,1)],
          [(1,0),(1,1),(1,2),(2,2)],
          [(0,1),(1,1),(2,1),(0,2)],
          [(0,0),(1,0),(1,1),(1,2)]],
}
PIECE_ORDER = list(SHAPES.keys())


# ─── Font helper ──────────────────────────────────────────────────────────────
class TermFont:
    """Manages a monospace font sized to one cell."""
    def __init__(self, cell):
        pygame.font.init()
        # Try to use a monospace font; fall back to default
        for name in ("Courier New", "Consolas", "Lucida Console", "monospace"):
            try:
                self.font = pygame.font.SysFont(name, cell, bold=True)
                break
            except Exception:
                pass
        else:
            self.font = pygame.font.Font(None, cell)
        # Measure actual glyph size
        w, h = self.font.size("█")
        self.cw = w
        self.ch = h

    def render(self, surf, text, x, y, color=GREEN, alpha=255):
        s = self.font.render(text, True, color)
        if alpha < 255:
            s.set_alpha(alpha)
        surf.blit(s, (x, y))

    def size(self, text):
        return self.font.size(text)


# ─── Piece ────────────────────────────────────────────────────────────────────
class Piece:
    def __init__(self, kind=None):
        self.kind = kind or random.choice(PIECE_ORDER)
        self.rotations = SHAPES[self.kind]
        self.rot = 0
        self.x = COLS // 2 - 2
        self.y = 0

    @property
    def cells(self):
        return [(self.x + dx, self.y + dy) for dx, dy in self.rotations[self.rot]]

    def rotated(self, dr=1):
        r = (self.rot + dr) % len(self.rotations)
        return [(self.x + dx, self.y + dy) for dx, dy in self.rotations[r]], r


# ─── Board ────────────────────────────────────────────────────────────────────
class Board:
    def __init__(self):
        self.grid = [[0] * COLS for _ in range(ROWS)]

    def valid(self, cells):
        for cx, cy in cells:
            if cx < 0 or cx >= COLS:          # 좌우 벽
                return False
            if cy >= ROWS:                     # 바닥
                return False
            if cy < 0:                         # 천장 위 (스폰 버퍼) — 충돌 없이 허용
                continue
            if self.grid[cy][cx]:
                return False
        return True

    def lock(self, cells):
        for cx, cy in cells:
            if 0 <= cy < ROWS:
                self.grid[cy][cx] = 1

    def clear_lines(self):
        new_grid = [row for row in self.grid if not all(row)]
        cleared = ROWS - len(new_grid)
        # [[0]*COLS]*N 은 같은 리스트 객체를 N번 참조 → 한 칸 수정 시 전 줄 오염
        self.grid = [[0]*COLS for _ in range(cleared)] + new_grid
        return cleared


# ─── CRT effect surfaces ──────────────────────────────────────────────────────
def make_scanline_surf(w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, 2):
        pygame.draw.line(s, (0, 0, 0, 60), (0, y), (w, y))
    return s

def make_vignette_surf(w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w // 2, h // 2
    max_r = math.sqrt(cx**2 + cy**2)
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            d = math.sqrt((x - cx)**2 + (y - cy)**2) / max_r
            a = int(min(255, d * d * 180))
            pygame.draw.rect(s, (0, 0, 0, a), (x, y, 4, 4))
    return s


# ─── Game ─────────────────────────────────────────────────────────────────────
class TetrisGame:
    SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
    LEVEL_SPEED = [800, 700, 600, 500, 400, 320, 260, 200, 150, 100]

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("TETRIS // TERMINAL v1.0")
        self.clock = pygame.time.Clock()
        self.font = TermFont(CELL)

        self.scanlines = make_scanline_surf(WIN_W, WIN_H)
        self.vignette  = make_vignette_surf(WIN_W, WIN_H)

        self.reset()

    def reset(self):
        self.board   = Board()
        self.score   = 0
        self.level   = 1
        self.lines   = 0
        self.piece   = Piece()
        self.next_p  = Piece()
        self.held    = None
        self.can_hold= True
        self.game_over = False
        self.paused  = False
        self.fall_timer = 0
        self.flash_rows = []
        self.flash_timer = 0
        self.blink_phase = 0   # 0..FPS, for CRT cursor blink
        self.ghost_cells = []
        self._update_ghost()

    # ── Ghost piece ──────────────────────────────────────────────────────────
    def _update_ghost(self):
        tmp = Piece(self.piece.kind)
        tmp.rot = self.piece.rot
        tmp.x   = self.piece.x
        tmp.y   = self.piece.y
        while self.board.valid([(cx, cy+1) for cx, cy in tmp.cells]):
            tmp.y += 1
        self.ghost_cells = tmp.cells

    # ── Input ────────────────────────────────────────────────────────────────
    def handle_keys(self, event):
        if event.type != pygame.KEYDOWN:
            return
        key = event.key

        if key == pygame.K_ESCAPE:
            pygame.quit(); sys.exit()

        if self.game_over:
            if key == pygame.K_r:
                self.reset()
            return

        if key == pygame.K_p:
            self.paused = not self.paused
            return

        if self.paused:
            return

        if key in (pygame.K_LEFT, pygame.K_a):
            cells = [(cx-1, cy) for cx, cy in self.piece.cells]
            if self.board.valid(cells):
                self.piece.x -= 1
                self._update_ghost()
        elif key in (pygame.K_RIGHT, pygame.K_d):
            cells = [(cx+1, cy) for cx, cy in self.piece.cells]
            if self.board.valid(cells):
                self.piece.x += 1
                self._update_ghost()
        elif key in (pygame.K_DOWN, pygame.K_s):
            self._try_drop()
        elif key in (pygame.K_UP, pygame.K_w):
            self._try_rotate()
        elif key == pygame.K_z:
            self._try_rotate(dr=-1)
        elif key == pygame.K_SPACE:
            self._hard_drop()
        elif key == pygame.K_c:
            self._hold()

    def _try_drop(self):
        cells = [(cx, cy+1) for cx, cy in self.piece.cells]
        if self.board.valid(cells):
            self.piece.y += 1
            self.score += 1
            self._update_ghost()
            self.fall_timer = 0
        else:
            self._lock_piece()

    def _try_rotate(self, dr=1):
        cells, new_rot = self.piece.rotated(dr)
        # Wall-kick: x 방향 우선, y는 보드 상단 근처에서만 위로 밀기
        for dx in (0, -1, 1, -2, 2):
            for dy in (0, -1, -2):            # 상단 근처 천장 kick 지원
                kicked = [(cx+dx, cy+dy) for cx, cy in cells]
                if self.board.valid(kicked):
                    self.piece.x += dx
                    self.piece.y += dy
                    self.piece.rot = new_rot
                    self._update_ghost()
                    return

    def _hard_drop(self):
        while True:
            cells = [(cx, cy+1) for cx, cy in self.piece.cells]
            if self.board.valid(cells):
                self.piece.y += 1
                self.score += 2
            else:
                break
        self._lock_piece()

    def _hold(self):
        if not self.can_hold:
            return
        if self.held is None:
            self.held = self.piece.kind
            self._spawn_next()
        else:
            self.held, self.piece = self.piece.kind, Piece(self.held)
            self.piece.x = COLS//2 - 2
            self.piece.y = 0
        self.can_hold = False
        self._update_ghost()

    def _spawn_next(self):
        self.piece  = self.next_p
        self.next_p = Piece()
        self.can_hold = True
        if not self.board.valid(self.piece.cells):
            self.game_over = True

    def _lock_piece(self):
        self.board.lock(self.piece.cells)
        cleared = self.board.clear_lines()
        if cleared:
            self.score += self.SCORES.get(cleared, 0) * self.level
            self.lines += cleared
            self.level  = min(10, self.lines // 10 + 1)
        self._spawn_next()
        self._update_ghost()

    # ── Update ───────────────────────────────────────────────────────────────
    def update(self, dt):
        if self.game_over or self.paused:
            return
        self.blink_phase = (self.blink_phase + dt * 0.001) % 1.0
        speed = self.LEVEL_SPEED[min(self.level-1, len(self.LEVEL_SPEED)-1)]
        self.fall_timer += dt
        if self.fall_timer >= speed:
            self.fall_timer = 0
            self._try_drop()

    # ── Draw helpers ─────────────────────────────────────────────────────────
    def _px(self, col): return col * CELL
    def _py(self, row): return row * CELL
    def _board_x(self): return 0
    def _panel_x(self): return COLS * CELL

    def _draw_cell(self, surf, col, row, color, char="█"):
        x = self._px(col)
        y = self._py(row)
        self.font.render(surf, char, x, y, color)

    def _draw_border(self, surf):
        # Left/right/bottom border using box-drawing chars
        for r in range(ROWS):
            self.font.render(surf, "│", self._px(-1), self._py(r), DKGRN)
            self.font.render(surf, "│", self._px(COLS), self._py(r), DKGRN)
        for c in range(-1, COLS+1):
            self.font.render(surf, "─", self._px(c), self._py(ROWS), DKGRN)
        self.font.render(surf, "└", self._px(-1),   self._py(ROWS), DKGRN)
        self.font.render(surf, "┘", self._px(COLS),  self._py(ROWS), DKGRN)

    def _draw_board(self, surf):
        # Background grid dots
        for r in range(ROWS):
            for c in range(COLS):
                self.font.render(surf, "·", self._px(c), self._py(r), DIMGRN)
        # Locked cells
        for r, row in enumerate(self.board.grid):
            for c, v in enumerate(row):
                if v:
                    self.font.render(surf, "█", self._px(c), self._py(r), GREEN)
        # Ghost
        for cx, cy in self.ghost_cells:
            if 0 <= cy < ROWS:
                self.font.render(surf, "░", self._px(cx), self._py(cy), DKGRN)
        # Active piece
        for cx, cy in self.piece.cells:
            if 0 <= cy < ROWS:
                self.font.render(surf, "█", self._px(cx), self._py(cy), WHITE)

    def _panel_text(self, surf, line, col_offset, row, color=GREEN):
        """Render text in panel; col_offset is char offset from panel left."""
        px = self._panel_x() + col_offset * self.font.cw
        py = self._py(row)
        self.font.render(surf, line, px, py, color)

    def _draw_mini_piece(self, surf, kind, start_col, start_row):
        if kind is None:
            return
        cells = SHAPES[kind][0]
        for dx, dy in cells:
            self.font.render(surf,
                "█",
                self._panel_x() + (start_col + dx) * self.font.cw,
                self._py(start_row + dy),
                GREEN)

    def _draw_panel(self, surf):
        px = self._panel_x()
        cw = self.font.cw

        # ── Header ──────────────────────────────────────────────────
        header = " TETRIS v1.0 "
        self.font.render(surf, "┌" + "─"*len(header) + "┐", px, self._py(0), DKGRN)
        self.font.render(surf, "│" + header + "│",           px, self._py(1), GREEN)
        self.font.render(surf, "└" + "─"*len(header) + "┘", px, self._py(2), DKGRN)

        # ── Score ────────────────────────────────────────────────────
        self.font.render(surf, "> SCORE",          px, self._py(4),  DKGRN)
        self.font.render(surf, f"  {self.score:08d}", px, self._py(5), AMBER)

        self.font.render(surf, "> LEVEL",          px, self._py(7),  DKGRN)
        self.font.render(surf, f"  {self.level:02d}",   px, self._py(8), GREEN)

        self.font.render(surf, "> LINES",          px, self._py(10), DKGRN)
        self.font.render(surf, f"  {self.lines:04d}",   px, self._py(11), GREEN)

        # ── Next ─────────────────────────────────────────────────────
        self.font.render(surf, "> NEXT",           px, self._py(13), DKGRN)
        self._draw_mini_piece(surf, self.next_p.kind, 2, 14)

        # ── Hold ─────────────────────────────────────────────────────
        self.font.render(surf, "> HOLD",           px, self._py(17), DKGRN)
        if self.held:
            col = DKGRN if not self.can_hold else GREEN
            self._draw_mini_piece(surf, self.held, 2, 18)
        else:
            self.font.render(surf, "  NONE", px, self._py(18), DIMGRN)

        # ── Controls ─────────────────────────────────────────────────
        controls = [
            "> CONTROLS",
            "  ←→   MOVE",
            "  ↑/W  ROTATE",
            "  Z    ROT CCW",
            "  ↓/S  SOFT DROP",
            "  SPC  HARD DROP",
            "  C    HOLD",
            "  P    PAUSE",
            "  R    RESTART",
            "  ESC  QUIT",
        ]
        for i, line in enumerate(controls):
            color = DKGRN if line.startswith(">") else DIMGRN
            self.font.render(surf, line, px, self._py(ROWS - len(controls) + i), color)

    def _draw_overlay(self, surf):
        """Pause / Game-over overlay."""
        overlay = pygame.Surface((COLS*CELL, ROWS*CELL), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surf.blit(overlay, (0, 0))

        if self.game_over:
            lines = ["GAME  OVER", f"SCORE: {self.score}", "", "[R] RESTART", "[ESC] QUIT"]
            color = AMBER
        else:
            blink = math.sin(time.time() * 4) > 0
            lines = ["PAUSED", "", "[P] RESUME" if blink else ""]
            color = GREEN

        start_r = ROWS // 2 - len(lines) // 2
        for i, line in enumerate(lines):
            tw, _ = self.font.size(line)
            x = (COLS*CELL - tw) // 2
            y = self._py(start_r + i)
            self.font.render(surf, line, x, y, color)

    # ── CRT flicker ──────────────────────────────────────────────────────────
    def _crt_flicker(self, surf):
        # Subtle brightness pulse
        flicker = 0.97 + 0.03 * math.sin(time.time() * 30)
        if flicker < 1.0:
            dim = pygame.Surface((WIN_W, WIN_H))
            dim.fill(BLACK)
            dim.set_alpha(int((1 - flicker) * 80))
            surf.blit(dim, (0, 0))

    # ── Main draw ────────────────────────────────────────────────────────────
    def draw(self):
        self.screen.fill(BGGRN)

        self._draw_border(self.screen)
        self._draw_board(self.screen)
        self._draw_panel(self.screen)

        if self.game_over or self.paused:
            self._draw_overlay(self.screen)

        # CRT effects on top
        self.screen.blit(self.scanlines, (0, 0))
        self._crt_flicker(self.screen)
        self.screen.blit(self.vignette, (0, 0))

        pygame.display.flip()

    # ── Main loop ────────────────────────────────────────────────────────────
    def run(self):
        while True:
            dt = self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                self.handle_keys(event)
            self.update(dt)
            self.draw()


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TetrisGame().run()
