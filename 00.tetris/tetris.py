import pygame
import random
import sys

# --- Constants ---
SCREEN_W, SCREEN_H = 300, 600
BOARD_COLS, BOARD_ROWS = 10, 20
CELL = 28
BOARD_X, BOARD_Y = 20, 20
BOARD_W = BOARD_COLS * CELL
BOARD_H = BOARD_ROWS * CELL
 
SIDEBAR_X = BOARD_X + BOARD_W + 16

FPS = 60
FALL_INTERVAL = 500  # ms

COLORS = {
    0: (20, 20, 30),
    1: (0, 240, 240),   # I - cyan
    2: (240, 240, 0),   # O - yellow
    3: (160, 0, 240),   # T - purple
    4: (0, 240, 0),     # S - green
    5: (240, 0, 0),     # Z - red
    6: (0, 0, 240),     # J - blue
    7: (240, 160, 0),   # L - orange
}

GHOST_COLOR = (80, 80, 100)
GRID_COLOR = (40, 40, 55)
BORDER_COLOR = (100, 100, 140)
BG_COLOR = (12, 12, 20)
TEXT_COLOR = (220, 220, 255)

TETROMINOES = {
    'I': [[0,0,0,0],
          [1,1,1,1],
          [0,0,0,0],
          [0,0,0,0]],
    'O': [[2,2],
          [2,2]],
    'T': [[0,3,0],
          [3,3,3],
          [0,0,0]],
    'S': [[0,4,4],
          [4,4,0],
          [0,0,0]],
    'Z': [[5,5,0],
          [0,5,5],
          [0,0,0]],
    'J': [[6,0,0],
          [6,6,6],
          [0,0,0]],
    'L': [[0,0,7],
          [7,7,7],
          [0,0,0]],
}

SCORE_TABLE = {1: 100, 2: 300, 3: 500, 4: 800}


def rotate_cw(matrix):
    return [list(row) for row in zip(*matrix[::-1])]


def new_board():
    return [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]


class Piece:
    def __init__(self, shape_name=None):
        if shape_name is None:
            shape_name = random.choice(list(TETROMINOES.keys()))
        self.name = shape_name
        self.matrix = [row[:] for row in TETROMINOES[shape_name]]
        self.col = BOARD_COLS // 2 - len(self.matrix[0]) // 2
        self.row = 0

    def rotated(self):
        return rotate_cw(self.matrix)


def fits(board, matrix, row, col):
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
    for r, line in enumerate(piece.matrix):
        for c, cell in enumerate(line):
            if cell:
                board[piece.row + r][piece.col + c] = cell


def clear_lines(board):
    cleared = 0
    new_board_rows = [row for row in board if any(c == 0 for c in row)]
    cleared = BOARD_ROWS - len(new_board_rows)
    for _ in range(cleared):
        new_board_rows.insert(0, [0] * BOARD_COLS)
    board[:] = new_board_rows
    return cleared


def ghost_row(board, piece):
    drop = 0
    while fits(board, piece.matrix, piece.row + drop + 1, piece.col):
        drop += 1
    return piece.row + drop


def draw_cell(surface, r, c, color, origin_x, origin_y, alpha=255):
    x = origin_x + c * CELL
    y = origin_y + r * CELL
    rect = pygame.Rect(x + 1, y + 1, CELL - 2, CELL - 2)
    if alpha < 255:
        s = pygame.Surface((CELL - 2, CELL - 2), pygame.SRCALPHA)
        s.fill((*color, alpha))
        surface.blit(s, (x + 1, y + 1))
    else:
        pygame.draw.rect(surface, color, rect, border_radius=3)
        highlight = pygame.Rect(x + 2, y + 2, CELL // 3, CELL // 3)
        light = tuple(min(255, v + 80) for v in color)
        pygame.draw.rect(surface, light, highlight, border_radius=2)


def draw_board(surface, board):
    # border
    pygame.draw.rect(surface, BORDER_COLOR,
                     (BOARD_X - 2, BOARD_Y - 2, BOARD_W + 4, BOARD_H + 4), 2)
    # grid
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            rect = pygame.Rect(BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL, CELL)
            pygame.draw.rect(surface, GRID_COLOR, rect, 1)
            if board[r][c]:
                draw_cell(surface, r, c, COLORS[board[r][c]], BOARD_X, BOARD_Y)


def draw_piece(surface, piece, ghost_r=None):
    # ghost
    if ghost_r is not None and ghost_r != piece.row:
        for r, line in enumerate(piece.matrix):
            for c, cell in enumerate(line):
                if cell:
                    draw_cell(surface, ghost_r + r, c, GHOST_COLOR,
                              BOARD_X, BOARD_Y, alpha=120)
    # actual piece
    for r, line in enumerate(piece.matrix):
        for c, cell in enumerate(line):
            if cell:
                draw_cell(surface, piece.row + r, piece.col + c,
                          COLORS[cell], BOARD_X, BOARD_Y)


def draw_next(surface, font, next_piece):
    label = font.render("NEXT", True, TEXT_COLOR)
    surface.blit(label, (SIDEBAR_X, BOARD_Y))
    ox = SIDEBAR_X
    oy = BOARD_Y + 30
    for r, line in enumerate(next_piece.matrix):
        for c, cell in enumerate(line):
            if cell:
                draw_cell(surface, r, c, COLORS[cell], ox, oy)


def draw_score(surface, font, score, level, lines):
    y = BOARD_Y + 130
    for label, val in [("SCORE", score), ("LEVEL", level), ("LINES", lines)]:
        lbl = font.render(label, True, (160, 160, 200))
        val_s = font.render(str(val), True, TEXT_COLOR)
        surface.blit(lbl, (SIDEBAR_X, y))
        surface.blit(val_s, (SIDEBAR_X, y + 20))
        y += 60


def draw_controls(surface, small_font):
    controls = [
        "← → : 이동",
        "↑    : 회전",
        "↓    : 내리기",
        "Space: 즉시 낙하",
        "P    : 일시정지",
    ]
    y = BOARD_Y + 380
    for line in controls:
        s = small_font.render(line, True, (120, 120, 160))
        surface.blit(s, (SIDEBAR_X, y))
        y += 18


def show_overlay(surface, font, big_font, text, sub=""):
    overlay = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (BOARD_X, BOARD_Y))
    t = big_font.render(text, True, (255, 255, 100))
    surface.blit(t, t.get_rect(center=(BOARD_X + BOARD_W // 2,
                                       BOARD_Y + BOARD_H // 2 - 20)))
    if sub:
        s = font.render(sub, True, TEXT_COLOR)
        surface.blit(s, s.get_rect(center=(BOARD_X + BOARD_W // 2,
                                           BOARD_Y + BOARD_H // 2 + 20)))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("malgun gothic", 16)
        small_font = pygame.font.SysFont("malgun gothic", 13)
        big_font = pygame.font.SysFont("malgun gothic", 32, bold=True)
    except Exception:
        font = pygame.font.SysFont(None, 18)
        small_font = pygame.font.SysFont(None, 14)
        big_font = pygame.font.SysFont(None, 36, bold=True)

    def new_game():
        board = new_board()
        current = Piece()
        nxt = Piece()
        score = 0
        level = 1
        total_lines = 0
        fall_timer = 0
        game_over = False
        paused = False
        return board, current, nxt, score, level, total_lines, fall_timer, game_over, paused

    board, current, nxt, score, level, total_lines, fall_timer, game_over, paused = new_game()

    das_delay = 170  # ms
    das_repeat = 50  # ms
    das_timer = 0
    das_dir = 0
    das_active = False

    while True:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    board, current, nxt, score, level, total_lines, fall_timer, game_over, paused = new_game()
                    das_dir = 0
                    das_active = False
                    continue

                if game_over:
                    continue

                if event.key == pygame.K_p:
                    paused = not paused

                if paused:
                    continue

                if event.key == pygame.K_LEFT:
                    das_dir = -1
                    das_active = False
                    das_timer = 0
                    if fits(board, current.matrix, current.row, current.col - 1):
                        current.col -= 1
                elif event.key == pygame.K_RIGHT:
                    das_dir = 1
                    das_active = False
                    das_timer = 0
                    if fits(board, current.matrix, current.row, current.col + 1):
                        current.col += 1
                elif event.key == pygame.K_UP:
                    rot = current.rotated()
                    for kick in [0, -1, 1, -2, 2]:
                        if fits(board, rot, current.row, current.col + kick):
                            current.matrix = rot
                            current.col += kick
                            break
                elif event.key == pygame.K_DOWN:
                    if fits(board, current.matrix, current.row + 1, current.col):
                        current.row += 1
                        score += 1
                elif event.key == pygame.K_SPACE:
                    gr = ghost_row(board, current)
                    score += 2 * (gr - current.row)
                    current.row = gr
                    lock_piece(board, current)
                    cleared = clear_lines(board)
                    if cleared:
                        score += SCORE_TABLE.get(cleared, 0) * level
                        total_lines += cleared
                        level = total_lines // 10 + 1
                    current = nxt
                    nxt = Piece()
                    fall_timer = 0
                    if not fits(board, current.matrix, current.row, current.col):
                        game_over = True

            if event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    das_dir = 0
                    das_active = False
                    das_timer = 0

        if not game_over and not paused:
            # DAS (delayed auto shift)
            if das_dir != 0:
                das_timer += dt
                threshold = das_delay if not das_active else das_repeat
                if das_timer >= threshold:
                    das_timer = 0
                    das_active = True
                    if fits(board, current.matrix, current.row, current.col + das_dir):
                        current.col += das_dir

            # gravity
            fall_speed = max(50, FALL_INTERVAL - (level - 1) * 45)
            fall_timer += dt
            if fall_timer >= fall_speed:
                fall_timer = 0
                if fits(board, current.matrix, current.row + 1, current.col):
                    current.row += 1
                else:
                    lock_piece(board, current)
                    cleared = clear_lines(board)
                    if cleared:
                        score += SCORE_TABLE.get(cleared, 0) * level
                        total_lines += cleared
                        level = total_lines // 10 + 1
                    current = nxt
                    nxt = Piece()
                    fall_timer = 0
                    if not fits(board, current.matrix, current.row, current.col):
                        game_over = True

        # --- draw ---
        screen.fill(BG_COLOR)
        draw_board(screen, board)

        if not game_over:
            gr = ghost_row(board, current)
            draw_piece(screen, current, ghost_r=gr)

        draw_next(screen, font, nxt)
        draw_score(screen, font, score, level, total_lines)
        draw_controls(screen, small_font)

        r_hint = small_font.render("R: 재시작", True, (120, 120, 160))
        screen.blit(r_hint, (SIDEBAR_X, SCREEN_H - 30))

        if paused:
            show_overlay(screen, font, big_font, "PAUSE", "P 키로 계속")
        if game_over:
            show_overlay(screen, font, big_font, "GAME OVER", "R 키로 재시작")

        pygame.display.flip()


if __name__ == "__main__":
    main()
