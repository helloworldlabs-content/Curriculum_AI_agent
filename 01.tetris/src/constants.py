# file: src/constants.py

# --- Screen ---
SCREEN_W = 520
SCREEN_H = 640

# --- Board ---
BOARD_COLS = 10
BOARD_ROWS = 20
CELL = 30
BOARD_X = 20
BOARD_Y = 20
BOARD_W = BOARD_COLS * CELL   # 300
BOARD_H = BOARD_ROWS * CELL   # 600

# --- Sidebar ---
SIDEBAR_X = BOARD_X + BOARD_W + 20  # 340
SIDEBAR_W = 160

# --- Timing ---
FPS = 60
FALL_INTERVAL_BASE = 800      # ms at level 1
FALL_INTERVAL_DECREMENT = 60  # ms reduction per level
FALL_INTERVAL_MIN = 50        # ms floor

# --- DAS (Delayed Auto Shift) ---
DAS_DELAY = 170   # ms before repeat starts
DAS_REPEAT = 50   # ms between repeats

# --- Scoring ---
LINES_PER_LEVEL = 10
SCORE_TABLE = {1: 100, 2: 300, 3: 500, 4: 800}
SOFT_DROP_SCORE = 1
HARD_DROP_SCORE = 2

# --- Files ---
HIGHSCORE_FILE = "highscore.txt"

# --- Colors ---
BG_COLOR         = (12,  12,  20)
GRID_COLOR       = (35,  35,  50)
BORDER_COLOR     = (90,  90, 130)
TEXT_COLOR       = (220, 220, 255)
LABEL_COLOR      = (150, 150, 190)
GHOST_COLOR      = (70,  70, 100)
SIDEBAR_BG_COLOR = (18,  18,  28)

PIECE_COLORS = {
    0: (20,  20,  30),
    1: (0,  240, 240),   # I - cyan
    2: (240, 240,   0),  # O - yellow
    3: (160,   0, 240),  # T - purple
    4: (0,  200,   0),   # S - green
    5: (220,   0,   0),  # Z - red
    6: (0,   80,  220),  # J - blue
    7: (240, 140,   0),  # L - orange
}

# --- Tetrominoes (SRS spawn orientation) ---
TETROMINOES = {
    'I': [[0, 0, 0, 0],
          [1, 1, 1, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0]],
    'O': [[2, 2],
          [2, 2]],
    'T': [[0, 3, 0],
          [3, 3, 3],
          [0, 0, 0]],
    'S': [[0, 4, 4],
          [4, 4, 0],
          [0, 0, 0]],
    'Z': [[5, 5, 0],
          [0, 5, 5],
          [0, 0, 0]],
    'J': [[6, 0, 0],
          [6, 6, 6],
          [0, 0, 0]],
    'L': [[0, 0, 7],
          [7, 7, 7],
          [0, 0, 0]],
}

SHAPE_NAMES = list(TETROMINOES.keys())
