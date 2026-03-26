# file: src/input_handler.py
import pygame
from .game import GameState
from .constants import DAS_DELAY, DAS_REPEAT


class InputHandler:
    """pygame 이벤트 → Game 액션 변환. DAS(지연 반복 입력) 포함."""

    def __init__(self, game):
        self.game = game
        self.das_dir    = 0      # -1(왼쪽) / 0(없음) / 1(오른쪽)
        self.das_timer  = 0
        self.das_active = False
        self.soft_drop_held = False

    def handle_events(self):
        """이벤트 큐 처리. False 반환 시 종료."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                self._on_keydown(event.key)
            if event.type == pygame.KEYUP:
                self._on_keyup(event.key)
        return True

    def update(self, dt):
        """프레임마다 DAS / 소프트 드롭 연속 처리."""
        g = self.game
        if g.state != GameState.PLAYING:
            self.das_dir = 0
            self.das_active = False
            return

        if self.das_dir != 0:
            self.das_timer += dt
            threshold = DAS_DELAY if not self.das_active else DAS_REPEAT
            if self.das_timer >= threshold:
                self.das_timer  = 0
                self.das_active = True
                if self.das_dir == -1:
                    g.move_left()
                else:
                    g.move_right()

        if self.soft_drop_held:
            g.soft_drop()

    # ── 내부 이벤트 핸들러 ─────────────────────────────────────────────────

    def _on_keydown(self, key):
        g = self.game

        # 항상 동작하는 키
        if key == pygame.K_r:
            g.start()
            self._reset_das()
            self.soft_drop_held = False
            return

        if g.state == GameState.START:
            if key == pygame.K_RETURN:
                g.start()
            return

        if key == pygame.K_p and g.state in (GameState.PLAYING, GameState.PAUSED):
            g.toggle_pause()
            return

        if g.state != GameState.PLAYING:
            return

        if key == pygame.K_LEFT:
            self.das_dir    = -1
            self.das_active = False
            self.das_timer  = 0
            g.move_left()
        elif key == pygame.K_RIGHT:
            self.das_dir    = 1
            self.das_active = False
            self.das_timer  = 0
            g.move_right()
        elif key == pygame.K_UP:
            g.rotate(clockwise=True)
        elif key == pygame.K_z:
            g.rotate(clockwise=False)
        elif key == pygame.K_DOWN:
            self.soft_drop_held = True
            g.soft_drop()
        elif key == pygame.K_SPACE:
            self.soft_drop_held = False
            g.hard_drop()
        elif key == pygame.K_c:
            g.hold()

    def _on_keyup(self, key):
        if key in (pygame.K_LEFT, pygame.K_RIGHT):
            self._reset_das()
        if key == pygame.K_DOWN:
            self.soft_drop_held = False

    def _reset_das(self):
        self.das_dir    = 0
        self.das_active = False
        self.das_timer  = 0
