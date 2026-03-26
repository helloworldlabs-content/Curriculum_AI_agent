# file: main.py
import pygame
import sys
from src.game import Game
from src.renderer import Renderer
from src.input_handler import InputHandler
from src.constants import SCREEN_W, SCREEN_H, FPS


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()

    game    = Game()
    renderer = Renderer(screen)
    handler  = InputHandler(game)

    while True:
        dt = clock.tick(FPS)

        if not handler.handle_events():
            break
        handler.update(dt)
        game.update(dt)
        renderer.draw(game)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
