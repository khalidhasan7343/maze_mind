"""Let players choose a presentation before opening its level menu.

The home screen draws representative previews without creating a round, loading
models, or writing telemetry. It returns a view name to the CLI, which delegates
gameplay to the chosen interface and its existing session services.
"""

import pygame

from game.maze_gen import generate_maze
from .isometric_render import BG, BLUE, INK, LINE, MUTED, IsometricRenderer
from .rendering import _cell_center, _draw_exit_portal, _draw_maze, _draw_player_icon

VIEWS = ("classic", "isometric")


class ViewPicker:
    """Keep home-screen selection and click geometry together so mouse and keyboard
    users choose from the same two view options.
    """

    def __init__(self, surface, player_tag="player1", seed=None):
        """Prepare fonts and one preview maze so comparing views does not start a
        playable engine or repeatedly regenerate layouts on each frame.
        """
        self.surface = surface
        self.player_tag = player_tag
        self.selected_view = "isometric"
        self.fonts = {size: pygame.font.SysFont("Arial,DejaVu Sans", size, bold=size >= 24)
                      for size in (12, 14, 16, 18, 24, 32, 40)}
        self.maze = generate_maze(8, 8, seed=42 if seed is None else seed)
        self.renderer = IsometricRenderer()
        self.cards = {}
        self.buttons = {}
        self.close_rect = pygame.Rect(0, 0, 0, 0)

    def text(self, value, point, size=16, color=INK, *, center=False):
        """Reuse font sizing and alignment so titles, descriptions, and actions have
        consistent typography across both preview cards.
        """
        label = self.fonts[size].render(value, True, color)
        rect = label.get_rect(midtop=point) if center else label.get_rect(topleft=point)
        self.surface.blit(label, rect)

    def classic_preview(self, rect):
        """Use the real classic drawing helpers so the home preview shows the flat
        board the player will see after selecting Classic.
        """
        pygame.draw.rect(self.surface, (18, 22, 33), rect, border_radius=12)
        size = max(2, min((rect.width - 36) // 8, (rect.height - 28) // 8))
        x, y = rect.centerx - size * 4, rect.centery - size * 4
        _draw_maze(self.surface, self.maze, x, y, size, set(), [])
        px, py = _cell_center(self.maze.start, x, y, size)
        _draw_player_icon(self.surface, px, py, size, "S")
        ex, ey = _cell_center(self.maze.exit, x, y, size)
        _draw_exit_portal(self.surface, ex, ey, size)

    def draw(self):
        """Draw selectable view previews and refresh their hit areas so the layout
        remains usable when the home window is resized.
        """
        surface = self.surface
        w, h = surface.get_size()
        surface.fill(BG)
        self.cards.clear()
        self.buttons.clear()
        pygame.draw.rect(surface, BLUE, (48, 34, 24, 24), border_radius=7)
        self.text("m", (60, 34), 18, (255, 255, 255), center=True)
        self.text("maze mind", (84, 29), 24)
        self.text("HOME", (238, 38), 12, MUTED)
        self.text(self.player_tag[:22], (w - 250, 37), 14, MUTED)
        self.close_rect = pygame.Rect(w - 128, 29, 80, 36)
        pygame.draw.rect(surface, (255, 255, 255), self.close_rect, border_radius=10)
        pygame.draw.rect(surface, LINE, self.close_rect, 1, border_radius=10)
        self.text("Close", (self.close_rect.centerx, 38), 16, center=True)
        pygame.draw.line(surface, LINE, (48, 88), (w - 48, 88))
        self.text("Choose your view.", (w // 2, 122), 40, center=True)
        self.text("Pick a view, then choose your level.", (w // 2, 178), 18, MUTED, center=True)

        gap = 28
        width = (w - 96 - gap) // 2
        for index, view in enumerate(VIEWS):
            card = pygame.Rect(48 + index * (width + gap), 235, width, h - 330)
            selected = self.selected_view == view
            pygame.draw.rect(surface, (235, 237, 242), card.move(0, 3), border_radius=16)
            pygame.draw.rect(surface, (255, 255, 255), card, border_radius=16)
            pygame.draw.rect(surface, BLUE if selected else LINE, card, 2 if selected else 1, border_radius=16)
            self.cards[view] = card
            self.text(view.capitalize(), (card.x + 24, card.y + 24), 32)
            description = "A clear, top-down maze." if view == "classic" else "Raised walls and a fresh perspective."
            self.text(description, (card.x + 24, card.y + 70), 16, MUTED)
            preview = pygame.Rect(card.x + 22, card.y + 108, card.width - 44, card.height - 205)
            if view == "classic":
                self.classic_preview(preview)
            else:
                self.renderer.draw(surface, preview, self.maze, player=self.maze.start)
            button = pygame.Rect(card.x + 24, card.bottom - 70,
                                 card.width - 48, 46)
            self.buttons[view] = button
            pygame.draw.rect(surface, BLUE if selected else (242, 243, 250), button, border_radius=10)
            self.text(f"Choose {view.capitalize()}", (button.centerx, button.y + 12), 16,
                      (255, 255, 255) if selected else INK, center=True)
        self.text("Click a view to continue, or use Left / Right and Enter.",
                  (w // 2, h - 58), 14, MUTED, center=True)

    def handle_event(self, event):
        """Return a view choice, an exit signal, or no action so the launcher can
        distinguish cancelling the home screen from choosing a game interface.
        """
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "quit"
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_TAB):
                self.selected_view = VIEWS[(VIEWS.index(self.selected_view) + 1) % len(VIEWS)]
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                return self.selected_view
        elif event.type == pygame.MOUSEMOTION:
            for view, rect in self.cards.items():
                if rect.collidepoint(event.pos):
                    self.selected_view = view
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.close_rect.collidepoint(event.pos):
                return "quit"
            for view, rect in self.cards.items():
                if rect.collidepoint(event.pos):
                    return view
        return None


def choose_view(player_tag="player1", seed=None):
    """Run the home window and return the selected view, or None on cancellation,
    so closing Home never starts a game or creates a completed-round record.
    """
    pygame.init()
    try:
        surface = pygame.display.set_mode((1280, 840), pygame.RESIZABLE)
        pygame.display.set_caption("Maze Mind / Home")
        picker = ViewPicker(surface, player_tag, seed)
        clock = pygame.time.Clock()
        picker.draw()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    picker.surface = pygame.display.set_mode((max(1080, event.w), max(760, event.h)), pygame.RESIZABLE)
                    picker.draw()
                else:
                    action = picker.handle_event(event)
                    if action == "quit":
                        return None
                    if action in VIEWS:
                        return action
            picker.draw()
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()
