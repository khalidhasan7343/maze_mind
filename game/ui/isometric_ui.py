"""Provide the default light-themed level menu, gameplay, pause, and results screens. The
controller delegates rules and persistence to existing services so the new view remains
part of the same game.

A light, minimal isometric interface alongside the classic Pygame view.
"""

import time

import pygame

from data.logger import RoundLogger
from game.engine import Outcome, RoundConfig
from game.maze_gen import generate_maze
from game.session import RoundSession, TIERS, configure_round
from .controls import HeldMovement, KEY_TO_DIR, move_toward_cell, radar_guide
from .isometric_render import BG, BLUE, GREEN, INK, LINE, MUTED, RED, IsometricRenderer

LEVEL_COPY = {
    "Recruit": ("Find your rhythm", "Guided route. More room to escape."),
    "Runner": ("Pick up the pace", "A bigger maze and a closer chase."),
    "Master": ("Test your instincts", "Less time. A faster hunter."),
    "Adaptive": ("A pace that fits you", "Difficulty follows your recent rounds."),
    "Custom": ("Your own challenge", "Use your chosen grid and time limit."),
}


class IsometricGame:
    """Manage isometric screen transitions around the shared session so choosing a level
    does not create a timed round before Start.

    Own menu selection and round lifecycle; no round exists before Start.
    """

    def __init__(self, config, player_tag, logger, skill_hint=None, initial_tier="Recruit"):
        """Prepare the selected level, shared session, and input state without starting an
        engine so the player can choose a challenge before the clock begins.
        """
        if initial_tier not in TIERS:
            raise ValueError(f"Unknown difficulty: {initial_tier}")
        self.session = RoundSession(config, player_tag, logger, skill_hint)
        self.selected_tier = initial_tier
        self.state = "menu"
        self.held = HeldMovement()
        self.renderer = IsometricRenderer()
        self.preview_renderer = IsometricRenderer()
        self.previews = {}
        self.radar_path = []
        self.radar_until = 0.0
        self.notice = ""
        self.notice_until = 0.0

    @property
    def engine(self):
        """Expose the session-owned engine so input and rendering access one current round
        instead of keeping a second engine reference.
        """
        return self.session.engine

    def start(self):
        """Start the selected level with first-move activation and reset presentation state
        so replay never inherits held keys or animations from the previous round.
        """
        self.session.start(self.selected_tier, start_on_move=True)
        self.renderer = IsometricRenderer()
        self.held.clear()
        self.radar_path = []
        self.notice = ""
        self.state = "playing"

    def select(self, offset):
        """Cycle through difficulty choices with wraparound so keyboard users can select
        every level without a mouse.
        """
        self.selected_tier = TIERS[(TIERS.index(self.selected_tier) + offset) % len(TIERS)]

    def preview(self):
        """Cache a representative maze and its settings so the menu can show level size
        without creating or timing an active round.
        """
        config = configure_round(self.session.base_config, self.selected_tier, self.session.skill)
        key = (config.width, config.height, config.seed)
        if key not in self.previews:
            self.previews[key] = generate_maze(config.width, config.height, seed=12 if config.seed is None else config.seed)
        return self.previews[key], config

    def pause(self):
        """Keep UI state synchronized with engine pause state and clear held movement so
        opening a pause screen actually freezes play.
        """
        if self.state == "playing":
            self.engine.pause()
            if self.engine.is_paused:
                self.state = "paused"
        elif self.state == "paused":
            self.engine.resume()
            self.state = "playing"
        self.held.clear()

    def ability(self, name):
        """Route ability input to the shared engine and retain radar display information so
        UI-specific effects stay separate from game rules.
        """
        if self.state != "playing":
            return
        if name == "radar":
            guide = radar_guide(self.engine)
            if guide:
                self.radar_path = guide[0]
                self.radar_until = self.engine.elapsed_s + guide[1]
        else:
            self.engine.use_ability(name)

    def move(self, direction):
        """Request engine movement and show feedback for blocked attempts so the player
        understands why a direction did not work.
        """
        if self.state == "playing" and not self.engine.move_player(direction):
            self.notice = "Wall ahead. Try another direction."
            self.notice_until = time.monotonic() + 1.3

    def update(self, now):
        """Process held movement, engine time, and snapshots together so each active frame
        can finish the round and switch to its results screen.
        """
        if self.state != "playing":
            return
        direction = self.held.update(now)
        if direction:
            self.move(direction)
        self.engine.update()
        self.session.capture_snapshot()
        self.finish()

    def finish(self):
        """Complete logging before leaving a terminal round so a quit event arriving with
        the final move does not discard its result.

        Settle a terminal move even when a quit event arrives in the same batch.
        """
        if self.engine is None:
            return
        if self.engine.outcome != Outcome.PLAYING:
            self.session.finish()
            self.state = "results"
            self.held.clear()

    def menu(self):
        # Unfinished rounds are abandoned, just as closing the classic view is.
        """Return to level selection and clear held input so choosing a new challenge does
        not carry movement from the previous screen.
        """
        self.state = "menu"
        self.held.clear()


class Interface:
    """Group light-themed screen drawing and hit areas so visual controls and event
    handling stay aligned.
    """
    def __init__(self, surface, game):
        """Prepare reusable fonts and click geometry so drawing and input handling refer to
        the same UI layout.
        """
        self.surface = surface
        self.game = game
        self.fonts = {size: pygame.font.SysFont("Arial,DejaVu Sans", size, bold=size >= 20)
                      for size in (12, 14, 16, 18, 20, 24, 32, 40, 48, 56)}
        self.buttons = {}
        self.projection = None
        self.board_rect = pygame.Rect(0, 0, 0, 0)

    def text(self, text, x, y, size=16, color=INK, *, center=False):
        """Centralize font selection and alignment so every screen uses consistent text
        placement.
        """
        rendered = self.fonts[size].render(str(text), True, color)
        rect = rendered.get_rect(midtop=(x, y)) if center else rendered.get_rect(topleft=(x, y))
        self.surface.blit(rendered, rect)

    def card(self, rect, color=(255, 255, 255), border=LINE):
        """Draw the shared panel background, border, and shadow so the menu, sidebar, and
        modals keep one visual theme.
        """
        pygame.draw.rect(self.surface, (235, 237, 242), rect.move(0, 3), border_radius=16)
        pygame.draw.rect(self.surface, color, rect, border_radius=16)
        pygame.draw.rect(self.surface, border, rect, 1, border_radius=16)

    def button(self, action, rect, title, *, primary=False, enabled=True, subtitle=None):
        """Draw an action button and register enabled hit areas so disabled abilities
        cannot be activated by mouse clicks.
        """
        hover = rect.collidepoint(pygame.mouse.get_pos()) and enabled
        color = BLUE if primary else ((241, 242, 250) if hover else (255, 255, 255))
        if primary and hover:
            color = (78, 79, 208)
        pygame.draw.rect(self.surface, color, rect, border_radius=10)
        if not primary:
            pygame.draw.rect(self.surface, LINE, rect, 1, border_radius=10)
        fg = (255, 255, 255) if primary else INK
        if not enabled:
            fg = MUTED
        if subtitle:
            self.text(title, rect.x + 16, rect.y + 8, 16, fg)
            self.text(subtitle, rect.x + 16, rect.y + 29, 12, MUTED)
        else:
            self.text(title, rect.centerx, rect.y + (rect.height - 22) / 2, 16, fg, center=True)
        if enabled:
            self.buttons[action] = rect

    def header(self):
        """Provide common branding and the current screen action so players can
        consistently find Close or Pause.
        """
        w = self.surface.get_width()
        pygame.draw.rect(self.surface, BLUE, (48, 34, 24, 24), border_radius=7)
        self.text("m", 60, 31, 20, (255, 255, 255), center=True)
        self.text("maze mind", 84, 29, 24)
        self.text("ISOMETRIC", 238, 38, 12, MUTED)
        self.text(self.game.session.player_tag[:22], w - 245, 36, 14, MUTED)
        if self.game.state == "menu":
            self.button("quit", pygame.Rect(w - 128, 29, 80, 36), "Close")
        else:
            self.button("pause", pygame.Rect(w - 154, 29, 106, 36), "Resume" if self.game.state == "paused" else "Pause")
        pygame.draw.line(self.surface, LINE, (48, 88), (w - 48, 88))

    def draw_menu(self):
        """Show difficulty choices, a preview, and Start before gameplay so the player
        knows which challenge will begin.
        """
        g, surface = self.game, self.surface
        w, h = surface.get_size()
        self.text("Choose your pace.", 48, 119, 40)
        self.text("Find the exit. Stay ahead of the hunter.", 50, 173, 16, MUTED)
        card_height = min(78, (h - 354) // 5 - 10)
        for i, tier in enumerate(TIERS):
            rect = pygame.Rect(48, 223 + i * (card_height + 10), 350, card_height)
            selected = tier == g.selected_tier
            self.card(rect, (239, 239, 253) if selected else (255, 255, 255), BLUE if selected else LINE)
            self.text(tier, rect.x + 20, rect.y + 11, 20, BLUE if selected else INK)
            self.text(LEVEL_COPY[tier][0], rect.x + 20, rect.y + 40, 14, MUTED)
            if selected:
                pygame.draw.circle(surface, BLUE, (rect.right - 26, rect.centery), 8)
                pygame.draw.circle(surface, (255, 255, 255), (rect.right - 26, rect.centery), 3)
            self.buttons[f"level:{tier}"] = rect
        maze, config = g.preview()
        preview = pygame.Rect(426, 223, w - 474, h - 335)
        self.card(preview)
        self.text("THE PLAYFIELD", preview.x + 24, preview.y + 20, 12, MUTED)
        g.preview_renderer.draw(surface, pygame.Rect(preview.x + 16, preview.y + 48,
                                                    preview.width - 32, preview.height - 108), maze,
                                player=maze.start, now=time.monotonic())
        self.text(f"{config.width} x {config.height} grid    /    {config.round_time_limit_s:g} seconds",
                  preview.centerx, preview.bottom - 50, 18, center=True)
        self.text(LEVEL_COPY[g.selected_tier][1], preview.centerx, preview.bottom - 27, 14, MUTED, center=True)
        self.button("start", pygame.Rect(48, h - 94, 350, 52), "Start playing   /   Enter", primary=True)
        self.text("The hunter waits for your first move.", 426, h - 85, 16)
        self.text("Choose a level with your mouse or arrow keys.", 426, h - 59, 14, MUTED)

    def draw_play(self):
        """Present the maze, escape progress, hunter state, and abilities so navigation
        decisions have clear visual feedback.
        """
        g, e = self.game, self.game.engine
        w, h = self.surface.get_size()
        self.text(g.selected_tier, 48, 116, 24)
        self.text(f"{e.config.width} x {e.config.height}    /    Reach the green exit", 48, 151, 14, MUTED)
        self.board_rect = pygame.Rect(48, 190, w - 404, h - 278)
        self.card(self.board_rect)
        guide = g.radar_path if e.elapsed_s < g.radar_until else e.route_guide_path()
        self.projection = g.renderer.draw(self.surface, self.board_rect.inflate(-16, -16), e.maze,
                                          player=e.player_pos, hunter=e.hunter.position, guide=guide,
                                          now=time.monotonic(), stunned=e.hunter.is_stunned,
                                          font=self.fonts[12])
        panel = pygame.Rect(w - 328, 116, 280, h - 204)
        self.card(panel)
        x, y = panel.x + 22, panel.y + 22
        self.text("TIME REMAINING", x, y, 12, MUTED)
        remaining = max(0, e.config.round_time_limit_s - e.elapsed_s)
        minutes, seconds = divmod(int(remaining + .999), 60)
        self.text(f"{minutes:02}:{seconds:02}", x, y + 20, 48, RED if remaining < 15 else INK)
        self.text("Escape progress", x, y + 98, 16)
        self.text(f"{int(e.route_progress * 100)}%", panel.right - 62, y + 99, 14, BLUE)
        bar = pygame.Rect(x, y + 127, panel.width - 44, 6)
        pygame.draw.rect(self.surface, LINE, bar, border_radius=3)
        if e.route_progress:
            pygame.draw.rect(self.surface, BLUE, (bar.x, bar.y, int(bar.width * e.route_progress), 6), border_radius=3)
        steps = e.maze.shortest_path_len(e.player_pos, e.maze.exit)
        self.text(f"{steps} steps to the exit", x, y + 148, 14, MUTED)
        status = "Waiting for you" if not e.chase_started else ("Frozen by smoke" if e.hunter.is_stunned else "Chasing you")
        self.text("HUNTER", x, y + 195, 12, MUTED)
        pygame.draw.circle(self.surface, RED if e.chase_started else MUTED, (x + 5, y + 231), 4)
        self.text(status, x + 18, y + 219, 16)
        self.button("dash", pygame.Rect(x, y + 261, panel.width - 44, 50), "Shift   /   Dash",
                    enabled=e.player.can_dash, subtitle="Move up to two cells" if e.player.can_dash else f"Ready in {e.player.dash_cooldown_remaining_s:.1f}s")
        self.button("smoke", pygame.Rect(x, y + 321, panel.width - 44, 50), f"F   /   Smoke     {e.player.smoke_charges}",
                    enabled=e.player.can_smoke, subtitle="Freeze the hunter for 3 seconds")
        self.button("radar", pygame.Rect(x, y + 381, panel.width - 44, 50), f"R   /   Radar     {e.player.radar_charges}",
                    enabled=e.player.can_radar, subtitle="Reveal your next eight steps")
        self.text("MOVE   /   WASD or arrows", x, y + 455, 12, MUTED)
        # The labels match the projected cardinal directions on the playfield.
        for direction, label, ox, oy in [("W", "A / Left", 0, 481), ("N", "W / Up", 124, 481),
                                         ("S", "S / Down", 0, 519), ("E", "D / Right", 124, 519)]:
            self.button(f"move:{direction}", pygame.Rect(x + ox, y + oy, 112, 32), label)
        legend_y = h - 60
        for offset, color, label in [(0, BLUE, "You"), (92, RED, "Hunter"), (212, GREEN, "Exit")]:
            pygame.draw.circle(self.surface, color, (54 + offset, legend_y + 9), 5)
            self.text(label, 68 + offset, legend_y, 14, MUTED)
        self.text("Click an adjacent tile to move.    P / Esc to pause.", 48, h - 33, 12, MUTED)
        if not e.chase_started:
            self.text("Ready when you are. Move to begin the chase.", self.board_rect.centerx,
                      self.board_rect.y + 20, 16, center=True)
        elif g.notice and time.monotonic() < g.notice_until:
            self.text(g.notice, self.board_rect.centerx, self.board_rect.y + 20, 14, MUTED, center=True)
        elif e.recent_events:
            self.text(e.recent_events[-1]["text"], self.board_rect.centerx, self.board_rect.y + 20, 14, BLUE, center=True)

    def draw_overlay(self):
        """Present pause or completed-round actions over the board so the player can
        resume, replay, or choose another level.
        """
        g = self.game
        w, h = self.surface.get_size()
        veil = pygame.Surface((w, h), pygame.SRCALPHA)
        veil.fill((245, 246, 248, 225))
        self.surface.blit(veil, (0, 0))
        self.buttons.clear()
        rect = pygame.Rect(w // 2 - 240, h // 2 - 174, 480, 348)
        self.card(rect)
        paused = g.state == "paused"
        won = g.engine.outcome == Outcome.ESCAPED
        title = "Take a breath." if paused else ("You made it." if won else "Another way out awaits.")
        self.text("PAUSED" if paused else ("ESCAPED" if won else g.engine.outcome.value.upper()),
                  rect.centerx, rect.y + 28, 12, BLUE if paused else (GREEN if won else RED), center=True)
        self.text(title, rect.centerx, rect.y + 65, 32, center=True)
        if paused:
            detail = "The clock and hunter are on hold."
        elif won:
            detail = f"{g.engine.elapsed_s:.1f} seconds  /  {len(g.engine.path_taken) - 1} steps"
        else:
            detail = "Try Recruit for a guided route and a slower hunter."
        self.text(detail, rect.centerx, rect.y + 119, 14, MUTED, center=True)
        self.button("pause" if paused else "start", pygame.Rect(rect.x + 48, rect.y + 175, 384, 52),
                    "Resume   /   Enter" if paused else "Play again   /   Enter", primary=True)
        self.button("menu", pygame.Rect(rect.x + 48, rect.y + 240, 384, 44), "Choose a level")
        if not paused:
            self.text("Round saved to your records.", rect.centerx, rect.bottom - 36, 12, MUTED, center=True)

    def draw(self):
        """Rebuild the current screen and its active hit areas together so clicks cannot
        trigger controls belonging to a previous screen.
        """
        self.buttons.clear()
        self.surface.fill(BG)
        self.header()
        if self.game.state == "menu":
            self.draw_menu()
        else:
            self.draw_play()
            if self.game.state in ("paused", "results"):
                self.draw_overlay()

    def action(self, action):
        """Translate named UI actions into game operations so button drawing remains
        separate from round state changes.
        """
        g = self.game
        if action == "quit":
            return False
        if action.startswith("level:"):
            g.selected_tier = action.split(":")[1]
        elif action.startswith("move:"):
            g.move(action.split(":")[1])
        elif action == "start":
            g.start()
        elif action == "pause":
            g.pause()
        elif action == "menu":
            g.menu()
        elif action in ("dash", "smoke", "radar"):
            g.ability(action)
        return True

    def event(self, event, now):
        """Interpret keyboard, mouse, and focus events according to the current screen so
        menu navigation cannot accidentally move an active player.
        """
        g = self.game
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.WINDOWFOCUSLOST:
            if g.state == "playing":
                g.pause()
            g.held.clear()
        elif event.type == pygame.KEYUP:
            g.held.release(event.key)
        elif event.type == pygame.KEYDOWN:
            if g.state == "menu":
                if event.key == pygame.K_RETURN:
                    g.start()
                elif event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                    g.select(1)
                elif event.key in (pygame.K_LEFT, pygame.K_UP):
                    g.select(-1)
                elif event.key == pygame.K_ESCAPE:
                    return False
            elif g.state == "results":
                if event.key == pygame.K_RETURN:
                    g.start()
                elif event.key == pygame.K_ESCAPE:
                    g.menu()
            elif event.key in (pygame.K_ESCAPE, pygame.K_p) or (g.state == "paused" and event.key == pygame.K_RETURN):
                g.pause()
            elif g.state == "playing":
                if event.key in KEY_TO_DIR:
                    direction = g.held.press(event.key, now)
                    if direction:
                        g.move(direction)
                elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    g.ability("dash")
                elif event.key == pygame.K_f:
                    g.ability("smoke")
                elif event.key == pygame.K_r:
                    g.ability("radar")
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Button geometry belongs to the current screen, including transitions
            # caused by an earlier event in this same batch.
            self.draw()
            for action, rect in self.buttons.items():
                if rect.collidepoint(event.pos):
                    return self.action(action)
            if g.state == "playing" and self.projection and self.board_rect.collidepoint(event.pos):
                cell = self.projection.cell_at(event.pos, g.engine.maze.width, g.engine.maze.height)
                if cell:
                    move_toward_cell(g.engine, cell)
        return True


def run_round(config: RoundConfig, player_tag="player1", skill_hint=None, logger=None,
              initial_tier="Recruit", headless_for_testing=False):
    """Own the window, event loop, resize handling, and cleanup so the CLI can launch the
    isometric experience with the same configuration and logger as classic play.

    Launch the optional view; the level menu always appears before gameplay.
    """
    pygame.init()
    try:
        surface = pygame.display.set_mode((1280, 840), pygame.RESIZABLE)
        pygame.display.set_caption("Maze Mind / Isometric")
        game = IsometricGame(config, player_tag, logger or RoundLogger(), skill_hint, initial_tier)
        interface = Interface(surface, game)
        clock = pygame.time.Clock()
        running, frames = True, 0
        interface.draw()
        while running:
            now = time.monotonic()
            for event in pygame.event.get():
                if event.type == pygame.VIDEORESIZE:
                    surface = pygame.display.set_mode((max(1080, event.w), max(820, event.h)), pygame.RESIZABLE)
                    interface.surface = surface
                elif not interface.event(event, now):
                    running = False
                    break
            if not running:
                game.finish()
                break
            game.update(now)
            interface.draw()
            pygame.display.flip()
            clock.tick(30)
            frames += 1
            if headless_for_testing and frames >= 3:
                break
        return game.engine.outcome if game.engine else Outcome.PLAYING
    finally:
        pygame.quit()
