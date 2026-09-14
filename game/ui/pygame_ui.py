"""
Master Class Pygame UI, Tactical Cockpit & Interactive Control System.

===============================================================================
THEORETICAL FOUNDATION & DESIGN PATTERNS:
===============================================================================
1. TACTICAL COCKPIT ARCHITECTURE:
   - Split Viewport Architecture (1040x760px):
     * Left Stage: Spacious, high-resolution Maze canvas (dynamically scaled
       cells from 40px to 52px) with glowing exit beacon, visited-path
       breadcrumbs, and radar path guidance.
     * Right Command Console: Ergonomic tactical telemetry HUD, pause toggle,
       ability cards with visual cooldowns, and interactive navigation pad.
   - Screen State Machine: Dispatches rendering and events across:
     * STATE_MENU: Level/Difficulty selection (Recruit, Runner, Master, Adaptive).
     * STATE_PLAYING: Live round loop with real-time ML inference.
     * STATE_PAUSED: Suspends game clock and entities with a frosted modal.
     * STATE_HISTORY: Career record dossier, lifetime stats, and match log table.
     * STATE_GAMEOVER: Performance breakdown, efficiency metric, and replay loop.

2. FAIR ASYMMETRIC GAMEPLAY & NAVIGATION ASSISTANCE:
   - True Hunter Patrol: The pursuer patrols map waypoints and only pursues
     when within sensory detection radius (8 cells).
   - Visited Breadcrumbs: Soft glowing markers along explored paths prevent
     disorientation and looping.
   - Sonar Radar Golden Trail: Activating Radar illuminates the next 8 steps
     of the optimal route to the exit for 4 seconds.
   - High-Fidelity Icons:
     * Window Icon: Custom synthesized taskbar & window icon.
     * Player: Tactical operative badge with live directional chevron.
     * Hunter: Sinister cyber-skull with glowing visor slits and alert auras.
     * Exit Portal: Dimensional radiant diamond beacon with pulsing energy ring.
===============================================================================
"""

from __future__ import annotations

import math
import os
import sys
import time

import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from game.engine import RoundEngine, RoundConfig, Outcome  # noqa: E402
from game.astar import HunterState  # noqa: E402
from data.logger import RoundLogger, RoundRecord  # noqa: E402
from data.preprocess import extract_round_features  # noqa: E402

# Window & Layout Geometry (Spacious Widescreen Cockpit)
SCREEN_WIDTH = 1040
SCREEN_HEIGHT = 760
MAZE_STAGE_WIDTH = 660
MAZE_STAGE_HEIGHT = 680
CONSOLE_X = 690
CONSOLE_WIDTH = 326
MARGIN = 20

# Visual Design Tokens (Modern Slate & Indigo Palette)
COLOR_BG = (13, 16, 23)
COLOR_STAGE_BG = (18, 22, 33)
COLOR_CARD = (22, 27, 39)
COLOR_CARD_BORDER = (45, 55, 78)
COLOR_CARD_ACTIVE = (99, 102, 241)
COLOR_HUD_BG = (18, 22, 33)

COLOR_FLOOR = (19, 24, 35)
COLOR_WALL = (68, 80, 112)
COLOR_WALL_BORDER = (88, 102, 142)

COLOR_PLAYER = (59, 130, 246)
COLOR_PLAYER_GLOW = (96, 165, 250)
COLOR_HUNTER = (239, 68, 68)
COLOR_EXIT = (34, 197, 94)
COLOR_BREADCRUMB = (96, 165, 250)
COLOR_RADAR_TRAIL = (250, 204, 21)

COLOR_TEXT_PRIMARY = (248, 250, 252)
COLOR_TEXT_MUTED = (148, 163, 184)
COLOR_TEXT_ACCENT = (96, 165, 250)
COLOR_TEXT_WARNING = (251, 191, 36)
COLOR_TEXT_SUCCESS = (74, 222, 128)

COLOR_METER_BG = (32, 38, 54)
COLOR_METER_FILL = (34, 197, 94)
COLOR_METER_FILL_MID = (251, 191, 36)
COLOR_METER_FILL_LOW = (239, 68, 68)

STATE_MENU = "menu"
STATE_PLAYING = "playing"
STATE_PAUSED = "paused"
STATE_HISTORY = "history"
STATE_GAMEOVER = "game_over"

KEY_TO_DIR = {
    pygame.K_w: "N", pygame.K_UP: "N",
    pygame.K_s: "S", pygame.K_DOWN: "S",
    pygame.K_a: "W", pygame.K_LEFT: "W",
    pygame.K_d: "E", pygame.K_RIGHT: "E",
}

# Pre-configured Difficulty Tiers
DIFFICULTY_PRESETS = {
    "Recruit": {
        "width": 10, "height": 10, "time_limit": 90.0,
        "hunter_replan": 30, "hunter_move": 40, "awakening": 4.0,
        "smokes": 2, "radars": 3, "dash_cd": 5.0,
        "desc": "10x10 Grid • 90s Timer • Gentle Hunter • 2 Smokes & 3 Radars",
    },
    "Runner": {
        "width": 12, "height": 12, "time_limit": 75.0,
        "hunter_replan": 22, "hunter_move": 28, "awakening": 3.0,
        "smokes": 1, "radars": 2, "dash_cd": 7.0,
        "desc": "12x12 Grid • 75s Timer • Balanced Pursuit • 1 Smoke & 2 Radars",
    },
    "Master": {
        "width": 14, "height": 14, "time_limit": 60.0,
        "hunter_replan": 18, "hunter_move": 20, "awakening": 2.0,
        "smokes": 1, "radars": 1, "dash_cd": 8.0,
        "desc": "14x14 Grid • 60s Timer • Fast Relentless Hunter • Tactical Duel",
    },
}


def _create_window_icon() -> pygame.Surface:
    """Synthesize a crisp 32x32 application taskbar icon."""
    surf = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.rect(surf, (22, 27, 39), (0, 0, 32, 32), border_radius=6)
    pygame.draw.rect(surf, (99, 102, 241), (0, 0, 32, 32), width=2, border_radius=6)
    # Maze corridors
    pygame.draw.line(surf, (59, 130, 246), (6, 8), (26, 8), 2)
    pygame.draw.line(surf, (59, 130, 246), (26, 8), (26, 20), 2)
    pygame.draw.line(surf, (59, 130, 246), (12, 14), (12, 24), 2)
    pygame.draw.line(surf, (59, 130, 246), (12, 24), (22, 24), 2)
    # Beacons
    pygame.draw.circle(surf, (34, 197, 94), (22, 16), 3)
    pygame.draw.circle(surf, (96, 165, 250), (8, 20), 3)
    return surf


class Button:
    """Clickable UI button supporting hover states, disabled styling, and badges."""

    def __init__(
        self,
        rect: tuple[int, int, int, int],
        text: str,
        font: pygame.font.Font,
        bg_color: tuple = (32, 39, 56),
        hover_color: tuple = (50, 60, 85),
        text_color: tuple = COLOR_TEXT_PRIMARY,
        border_color: tuple | None = (55, 68, 96),
        border_radius: int = 8,
        badge_text: str | None = None,
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_color = border_color
        self.border_radius = border_radius
        self.badge_text = badge_text
        self.is_hovered = False
        self.disabled = False

    def update(self, mouse_pos: tuple[int, int]) -> None:
        self.is_hovered = self.rect.collidepoint(mouse_pos) and not self.disabled

    def draw(self, screen: pygame.Surface) -> None:
        if self.disabled:
            color = (24, 28, 38)
            border_c = (38, 45, 60)
            txt_color = (90, 100, 120)
        else:
            color = self.hover_color if self.is_hovered else self.bg_color
            border_c = (99, 102, 241) if self.is_hovered else self.border_color
            txt_color = self.text_color

        pygame.draw.rect(screen, color, self.rect, border_radius=self.border_radius)
        if border_c:
            pygame.draw.rect(screen, border_c, self.rect, width=2, border_radius=self.border_radius)

        txt_surf = self.font.render(self.text, True, txt_color)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)

        if self.badge_text:
            badge_font = pygame.font.SysFont("segoeui", 11, bold=True)
            b_surf = badge_font.render(self.badge_text, True, COLOR_TEXT_WARNING)
            screen.blit(b_surf, (self.rect.right - b_surf.get_width() - 8, self.rect.top + 5))

    def is_clicked(self, event: pygame.event.Event) -> bool:
        if self.disabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False


def _draw_meter(
    screen: pygame.Surface,
    font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    h: int,
    ratio: float,
    label: str,
    status_text: str = "",
    low_is_bad: bool = False,
) -> None:
    """Render a progress meter with gradient fill and labels."""
    ratio = max(0.0, min(1.0, ratio))
    pygame.draw.rect(screen, COLOR_METER_BG, (x, y, w, h), border_radius=4)

    if low_is_bad:
        if ratio >= 0.5:
            fill_c = COLOR_METER_FILL
        elif ratio >= 0.25:
            fill_c = COLOR_METER_FILL_MID
        else:
            fill_c = COLOR_METER_FILL_LOW
    else:
        fill_c = COLOR_METER_FILL

    fill_w = int(w * ratio)
    if fill_w > 0:
        pygame.draw.rect(screen, fill_c, (x, y, fill_w, h), border_radius=4)

    lbl_surf = font.render(label, True, COLOR_TEXT_MUTED)
    screen.blit(lbl_surf, (x, y - 16))

    if status_text:
        st_surf = font.render(status_text, True, COLOR_TEXT_PRIMARY)
        screen.blit(st_surf, (x + w - st_surf.get_width(), y - 16))


def _draw_player_icon(screen: pygame.Surface, px: int, py: int, cell_size: int, direction: str) -> None:
    """Render a tactical operative icon with glowing shield and directional chevron."""
    r = max(6, cell_size // 3)
    p_glow = int(cell_size * 0.42 + 2 * math.sin(time.monotonic() * 6))
    s_pglow = pygame.Surface((p_glow * 2, p_glow * 2), pygame.SRCALPHA)
    pygame.draw.circle(s_pglow, (*COLOR_PLAYER_GLOW, 65), (p_glow, p_glow), p_glow)
    screen.blit(s_pglow, (px - p_glow, py - p_glow))

    # Operative base circle
    pygame.draw.circle(screen, COLOR_PLAYER, (px, py), r)
    pygame.draw.circle(screen, (147, 197, 253), (px, py), r, width=2)

    # Directional chevron
    arrow_len = max(3, r // 2)
    if direction == "N":
        pts = [(px, py - arrow_len), (px - arrow_len, py + arrow_len // 2), (px + arrow_len, py + arrow_len // 2)]
    elif direction == "S":
        pts = [(px, py + arrow_len), (px - arrow_len, py - arrow_len // 2), (px + arrow_len, py - arrow_len // 2)]
    elif direction == "E":
        pts = [(px + arrow_len, py), (px - arrow_len // 2, py - arrow_len), (px - arrow_len // 2, py + arrow_len)]
    else:  # W
        pts = [(px - arrow_len, py), (px + arrow_len // 2, py - arrow_len), (px + arrow_len // 2, py + arrow_len)]
    pygame.draw.polygon(screen, (255, 255, 255), pts)


def _draw_hunter_icon(screen: pygame.Surface, hx: int, hy: int, cell_size: int) -> None:
    """Render a sinister pursuer skull/visor icon with glowing slit eyes."""
    r = max(6, cell_size // 3)
    # Main skull plate
    pygame.draw.circle(screen, (160, 20, 20), (hx, hy), r)
    pygame.draw.circle(screen, (239, 68, 68), (hx, hy), r - 1)
    pygame.draw.circle(screen, (255, 120, 120), (hx, hy), r, width=2)

    # Glowing visor slit eyes
    eye_w = max(2, r // 3)
    eye_h = max(2, r // 4)
    pygame.draw.rect(screen, (255, 240, 240), (hx - eye_w - 2, hy - 2, eye_w, eye_h), border_radius=1)
    pygame.draw.rect(screen, (255, 240, 240), (hx + 2, hy - 2, eye_w, eye_h), border_radius=1)
    # Forehead crest line
    pygame.draw.line(screen, (90, 10, 10), (hx - r + 3, hy - 4), (hx + r - 3, hy - 4), 2)


def _draw_exit_portal(screen: pygame.Surface, ex: int, ey: int, cell_size: int) -> None:
    """Render a radiant dimensional exit portal with glowing rings."""
    r = max(6, cell_size // 3)
    exit_glow = int(cell_size * 0.45 + 3 * math.sin(time.monotonic() * 5))
    s_exit = pygame.Surface((exit_glow * 2, exit_glow * 2), pygame.SRCALPHA)
    pygame.draw.circle(s_exit, (*COLOR_EXIT, 75), (exit_glow, exit_glow), exit_glow)
    screen.blit(s_exit, (ex - exit_glow, ey - exit_glow))

    # Outer portal diamond
    diamond_r = r + 2
    pts = [(ex, ey - diamond_r), (ex + diamond_r, ey), (ex, ey + diamond_r), (ex - diamond_r, ey)]
    pygame.draw.polygon(screen, COLOR_EXIT, pts)
    pygame.draw.polygon(screen, (167, 243, 208), pts, width=2)

    # Inner portal beacon
    inner_pts = [(ex, ey - r // 2), (ex + r // 2, ey), (ex, ey + r // 2), (ex - r // 2, ey)]
    pygame.draw.polygon(screen, (255, 255, 255), inner_pts)


def _draw_maze(
    screen: pygame.Surface,
    maze,
    offset_x: int,
    offset_y: int,
    cell_size: int,
    visited_cells: set[tuple[int, int]],
    radar_guide_path: list[tuple[int, int]],
    radar_pulse_alpha: int = 0,
) -> None:
    """Render the high-resolution maze grid, passages, breadcrumbs, and radar guide trail."""
    w_px = maze.width * cell_size
    h_px = maze.height * cell_size

    # Background Floor
    pygame.draw.rect(screen, COLOR_FLOOR, (offset_x, offset_y, w_px, h_px), border_radius=6)

    # 1. Visited Path Breadcrumbs (Footsteps)
    if visited_cells:
        for vx, vy in visited_cells:
            cx = offset_x + vx * cell_size + cell_size // 2
            cy = offset_y + vy * cell_size + cell_size // 2
            pygame.draw.circle(screen, (70, 90, 130), (cx, cy), max(2, cell_size // 9))

    # 2. Sonar Radar Guidance Trail (Golden Beacons)
    if radar_guide_path and radar_pulse_alpha > 0:
        s_trail = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
        for idx, (gx, gy) in enumerate(radar_guide_path):
            cx = gx * cell_size + cell_size // 2
            cy = gy * cell_size + cell_size // 2
            r = max(4, cell_size // 4)
            alpha = max(30, min(230, radar_pulse_alpha - idx * 15))
            pygame.draw.circle(s_trail, (*COLOR_RADAR_TRAIL, alpha), (cx, cy), r)
            pygame.draw.circle(s_trail, (255, 255, 255, alpha), (cx, cy), max(2, r // 2))
        screen.blit(s_trail, (offset_x, offset_y))

    # 3. Walls
    wall_thick = 3 if cell_size >= 40 else 2
    for y in range(maze.height):
        for x in range(maze.width):
            cx = offset_x + x * cell_size
            cy = offset_y + y * cell_size
            cell = maze.walls[y][x]
            if cell["N"]:
                pygame.draw.line(screen, COLOR_WALL, (cx, cy), (cx + cell_size, cy), wall_thick)
            if cell["W"]:
                pygame.draw.line(screen, COLOR_WALL, (cx, cy), (cx, cy + cell_size), wall_thick)
            if cell["S"]:
                pygame.draw.line(screen, COLOR_WALL, (cx, cy + cell_size), (cx + cell_size, cy + cell_size), wall_thick)
            if cell["E"]:
                pygame.draw.line(screen, COLOR_WALL, (cx + cell_size, cy), (cx + cell_size, cy + cell_size), wall_thick)

    # Outer Border Frame
    pygame.draw.rect(screen, COLOR_WALL_BORDER, (offset_x, offset_y, w_px, h_px), width=3, border_radius=6)


def _cell_center(cell: tuple[int, int], offset_x: int, offset_y: int, cell_size: int) -> tuple[int, int]:
    x, y = cell
    return (
        offset_x + x * cell_size + cell_size // 2,
        offset_y + y * cell_size + cell_size // 2,
    )


def _adapt_difficulty(skill_label: str) -> dict:
    try:
        from ml.knn_skill import difficulty_for_skill
        return difficulty_for_skill(skill_label)
    except Exception:
        return {
            "hunter_replan_interval": 20,
            "hunter_move_interval": 28,
            "awakening_duration_s": 2.5,
        }


def run_round(
    config: RoundConfig,
    player_tag: str = "player",
    skill_hint: str | None = None,
    logger: RoundLogger | None = None,
    headless_for_testing: bool = False,
) -> Outcome:
    """Execute the full interactive Maze Mind UI loop with Tactical Cockpit, History, and Pause."""
    if headless_for_testing:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    pygame.font.init()

    font_xs = pygame.font.SysFont("segoeui", 11)
    font_sm = pygame.font.SysFont("segoeui", 13)
    font = pygame.font.SysFont("segoeui", 15)
    font_bold = pygame.font.SysFont("segoeui", 16, bold=True)
    title_font = pygame.font.SysFont("segoeui", 20, bold=True)
    banner_font = pygame.font.SysFont("segoeui", 28, bold=True)
    big_banner_font = pygame.font.SysFont("segoeui", 34, bold=True)

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Mind — Tactical AI Maze Runner")
    try:
        pygame.display.set_icon(_create_window_icon())
    except Exception:
        pass

    clock = pygame.time.Clock()

    if logger is None:
        logger = RoundLogger()

    selected_tier = "Recruit"
    current_skill = skill_hint or "Beginner"

    # ML Predictors
    try:
        from ml.logreg_winprob import predict_win_probability
        winprob_available = True
    except Exception:
        winprob_available = False

    try:
        from ml.knn_skill import predict_skill
        skill_classifier_available = True
    except Exception:
        skill_classifier_available = False

    # Apply Selected Tier Preset
    def configure_round_engine(tier_name: str) -> RoundEngine:
        if tier_name in DIFFICULTY_PRESETS:
            p = DIFFICULTY_PRESETS[tier_name]
            config.width = p["width"]
            config.height = p["height"]
            config.round_time_limit_s = p["time_limit"]
            config.hunter_replan_interval = p["hunter_replan"]
            config.hunter_move_interval = p["hunter_move"]
            config.awakening_duration_s = p["awakening"]
            config.smoke_charges = p["smokes"]
            config.radar_charges = p["radars"]
            config.dash_cooldown_s = p["dash_cd"]
        else:
            diff = _adapt_difficulty(current_skill)
            config.hunter_replan_interval = diff["hunter_replan_interval"]
            config.hunter_move_interval = diff["hunter_move_interval"]
            config.awakening_duration_s = diff.get("awakening_duration_s", 2.5)

        return RoundEngine(config)

    engine = configure_round_engine(selected_tier)

    # Calculate Responsive Cell Sizing
    def calculate_cell_layout():
        c_size = min(52, max(36, 580 // max(engine.maze.width, engine.maze.height)))
        m_w = engine.maze.width * c_size
        m_h = engine.maze.height * c_size
        off_x = 20 + (MAZE_STAGE_WIDTH - m_w) // 2
        off_y = 65 + (MAZE_STAGE_HEIGHT - m_h) // 2
        return c_size, off_x, off_y

    cell_size, offset_x, offset_y = calculate_cell_layout()

    # ------------------ UI BUTTONS SETUP ------------------

    # 1. Top HUD Buttons
    btn_pause = Button(
        (SCREEN_WIDTH - 125, 14, 105, 36),
        "⏸ Pause [P]", font_sm, bg_color=(35, 42, 60), hover_color=(55, 68, 98),
    )

    # 2. Tactical Ability Buttons (Right Console)
    btn_dash = Button(
        (CONSOLE_X, 290, CONSOLE_WIDTH, 48),
        "⚡ BURST DASH [Shift]", font_bold, bg_color=(28, 38, 62), hover_color=(45, 65, 105),
    )
    btn_smoke = Button(
        (CONSOLE_X, 345, CONSOLE_WIDTH, 48),
        "❄ SMOKE FREEZE [F]", font_bold, bg_color=(46, 32, 50), hover_color=(75, 48, 82),
    )
    btn_radar = Button(
        (CONSOLE_X, 400, CONSOLE_WIDTH, 48),
        "📡 SONAR RADAR [R]", font_bold, bg_color=(24, 46, 44), hover_color=(36, 75, 70),
    )

    # 3. Ergonomic D-Pad Movement Buttons (Right Console)
    pad_cx = CONSOLE_X + CONSOLE_WIDTH // 2
    pad_y = 505
    b_sz = 46
    gap = 6

    btn_up = Button(
        (pad_cx - b_sz // 2, pad_y, b_sz, b_sz),
        "▲", font_bold, bg_color=(32, 40, 60), hover_color=(50, 65, 95),
    )
    btn_down = Button(
        (pad_cx - b_sz // 2, pad_y + b_sz + gap, b_sz, b_sz),
        "▼", font_bold, bg_color=(32, 40, 60), hover_color=(50, 65, 95),
    )
    btn_left = Button(
        (pad_cx - b_sz // 2 - b_sz - gap, pad_y + b_sz + gap, b_sz, b_sz),
        "◄", font_bold, bg_color=(32, 40, 60), hover_color=(50, 65, 95),
    )
    btn_right = Button(
        (pad_cx + b_sz // 2 + gap, pad_y + b_sz + gap, b_sz, b_sz),
        "►", font_bold, bg_color=(32, 40, 60), hover_color=(50, 65, 95),
    )
    arrow_buttons = [btn_up, btn_down, btn_left, btn_right]
    arrow_actions = {btn_up: "N", btn_down: "S", btn_left: "W", btn_right: "E"}

    # Quick Action Buttons below D-Pad
    btn_restart_round = Button(
        (CONSOLE_X, 690, 155, 38),
        "🔄 Restart Level", font_sm, bg_color=(35, 42, 60), hover_color=(50, 60, 85),
    )
    btn_menu_return = Button(
        (CONSOLE_X + 170, 690, 155, 38),
        "🏠 Main Menu", font_sm, bg_color=(28, 34, 48), hover_color=(45, 52, 70),
    )

    # 4. Start Menu Tier Buttons & Action Buttons
    menu_tier_buttons = {}
    tier_names = list(DIFFICULTY_PRESETS.keys())
    for i, tname in enumerate(tier_names):
        bx = SCREEN_WIDTH // 2 - 260 + i * 176
        menu_tier_buttons[tname] = Button(
            (bx, 360, 168, 50),
            tname, font_bold,
            bg_color=(30, 36, 52) if tname != selected_tier else (37, 99, 235),
            hover_color=(50, 65, 95),
        )

    btn_start_game = Button(
        (SCREEN_WIDTH // 2 - 130, 485, 260, 52),
        "▶ Start Game [Enter]", font_bold, bg_color=(37, 99, 235), hover_color=(59, 130, 246),
    )
    btn_history_menu = Button(
        (SCREEN_WIDTH // 2 - 130, 545, 260, 42),
        "📜 Career Records & Logs", font, bg_color=(35, 44, 65), hover_color=(50, 65, 95),
    )
    btn_quit_menu = Button(
        (SCREEN_WIDTH // 2 - 130, 595, 260, 40),
        "Quit to Desktop", font_sm, bg_color=(28, 34, 48), hover_color=(45, 52, 70),
    )

    # 5. Pause Modal Buttons
    pause_modal_w, pause_modal_h = 420, 370
    pause_modal_x = (SCREEN_WIDTH - pause_modal_w) // 2
    pause_modal_y = (SCREEN_HEIGHT - pause_modal_h) // 2

    btn_resume_pause = Button(
        (pause_modal_x + 35, pause_modal_y + 140, 350, 44),
        "▶ Resume Game [P]", font_bold, bg_color=(37, 99, 235), hover_color=(59, 130, 246),
    )
    btn_restart_pause = Button(
        (pause_modal_x + 35, pause_modal_y + 195, 350, 40),
        "🔄 Restart Level [R]", font, bg_color=(38, 46, 66), hover_color=(55, 68, 96),
    )
    btn_history_pause = Button(
        (pause_modal_x + 35, pause_modal_y + 245, 350, 40),
        "📜 Career Records", font, bg_color=(32, 40, 58), hover_color=(48, 62, 90),
    )
    btn_menu_pause = Button(
        (pause_modal_x + 35, pause_modal_y + 295, 170, 40),
        "🏠 Main Menu [M]", font_sm, bg_color=(28, 34, 48), hover_color=(45, 55, 75),
    )
    btn_quit_pause = Button(
        (pause_modal_x + 215, pause_modal_y + 295, 170, 40),
        "✖ Quit [Esc]", font_sm, bg_color=(28, 32, 44), hover_color=(45, 50, 68),
    )

    # 6. Game Over Modal Buttons
    gameover_modal_w, gameover_modal_h = 500, 330
    go_modal_x = (SCREEN_WIDTH - gameover_modal_w) // 2
    go_modal_y = (SCREEN_HEIGHT - gameover_modal_h) // 2

    btn_replay = Button(
        (go_modal_x + 25, go_modal_y + gameover_modal_h - 60, 140, 42),
        "Play Again", font_bold, bg_color=(37, 99, 235), hover_color=(59, 130, 246),
    )
    btn_history_go = Button(
        (go_modal_x + 180, go_modal_y + gameover_modal_h - 60, 140, 42),
        "📜 Records", font, bg_color=(38, 46, 66), hover_color=(55, 68, 96),
    )
    btn_menu_go = Button(
        (go_modal_x + 335, go_modal_y + gameover_modal_h - 60, 140, 42),
        "Main Menu", font, bg_color=(28, 34, 48), hover_color=(45, 52, 70),
    )

    # 7. History Viewport Button
    btn_back_history = Button(
        (MARGIN + 6, 14, 110, 36),
        "◀ Back [Esc]", font_bold, bg_color=(35, 42, 60), hover_color=(55, 68, 98),
    )

    # Active Round States
    current_state = STATE_PLAYING if headless_for_testing else STATE_MENU
    previous_state = STATE_MENU
    round_recorded = False
    radar_guide_path: list[tuple[int, int]] = []
    radar_guide_timer = 0.0
    last_tick_time = time.monotonic()
    player_summary = logger.get_player_summary(player_tag)

    def start_new_round():
        nonlocal engine, round_recorded, cell_size, offset_x, offset_y, radar_guide_path, radar_guide_timer, player_summary
        engine = configure_round_engine(selected_tier)
        cell_size, offset_x, offset_y = calculate_cell_layout()
        round_recorded = False
        radar_guide_path = []
        radar_guide_timer = 0.0
        player_summary = logger.get_player_summary(player_tag)

    running = True
    frames_to_render = 3 if headless_for_testing else None
    frame_count = 0

    while running:
        now = time.monotonic()
        dt = max(0.001, min(0.1, now - last_tick_time))
        last_tick_time = now

        mouse_pos = pygame.mouse.get_pos()

        # Update Radar timer
        if radar_guide_timer > 0:
            radar_guide_timer = max(0.0, radar_guide_timer - dt)
            if radar_guide_timer == 0:
                radar_guide_path = []

        # ------------------ EVENT HANDLING ------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            # STATE: START MENU
            if current_state == STATE_MENU:
                for tname, btn in menu_tier_buttons.items():
                    btn.update(mouse_pos)
                    if btn.is_clicked(event):
                        selected_tier = tname
                        for n, b in menu_tier_buttons.items():
                            b.bg_color = (37, 99, 235) if n == selected_tier else (30, 36, 52)

                btn_start_game.update(mouse_pos)
                btn_history_menu.update(mouse_pos)
                btn_quit_menu.update(mouse_pos)

                if btn_start_game.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE)):
                    start_new_round()
                    current_state = STATE_PLAYING
                elif btn_history_menu.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_h):
                    player_summary = logger.get_player_summary(player_tag)
                    previous_state = STATE_MENU
                    current_state = STATE_HISTORY
                elif btn_quit_menu.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # STATE: PLAYING
            elif current_state == STATE_PLAYING:
                btn_pause.update(mouse_pos)
                btn_dash.update(mouse_pos)
                btn_smoke.update(mouse_pos)
                btn_radar.update(mouse_pos)
                btn_restart_round.update(mouse_pos)
                btn_menu_return.update(mouse_pos)

                for btn in arrow_buttons:
                    btn.update(mouse_pos)
                    if btn.is_clicked(event) and engine.outcome == Outcome.PLAYING:
                        engine.move_player(arrow_actions[btn])

                # Pause toggle
                if btn_pause.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_p):
                    engine.pause()
                    current_state = STATE_PAUSED

                # Tactical Ability Click Triggers
                elif btn_dash.is_clicked(event):
                    engine.use_ability("dash")
                elif btn_smoke.is_clicked(event):
                    engine.use_ability("smoke")
                elif btn_radar.is_clicked(event):
                    info = engine.use_ability("radar")
                    if isinstance(info, dict):
                        radar_guide_path = info.get("guide_path", [])
                        radar_guide_timer = 4.0

                elif btn_restart_round.is_clicked(event):
                    start_new_round()
                elif btn_menu_return.is_clicked(event):
                    current_state = STATE_MENU

                # Mouse Click-To-Move (Direct Grid Navigation)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    gx = (mx - offset_x) // cell_size
                    gy = (my - offset_y) // cell_size
                    if 0 <= gx < engine.maze.width and 0 <= gy < engine.maze.height:
                        px, py = engine.player.position
                        dx, dy = gx - px, gy - py
                        for d, (ddx, ddy) in engine.maze.DIRS.items():
                            if (ddx, ddy) == (dx, dy):
                                engine.move_player(d)
                                break

                # Keyboard Hotkeys
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        engine.pause()
                        current_state = STATE_PAUSED
                    elif event.key in KEY_TO_DIR and engine.outcome == Outcome.PLAYING:
                        engine.move_player(KEY_TO_DIR[event.key])
                    elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                        engine.use_ability("dash")
                    elif event.key == pygame.K_f:
                        engine.use_ability("smoke")
                    elif event.key == pygame.K_r:
                        info = engine.use_ability("radar")
                        if isinstance(info, dict):
                            radar_guide_path = info.get("guide_path", [])
                            radar_guide_timer = 4.0

            # STATE: PAUSED
            elif current_state == STATE_PAUSED:
                btn_resume_pause.update(mouse_pos)
                btn_restart_pause.update(mouse_pos)
                btn_history_pause.update(mouse_pos)
                btn_menu_pause.update(mouse_pos)
                btn_quit_pause.update(mouse_pos)

                if (
                    btn_resume_pause.is_clicked(event)
                    or (event.type == pygame.KEYDOWN and event.key in (pygame.K_p, pygame.K_SPACE, pygame.K_RETURN))
                ):
                    engine.resume()
                    current_state = STATE_PLAYING
                elif btn_restart_pause.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_r):
                    engine.resume()
                    start_new_round()
                    current_state = STATE_PLAYING
                elif btn_history_pause.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_h):
                    player_summary = logger.get_player_summary(player_tag)
                    previous_state = STATE_PAUSED
                    current_state = STATE_HISTORY
                elif btn_menu_pause.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_m):
                    engine.resume()
                    current_state = STATE_MENU
                elif btn_quit_pause.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # STATE: HISTORY / DOSSIER
            elif current_state == STATE_HISTORY:
                btn_back_history.update(mouse_pos)
                if btn_back_history.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE)):
                    current_state = previous_state

            # STATE: GAME OVER
            elif current_state == STATE_GAMEOVER:
                btn_replay.update(mouse_pos)
                btn_history_go.update(mouse_pos)
                btn_menu_go.update(mouse_pos)

                if btn_replay.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE)):
                    start_new_round()
                    current_state = STATE_PLAYING
                elif btn_history_go.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_h):
                    player_summary = logger.get_player_summary(player_tag)
                    previous_state = STATE_GAMEOVER
                    current_state = STATE_HISTORY
                elif btn_menu_go.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_m):
                    current_state = STATE_MENU

        # Gameplay update
        if current_state == STATE_PLAYING and engine.outcome == Outcome.PLAYING:
            engine.update()
            if engine.outcome != Outcome.PLAYING:
                current_state = STATE_GAMEOVER

        # Automatic logging & difficulty adaptation on round conclusion
        if engine.outcome != Outcome.PLAYING and not round_recorded:
            round_recorded = True
            if logger is not None:
                optimal = engine.maze.shortest_path_len(engine.maze.start, engine.maze.exit)
                record = RoundRecord(
                    session_id=f"human-{player_tag}",
                    player_tag=player_tag,
                    is_bot=False,
                    maze_width=config.width,
                    maze_height=config.height,
                    optimal_path_len=optimal,
                    dead_end_count=engine.maze.dead_end_count(),
                    player_path_len=len(engine.path_taken) - 1,
                    time_taken_s=engine.elapsed_s,
                    backtracks=engine.backtracks,
                    decision_times_s=engine.decision_times_s,
                    outcome=engine.outcome.value,
                    hunter_replan_ms=int(1000 * config.hunter_replan_interval / config.fps),
                    hunter_move_ms=int(1000 * config.hunter_move_interval / config.fps),
                    skill_label=current_skill,
                )
                last_rid = logger.log_round(record)
                player_summary = logger.get_player_summary(player_tag)

                if skill_classifier_available:
                    try:
                        f_dict = extract_round_features({
                            "time_taken_s": record.time_taken_s,
                            "player_path_len": record.player_path_len,
                            "optimal_path_len": record.optimal_path_len,
                            "backtracks": record.backtracks,
                            "avg_decision_time_s": record.avg_decision_time_s,
                            "decision_times_s": record.decision_times_s,
                            "outcome": record.outcome,
                            "maze_width": record.maze_width,
                            "maze_height": record.maze_height,
                            "dead_end_count": record.dead_end_count,
                        })
                        predicted = predict_skill(f_dict)
                        if last_rid:
                            logger.update_skill_label(last_rid, predicted)
                        current_skill = predicted
                    except Exception:
                        pass

        # ------------------ RENDERING ------------------
        screen.fill(COLOR_BG)

        # RENDER: START MENU
        if current_state == STATE_MENU:
            title_surf = big_banner_font.render("MAZE MIND", True, COLOR_TEXT_PRIMARY)
            screen.blit(title_surf, title_surf.get_rect(center=(SCREEN_WIDTH // 2, 65)))

            sub_surf = font_bold.render("Tactical AI Maze Runner • Cognitive Learning Loop", True, COLOR_TEXT_ACCENT)
            screen.blit(sub_surf, sub_surf.get_rect(center=(SCREEN_WIDTH // 2, 110)))

            # Instructions Card
            card_w, card_h = 680, 180
            card_x = (SCREEN_WIDTH - card_w) // 2
            card_y = 145
            pygame.draw.rect(screen, COLOR_CARD, (card_x, card_y, card_w, card_h), border_radius=12)
            pygame.draw.rect(screen, COLOR_CARD_BORDER, (card_x, card_y, card_w, card_h), width=2, border_radius=12)

            lines = [
                (f"OPERATOR: {player_tag}   •   ACTIVE SKILL TIER: {current_skill}", COLOR_TEXT_ACCENT, font_bold),
                ("🎯 OBJECTIVE: Reach the Emerald Exit portal before the Hunter intercepts you!", COLOR_TEXT_PRIMARY, font),
                ("⚡ TACTICAL GEAR: [Shift] Dash 2-cells • [F] Freeze Hunter (3.5s) • [R] Radar Route Guide", COLOR_TEXT_WARNING, font),
                ("🧭 NAVIGATION: WASD, Arrow Keys, Click Adjacent Cells, or use On-Screen D-Pad", COLOR_TEXT_MUTED, font_sm),
                ("⏸ PAUSE ANYTIME: Press [P] or click the top-right Pause button to freeze time.", COLOR_TEXT_SUCCESS, font_sm),
            ]
            for i, (line, c, f) in enumerate(lines):
                txt = f.render(line, True, c)
                screen.blit(txt, (card_x + 20, card_y + 16 + i * 32))

            # Tier Selector Title
            tier_title = font_bold.render("SELECT MISSION SECTOR / DIFFICULTY:", True, COLOR_TEXT_PRIMARY)
            screen.blit(tier_title, tier_title.get_rect(center=(SCREEN_WIDTH // 2, 345)))

            for btn in menu_tier_buttons.values():
                btn.draw(screen)

            # Selected Tier Description
            tier_desc = DIFFICULTY_PRESETS.get(selected_tier, {}).get("desc", "")
            desc_surf = font_sm.render(tier_desc, True, COLOR_TEXT_WARNING)
            screen.blit(desc_surf, desc_surf.get_rect(center=(SCREEN_WIDTH // 2, 430)))

            btn_start_game.draw(screen)
            btn_history_menu.draw(screen)
            btn_quit_menu.draw(screen)

        # RENDER: CAREER HISTORY & PLAYER RECORD DOSSIER
        elif current_state == STATE_HISTORY:
            # Header
            pygame.draw.rect(screen, COLOR_HUD_BG, (0, 0, SCREEN_WIDTH, 56))
            pygame.draw.line(screen, COLOR_CARD_BORDER, (0, 56), (SCREEN_WIDTH, 56), 2)
            btn_back_history.draw(screen)

            dossier_title = title_font.render("📜 OPERATIVE DOSSIER & CAREER RECORDS", True, COLOR_TEXT_PRIMARY)
            screen.blit(dossier_title, (MARGIN + 130, 15))

            tag_info = f"Operator: {player_tag}  •  Skill: {current_skill}"
            tag_surf = font_bold.render(tag_info, True, COLOR_TEXT_ACCENT)
            screen.blit(tag_surf, (SCREEN_WIDTH - MARGIN - tag_surf.get_width() - 10, 18))

            # Top Row: 4 Metric Cards
            metrics = [
                ("TOTAL OPERATIONS", f"{player_summary['total_rounds']}", f"{player_summary['escapes']} Escaped • {player_summary['caught']} Caught", COLOR_TEXT_PRIMARY),
                ("ESCAPE WIN RATE", f"{player_summary['win_rate_pct']:.1f}%", "Successful Extractions", COLOR_TEXT_SUCCESS),
                ("RECORD ESCAPE TIME", f"{player_summary['best_time_s']:.1f}s" if player_summary['best_time_s'] else "—", "Fastest Operation Run", COLOR_TEXT_WARNING),
                ("PEAK EFFICIENCY", f"{player_summary['best_efficiency_pct']:.1f}%", "Path Optimality Score", COLOR_TEXT_ACCENT),
            ]
            c_w = (SCREEN_WIDTH - MARGIN * 2 - 36) // 4
            for i, (m_lbl, m_val, m_sub, m_col) in enumerate(metrics):
                cx = MARGIN + i * (c_w + 12)
                pygame.draw.rect(screen, COLOR_CARD, (cx, 68, c_w, 92), border_radius=10)
                pygame.draw.rect(screen, COLOR_CARD_BORDER, (cx, 68, c_w, 92), width=2, border_radius=10)

                l_surf = font_xs.render(m_lbl, True, COLOR_TEXT_MUTED)
                screen.blit(l_surf, (cx + 14, 76))
                v_surf = banner_font.render(m_val, True, m_col)
                screen.blit(v_surf, (cx + 14, 95))
                s_surf = font_xs.render(m_sub, True, COLOR_TEXT_MUTED)
                screen.blit(s_surf, (cx + 14, 134))

            # Bottom Table: Match History Logs
            tbl_x = MARGIN
            tbl_y = 172
            tbl_w = SCREEN_WIDTH - MARGIN * 2
            tbl_h = 565
            pygame.draw.rect(screen, COLOR_CARD, (tbl_x, tbl_y, tbl_w, tbl_h), border_radius=12)
            pygame.draw.rect(screen, COLOR_CARD_BORDER, (tbl_x, tbl_y, tbl_w, tbl_h), width=2, border_radius=12)

            t_header = font_bold.render("RECENT OPERATIONS LOG (LATEST 10 ROUNDS)", True, COLOR_TEXT_PRIMARY)
            screen.blit(t_header, (tbl_x + 18, tbl_y + 14))

            # Columns
            col_x = [tbl_x + 18, tbl_x + 110, tbl_x + 225, tbl_x + 370, tbl_x + 490, tbl_x + 640, tbl_x + 770, tbl_x + 890]
            headers = ["ROUND ID", "SECTOR", "OUTCOME", "TIME", "STEPS / OPT", "EFFICIENCY", "BACKTRACKS", "SKILL TIER"]
            for hx, h_txt in zip(col_x, headers):
                hs = font_xs.render(h_txt, True, COLOR_TEXT_MUTED)
                screen.blit(hs, (hx, tbl_y + 44))

            pygame.draw.line(screen, COLOR_CARD_BORDER, (tbl_x + 14, tbl_y + 64), (tbl_x + tbl_w - 14, tbl_y + 64), 1)

            recent_logs = player_summary.get("recent_rounds", [])
            if not recent_logs:
                empty_surf = font.render("No mission telemetry logged yet. Play your first round to establish records!", True, COLOR_TEXT_MUTED)
                screen.blit(empty_surf, empty_surf.get_rect(center=(tbl_x + tbl_w // 2, tbl_y + 200)))
            else:
                for idx, r in enumerate(recent_logs[:10]):
                    row_y = tbl_y + 74 + idx * 44
                    if idx % 2 == 1:
                        pygame.draw.rect(screen, (26, 32, 46), (tbl_x + 10, row_y - 4, tbl_w - 20, 36), border_radius=6)

                    # Round ID
                    r_id_txt = font_sm.render(f"#{r['round_id']}", True, COLOR_TEXT_MUTED)
                    screen.blit(r_id_txt, (col_x[0], row_y + 4))

                    # Sector
                    sec_txt = font_sm.render(f"{r['maze_width']}x{r['maze_height']}", True, COLOR_TEXT_PRIMARY)
                    screen.blit(sec_txt, (col_x[1], row_y + 4))

                    # Outcome Badge
                    out = r.get("outcome", "timeout")
                    if out == "escaped":
                        b_col, b_txt = COLOR_EXIT, "ESCAPED"
                    elif out == "caught":
                        b_col, b_txt = COLOR_HUNTER, "CAUGHT"
                    else:
                        b_col, b_txt = COLOR_TEXT_WARNING, "TIMEOUT"

                    out_badge = font_bold.render(f"[{b_txt}]", True, b_col)
                    screen.blit(out_badge, (col_x[2], row_y + 4))

                    # Time Taken
                    t_taken = font_sm.render(f"{r['time_taken_s']:.1f}s", True, COLOR_TEXT_PRIMARY)
                    screen.blit(t_taken, (col_x[3], row_y + 4))

                    # Steps / Optimal
                    steps_str = f"{r['player_path_len']} / {r['optimal_path_len']}"
                    steps_txt = font_sm.render(steps_str, True, COLOR_TEXT_MUTED)
                    screen.blit(steps_txt, (col_x[4], row_y + 4))

                    # Efficiency
                    if r.get("outcome") == "escaped":
                        opt = r.get("optimal_path_len", 0)
                        act = max(1, r.get("player_path_len", 1))
                        eff_val = min(100, int(100 * opt / act))
                        eff_txt = font_sm.render(f"{eff_val}%", True, COLOR_TEXT_SUCCESS if eff_val >= 75 else COLOR_TEXT_PRIMARY)
                    else:
                        eff_txt = font_sm.render("—", True, COLOR_TEXT_MUTED)
                    screen.blit(eff_txt, (col_x[5], row_y + 4))

                    # Backtracks
                    bt_txt = font_sm.render(f"{r.get('backtracks', 0)} turns", True, COLOR_TEXT_MUTED)
                    screen.blit(bt_txt, (col_x[6], row_y + 4))

                    # Skill Label
                    sk_txt = font_sm.render(f"{r.get('skill_label') or 'Beginner'}", True, COLOR_TEXT_ACCENT)
                    screen.blit(sk_txt, (col_x[7], row_y + 4))

        # RENDER: PLAYING / PAUSED / GAME OVER COCKPIT
        else:
            # 1. Top HUD Bar
            pygame.draw.rect(screen, COLOR_HUD_BG, (0, 0, SCREEN_WIDTH, 56))
            pygame.draw.line(screen, COLOR_CARD_BORDER, (0, 56), (SCREEN_WIDTH, 56), 2)

            brand_surf = title_font.render("MAZE MIND", True, COLOR_TEXT_PRIMARY)
            screen.blit(brand_surf, (MARGIN + 6, 14))

            sector_tag = f"Sector: {config.width}x{config.height} ({selected_tier})"
            sec_surf = font_bold.render(sector_tag, True, COLOR_TEXT_ACCENT)
            screen.blit(sec_surf, (MARGIN + 150, 17))

            # Dynamic transient event banner (e.g. "BURST DASH!", "HUNTER STUNNED!")
            if engine.recent_events:
                ev_text = engine.recent_events[-1]["text"]
                ev_surf = font_bold.render(ev_text, True, COLOR_TEXT_WARNING)
                screen.blit(ev_surf, (MARGIN + 360, 17))

            btn_pause.draw(screen)

            # 2. Left Stage: Maze Canvas
            stage_rect = pygame.Rect(MARGIN, 68, MAZE_STAGE_WIDTH, MAZE_STAGE_HEIGHT)
            pygame.draw.rect(screen, COLOR_STAGE_BG, stage_rect, border_radius=10)
            pygame.draw.rect(screen, COLOR_CARD_BORDER, stage_rect, width=2, border_radius=10)

            # Draw Maze, Footprints, Radar Trail, and Walls
            visited_set = set(engine.path_taken)
            pulse_alpha = int(180 + 50 * math.sin(time.monotonic() * 10)) if radar_guide_timer > 0 else 0
            _draw_maze(
                screen, engine.maze, offset_x, offset_y, cell_size,
                visited_cells=visited_set,
                radar_guide_path=radar_guide_path,
                radar_pulse_alpha=pulse_alpha,
            )

            # Exit Portal (Glowing Dimensional Beacon)
            ex, ey = _cell_center(engine.maze.exit, offset_x, offset_y, cell_size)
            _draw_exit_portal(screen, ex, ey, cell_size)

            # Player Entity (Tactical Operative Badge with Directional Chevron)
            px, py = _cell_center(engine.player.position, offset_x, offset_y, cell_size)
            _draw_player_icon(screen, px, py, cell_size, engine.player.last_move_direction)

            # Hunter Entity & Visual Threat Aura
            hx, hy = _cell_center(engine.hunter.position, offset_x, offset_y, cell_size)

            if engine.hunter.is_awakening:
                pulse_r = int(cell_size * 0.8 + 4 * math.sin(time.monotonic() * 6))
                s_aura = pygame.Surface((pulse_r * 2, pulse_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(s_aura, (251, 191, 36, 70), (pulse_r, pulse_r), pulse_r)
                screen.blit(s_aura, (hx - pulse_r, hy - pulse_r))

                awake_str = f"SLEEP {engine.hunter.awakening_time_left_s:.1f}s"
                awake_surf = font_xs.render(awake_str, True, COLOR_TEXT_WARNING)
                screen.blit(awake_surf, awake_surf.get_rect(center=(hx, hy - cell_size // 2 - 8)))

            elif engine.hunter.is_stunned:
                s_aura = pygame.Surface((cell_size * 2, cell_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s_aura, (148, 163, 184, 110), (cell_size, cell_size), int(cell_size * 0.75))
                screen.blit(s_aura, (hx - cell_size, hy - cell_size))

                stun_str = f"STUN {engine.hunter.stun_time_left_s:.1f}s"
                stun_surf = font_xs.render(stun_str, True, (147, 197, 253))
                screen.blit(stun_surf, stun_surf.get_rect(center=(hx, hy - cell_size // 2 - 8)))

            elif engine.hunter.state == HunterState.PURSUIT:
                pulse = int(cell_size * 1.3 + 6 * math.sin(time.monotonic() * 9))
                s_aura = pygame.Surface((pulse * 2, pulse * 2), pygame.SRCALPHA)
                pygame.draw.circle(s_aura, (239, 68, 68, 65), (pulse, pulse), pulse)
                screen.blit(s_aura, (hx - pulse, hy - pulse))

            # Sinister Hunter Avatar Icon
            _draw_hunter_icon(screen, hx, hy, cell_size)

            # 3. Right Stage: Command Console
            # Card 1: Tactical Radar & Mission Status
            card1_rect = pygame.Rect(CONSOLE_X, 68, CONSOLE_WIDTH, 175)
            pygame.draw.rect(screen, COLOR_CARD, card1_rect, border_radius=10)
            pygame.draw.rect(screen, COLOR_CARD_BORDER, card1_rect, width=2, border_radius=10)

            c1_title = font_bold.render("🎯 MISSION TELEMETRY", True, COLOR_TEXT_PRIMARY)
            screen.blit(c1_title, (CONSOLE_X + 16, 78))

            # Time Remaining Gauge
            time_left = max(0.0, config.round_time_limit_s - engine.elapsed_s)
            time_ratio = max(0.0, time_left / config.round_time_limit_s)
            _draw_meter(
                screen, font_xs, CONSOLE_X + 16, 122, CONSOLE_WIDTH - 32, 10,
                time_ratio, "⏱ TIME REMAINING", f"{time_left:.1f}s", low_is_bad=True
            )

            # Win Probability Gauge
            if winprob_available:
                try:
                    wp = predict_win_probability(engine.win_probability_features())
                except Exception:
                    wp = 0.5
            else:
                wp = 0.5

            wp_label = "FAVORABLE" if wp >= 0.65 else ("CONTESTED" if wp >= 0.35 else "CRITICAL")
            _draw_meter(
                screen, font_xs, CONSOLE_X + 16, 158, CONSOLE_WIDTH - 32, 10,
                wp, "📊 WIN PROBABILITY", f"{int(wp * 100)}% ({wp_label})", low_is_bad=True
            )

            # Hunter Distance Sensor & Compass
            dist_hunter = engine.maze.shortest_path_len(engine.player.position, engine.hunter.position)
            h_status = "STUNNED" if engine.hunter.is_stunned else (
                "AWAKENING" if engine.hunter.is_awakening else (
                    "PURSUIT" if engine.hunter.state == HunterState.PURSUIT else "PATROL"
                )
            )
            h_color = COLOR_HUNTER if dist_hunter <= 5 else (COLOR_TEXT_WARNING if dist_hunter <= 9 else COLOR_TEXT_SUCCESS)
            sensor_str = f"💀 HUNTER: {dist_hunter} cells away [{h_status}]"
            s_surf = font_xs.render(sensor_str, True, h_color)
            screen.blit(s_surf, (CONSOLE_X + 16, 180))

            dist_exit = engine.maze.shortest_path_len(engine.player.position, engine.maze.exit)
            compass_str = f"🏁 EXIT: {dist_exit} cells away"
            comp_surf = font_xs.render(compass_str, True, COLOR_TEXT_MUTED)
            screen.blit(comp_surf, (CONSOLE_X + 16, 198))

            # Dead-End Alert
            open_count = sum(1 for d in engine.maze.DIRS if not engine.maze.walls[engine.player.position[1]][engine.player.position[0]][d])
            if open_count <= 1 and engine.outcome == Outcome.PLAYING:
                dead_surf = font_xs.render("⚠️ DEAD END — Turn back!", True, (248, 113, 113))
                screen.blit(dead_surf, (CONSOLE_X + 16, 218))

            # Card 2: Tactical Abilities Deck
            deck_rect = pygame.Rect(CONSOLE_X, 252, CONSOLE_WIDTH, 206)
            pygame.draw.rect(screen, COLOR_CARD, deck_rect, border_radius=10)
            pygame.draw.rect(screen, COLOR_CARD_BORDER, deck_rect, width=2, border_radius=10)

            c2_title = font_bold.render("⚡ TACTICAL ABILITIES", True, COLOR_TEXT_PRIMARY)
            screen.blit(c2_title, (CONSOLE_X + 16, 262))

            btn_dash.disabled = not engine.player.can_dash
            btn_dash.badge_text = "READY" if engine.player.can_dash else f"{engine.player.dash_cooldown_remaining_s:.1f}s"
            btn_dash.draw(screen)

            btn_smoke.disabled = not engine.player.can_smoke
            btn_smoke.badge_text = f"{engine.player.smoke_charges}x LEFT" if engine.player.can_smoke else "EMPTY"
            btn_smoke.draw(screen)

            btn_radar.disabled = not engine.player.can_radar
            btn_radar.badge_text = f"{engine.player.radar_charges}x LEFT" if engine.player.can_radar else "EMPTY"
            btn_radar.draw(screen)

            # Card 3: Navigation Console
            nav_rect = pygame.Rect(CONSOLE_X, 468, CONSOLE_WIDTH, 208)
            pygame.draw.rect(screen, COLOR_CARD, nav_rect, border_radius=10)
            pygame.draw.rect(screen, COLOR_CARD_BORDER, nav_rect, width=2, border_radius=10)

            c3_title = font_bold.render("🧭 NAVIGATION CONSOLE", True, COLOR_TEXT_PRIMARY)
            screen.blit(c3_title, (CONSOLE_X + 16, 478))

            for btn in arrow_buttons:
                btn.draw(screen)

            nav_hint = font_xs.render("Use WASD, Arrow keys, or Click", True, COLOR_TEXT_MUTED)
            screen.blit(nav_hint, nav_hint.get_rect(center=(CONSOLE_X + CONSOLE_WIDTH // 2, 622)))

            steps_taken = max(0, len(engine.path_taken) - 1)
            telemetry_str = f"Steps: {steps_taken}  •  Backtracks: {engine.backtracks}"
            tel_surf = font_xs.render(telemetry_str, True, COLOR_TEXT_ACCENT)
            screen.blit(tel_surf, tel_surf.get_rect(center=(CONSOLE_X + CONSOLE_WIDTH // 2, 642)))

            # Quick Actions Row
            btn_restart_round.draw(screen)
            btn_menu_return.draw(screen)

            # 4. PAUSE OVERLAY MODAL
            if current_state == STATE_PAUSED:
                dim_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                dim_surf.fill((8, 11, 18, 215))
                screen.blit(dim_surf, (0, 0))

                pygame.draw.rect(screen, COLOR_CARD, (pause_modal_x, pause_modal_y, pause_modal_w, pause_modal_h), border_radius=14)
                pygame.draw.rect(screen, (99, 102, 241), (pause_modal_x, pause_modal_y, pause_modal_w, pause_modal_h), width=2, border_radius=14)

                p_title = banner_font.render("⏸  GAME PAUSED", True, COLOR_TEXT_PRIMARY)
                screen.blit(p_title, p_title.get_rect(center=(pause_modal_x + pause_modal_w // 2, pause_modal_y + 45)))

                p_sub = font_sm.render(f"Sector: {config.width}x{config.height} Grid  •  Time: {engine.elapsed_s:.1f}s", True, COLOR_TEXT_ACCENT)
                screen.blit(p_sub, p_sub.get_rect(center=(pause_modal_x + pause_modal_w // 2, pause_modal_y + 85)))

                p_guide = font_xs.render("Abilities: [Shift] Dash  •  [F] Smoke  •  [R] Radar", True, COLOR_TEXT_MUTED)
                screen.blit(p_guide, p_guide.get_rect(center=(pause_modal_x + pause_modal_w // 2, pause_modal_y + 110)))

                btn_resume_pause.draw(screen)
                btn_restart_pause.draw(screen)
                btn_history_pause.draw(screen)
                btn_menu_pause.draw(screen)
                btn_quit_pause.draw(screen)

            # 5. GAME OVER MODAL
            elif current_state == STATE_GAMEOVER:
                dim_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                dim_surf.fill((8, 11, 18, 215))
                screen.blit(dim_surf, (0, 0))

                pygame.draw.rect(screen, COLOR_CARD, (go_modal_x, go_modal_y, gameover_modal_w, gameover_modal_h), border_radius=14)
                pygame.draw.rect(screen, (99, 102, 241), (go_modal_x, go_modal_y, gameover_modal_w, gameover_modal_h), width=2, border_radius=14)

                if engine.outcome == Outcome.ESCAPED:
                    outcome_title = "YOU ESCAPED!"
                    outcome_color = COLOR_EXIT
                elif engine.outcome == Outcome.CAUGHT:
                    outcome_title = "CAUGHT BY HUNTER!"
                    outcome_color = COLOR_HUNTER
                else:
                    outcome_title = "TIME EXPIRED!"
                    outcome_color = COLOR_TEXT_WARNING

                ot_surf = banner_font.render(outcome_title, True, outcome_color)
                screen.blit(ot_surf, ot_surf.get_rect(center=(go_modal_x + gameover_modal_w // 2, go_modal_y + 40)))

                path_len = max(1, len(engine.path_taken) - 1)
                optimal = engine.maze.shortest_path_len(engine.maze.start, engine.maze.exit)
                eff = int(100 * optimal / path_len)

                stats_lines = [
                    f"Time Taken: {engine.elapsed_s:.1f}s   •   Path Efficiency: {eff}%",
                    f"Steps Taken: {path_len}   •   Optimal Route: {optimal}",
                    f"Current Skill Tier: {current_skill} (Move: {config.hunter_move_interval / 30:.2f}s/step)",
                ]
                for i, s in enumerate(stats_lines):
                    s_surf = font.render(s, True, COLOR_TEXT_PRIMARY if i == 2 else COLOR_TEXT_MUTED)
                    screen.blit(s_surf, s_surf.get_rect(center=(go_modal_x + gameover_modal_w // 2, go_modal_y + 90 + i * 28)))

                btn_replay.draw(screen)
                btn_history_go.draw(screen)
                btn_menu_go.draw(screen)

        pygame.display.flip()
        clock.tick(config.fps)

        frame_count += 1
        if headless_for_testing and frame_count >= frames_to_render:
            running = False

    pygame.quit()
    return engine.outcome
