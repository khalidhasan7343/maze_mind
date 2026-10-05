"""Provide the original classic interface and its event loop. This optional view remains
available through Home or --view classic and shares RoundSession with the isometric
interface.

Interactive screen controller: menu, play, pause, history, and results.
"""

from __future__ import annotations

import math
import os
import time
import pygame
from game.engine import RoundEngine, RoundConfig, Outcome
from game.astar import HunterState
from data.logger import RoundLogger
from game.session import RoundSession, TIERS, tier_description
from game.ui import screens
from game.ui.controls import KEY_TO_DIR, HeldMovement, move_toward_cell, radar_guide
from game.ui.theme import (
    COLOR_BG,
    COLOR_CARD,
    COLOR_CARD_BORDER,
    COLOR_HUD_BG,
    COLOR_HUNTER,
    COLOR_STAGE_BG,
    COLOR_TEXT_ACCENT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SUCCESS,
    COLOR_TEXT_WARNING,
    CONSOLE_WIDTH,
    CONSOLE_X,
    MARGIN,
    MAZE_STAGE_HEIGHT,
    MAZE_STAGE_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STATE_GAMEOVER,
    STATE_HISTORY,
    STATE_MENU,
    STATE_PAUSED,
    STATE_PLAYING,
)
from game.ui.widgets import Button, _draw_meter
from game.ui.rendering import (_create_window_icon, _draw_player_icon, _draw_hunter_icon, _draw_exit_portal, _draw_maze, _cell_center)
from game.ui.navigation import NavigationAction

def run_round(
    config: RoundConfig,
    player_tag: str = "player",
    skill_hint: str | None = None,
    logger: RoundLogger | None = None,
    headless_for_testing: bool = False,
    initial_tier: str = "Recruit",
) -> Outcome | NavigationAction:
    """Coordinate the classic screens, input, engine, and session services so the original
    flat-board interface remains available alongside isometric play.

    Execute the full interactive Maze Mind UI loop with Tactical Cockpit, History, and
    Pause.
    Back to Home returns a navigation request rather than an outcome so the
    launcher can reopen view selection without logging an unfinished round.
    """
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

    selected_tier = initial_tier
    session = RoundSession(config, player_tag, logger, skill_hint)
    current_skill = session.skill
    base_config = session.base_config

    # Apply Selected Tier Preset
    def configure_round_engine(tier_name: str) -> RoundEngine:
        """Start the selected preset through RoundSession and refresh local settings so the
        classic HUD reflects the actual active difficulty.
        """
        nonlocal config, current_skill
        engine = session.start(tier_name)
        config = engine.config
        current_skill = session.skill
        return engine

    engine = configure_round_engine(selected_tier)

    # Calculate Responsive Cell Sizing
    def calculate_cell_layout():
        """Size and center grid cells for the current maze so changing difficulty does not
        leave the board using the previous dimensions.
        """
        c_size = max(2, min(52, 620 // engine.maze.width, 620 // engine.maze.height))
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
        "Pause [P]", font_sm, bg_color=(35, 42, 60), hover_color=(55, 68, 98),
    )

    # 2. Tactical Ability Buttons (Right Console)
    btn_dash = Button(
        (CONSOLE_X, 290, CONSOLE_WIDTH, 48),
        "BURST DASH [Shift]", font_bold, bg_color=(28, 38, 62), hover_color=(45, 65, 105),
    )
    btn_smoke = Button(
        (CONSOLE_X, 345, CONSOLE_WIDTH, 48),
        "SMOKE FREEZE [F]", font_bold, bg_color=(46, 32, 50), hover_color=(75, 48, 82),
    )
    btn_radar = Button(
        (CONSOLE_X, 400, CONSOLE_WIDTH, 48),
        "SONAR RADAR [R]", font_bold, bg_color=(24, 46, 44), hover_color=(36, 75, 70),
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
        "Restart Level", font_sm, bg_color=(35, 42, 60), hover_color=(50, 60, 85),
    )
    btn_menu_return = Button(
        (CONSOLE_X + 170, 690, 155, 38),
        "Main Menu", font_sm, bg_color=(28, 34, 48), hover_color=(45, 52, 70),
    )

    # 4. Start Menu Tier Buttons & Action Buttons
    menu_tier_buttons = {}
    tier_names = TIERS
    for i, tname in enumerate(tier_names):
        bx = SCREEN_WIDTH // 2 - 400 + i * 162
        menu_tier_buttons[tname] = Button(
            (bx, 360, 154, 50),
            tname, font_bold,
            bg_color=(30, 36, 52) if tname != selected_tier else (37, 99, 235),
            hover_color=(50, 65, 95),
        )

    btn_start_game = Button(
        (SCREEN_WIDTH // 2 - 130, 485, 260, 52),
        "Start Game [Enter]", font_bold, bg_color=(37, 99, 235), hover_color=(59, 130, 246),
    )
    btn_history_menu = Button(
        (SCREEN_WIDTH // 2 - 130, 545, 260, 42),
        "Career Records & Logs", font, bg_color=(35, 44, 65), hover_color=(50, 65, 95),
    )
    btn_quit_menu = Button(
        (SCREEN_WIDTH // 2 - 130, 595, 260, 40),
        "Quit to Desktop", font_sm, bg_color=(28, 34, 48), hover_color=(45, 52, 70),
    )
    btn_home_menu = Button(
        (MARGIN, 20, 180, 36), "Back to Home [Esc]", font_sm,
        bg_color=(35, 44, 65), hover_color=(50, 65, 95),
    )

    # 5. Pause Modal Buttons
    pause_modal_w, pause_modal_h = 420, 370
    pause_modal_x = (SCREEN_WIDTH - pause_modal_w) // 2
    pause_modal_y = (SCREEN_HEIGHT - pause_modal_h) // 2

    btn_resume_pause = Button(
        (pause_modal_x + 35, pause_modal_y + 140, 350, 44),
        "Resume Game [P]", font_bold, bg_color=(37, 99, 235), hover_color=(59, 130, 246),
    )
    btn_restart_pause = Button(
        (pause_modal_x + 35, pause_modal_y + 195, 350, 40),
        "Restart Level [R]", font, bg_color=(38, 46, 66), hover_color=(55, 68, 96),
    )
    btn_history_pause = Button(
        (pause_modal_x + 35, pause_modal_y + 245, 350, 40),
        "Career Records", font, bg_color=(32, 40, 58), hover_color=(48, 62, 90),
    )
    btn_menu_pause = Button(
        (pause_modal_x + 35, pause_modal_y + 295, 170, 40),
        "Main Menu [M]", font_sm, bg_color=(28, 34, 48), hover_color=(45, 55, 75),
    )
    btn_quit_pause = Button(
        (pause_modal_x + 215, pause_modal_y + 295, 170, 40),
        "Quit [Esc]", font_sm, bg_color=(28, 32, 44), hover_color=(45, 50, 68),
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
        "Records", font, bg_color=(38, 46, 66), hover_color=(55, 68, 96),
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
    held_movement = HeldMovement()

    def start_new_round():
        """Settle completed telemetry and reset controls, guide state, and layout so a
        restart begins with a fresh round and correct HUD.
        """
        nonlocal engine, round_recorded, cell_size, offset_x, offset_y, radar_guide_path, radar_guide_timer, player_summary
        session.finish()
        held_movement.clear()
        engine = configure_round_engine(selected_tier)
        cell_size, offset_x, offset_y = calculate_cell_layout()
        round_recorded = False
        radar_guide_path = []
        radar_guide_timer = 0.0
        player_summary = logger.get_player_summary(player_tag)

    running = True
    requested_home = False
    frames_to_render = 3 if headless_for_testing else None
    frame_count = 0

    while running:
        now = time.monotonic()
        dt = max(0.001, min(0.1, now - last_tick_time))
        last_tick_time = now

        mouse_pos = pygame.mouse.get_pos()

        # Update Radar timer
        if radar_guide_timer > 0 and current_state == STATE_PLAYING:
            radar_guide_timer = max(0.0, radar_guide_timer - dt)
            if radar_guide_timer == 0:
                radar_guide_path = []

        # ------------------ EVENT HANDLING ------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYUP:
                held_movement.release(event.key)
            if event.type == pygame.WINDOWFOCUSLOST:
                held_movement.clear()
                if current_state == STATE_PLAYING:
                    engine.pause()
                    current_state = STATE_PAUSED

            # STATE: START MENU
            if current_state == STATE_MENU:
                btn_home_menu.update(mouse_pos)
                if btn_home_menu.is_clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    requested_home = True
                    held_movement.clear()
                    running = False
                    break
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
                elif btn_quit_menu.is_clicked(event):
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
                    guide = radar_guide(engine)
                    if guide:
                        radar_guide_path, radar_guide_timer = guide

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
                        move_toward_cell(engine, (gx, gy))

                # Keyboard Hotkeys
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        engine.pause()
                        current_state = STATE_PAUSED
                    elif event.key in KEY_TO_DIR and engine.outcome == Outcome.PLAYING:
                        direction = held_movement.press(event.key, now)
                        if direction:
                            engine.move_player(direction)
                    elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                        engine.use_ability("dash")
                    elif event.key == pygame.K_f:
                        engine.use_ability("smoke")
                    elif event.key == pygame.K_r:
                        guide = radar_guide(engine)
                        if guide:
                            radar_guide_path, radar_guide_timer = guide

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
        if current_state == STATE_PLAYING:
            if engine.outcome == Outcome.PLAYING:
                direction = held_movement.update(now)
                if direction:
                    engine.move_player(direction)
                engine.update()
                session.capture_snapshot()
            if engine.outcome != Outcome.PLAYING:
                current_state = STATE_GAMEOVER
        else:
            held_movement.clear()

        # Automatic logging & difficulty adaptation on round conclusion
        if session.finish():
            current_skill = session.skill
            player_summary = session.summary

        # ------------------ RENDERING ------------------
        screen.fill(COLOR_BG)

        # RENDER: START MENU
        if current_state == STATE_MENU:
            screens.draw_menu(
                base_config=base_config,
                big_banner_font=big_banner_font,
                btn_history_menu=btn_history_menu,
                btn_quit_menu=btn_quit_menu,
                btn_home_menu=btn_home_menu,
                btn_start_game=btn_start_game,
                current_skill=current_skill,
                font=font,
                font_bold=font_bold,
                font_sm=font_sm,
                menu_tier_buttons=menu_tier_buttons,
                player_tag=player_tag,
                screen=screen,
                selected_tier=selected_tier,
            )

        # RENDER: CAREER HISTORY
        elif current_state == STATE_HISTORY:
            screens.draw_history(
                banner_font=banner_font,
                btn_back_history=btn_back_history,
                current_skill=current_skill,
                font=font,
                font_bold=font_bold,
                font_sm=font_sm,
                font_xs=font_xs,
                player_summary=player_summary,
                player_tag=player_tag,
                screen=screen,
                title_font=title_font,
            )

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
            guide_path = engine.route_guide_path() if config.route_guide else radar_guide_path
            pulse_alpha = int(180 + 50 * math.sin(time.monotonic() * 10)) if config.route_guide or radar_guide_timer > 0 else 0
            _draw_maze(
                screen, engine.maze, offset_x, offset_y, cell_size,
                visited_cells=visited_set,
                radar_guide_path=guide_path,
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

            c1_title = font_bold.render("MISSION TELEMETRY", True, COLOR_TEXT_PRIMARY)
            screen.blit(c1_title, (CONSOLE_X + 16, 78))

            # Time Remaining Gauge
            time_left = max(0.0, config.round_time_limit_s - engine.elapsed_s)
            time_ratio = max(0.0, time_left / config.round_time_limit_s) if config.round_time_limit_s else 0.0
            _draw_meter(
                screen, font_xs, CONSOLE_X + 16, 122, CONSOLE_WIDTH - 32, 10,
                time_ratio, "TIME REMAINING", f"{time_left:.1f}s", low_is_bad=True
            )

            # Give players concrete feedback; bot-trained estimates remain secondary.
            wp = session.win_predictor(engine.win_probability_features()) if engine.outcome == Outcome.PLAYING else (1.0 if engine.outcome == Outcome.ESCAPED else 0.0)
            _draw_meter(
                screen, font_xs, CONSOLE_X + 16, 158, CONSOLE_WIDTH - 32, 10,
                engine.route_progress, "EXIT PROGRESS", f"{int(engine.route_progress * 100)}%", low_is_bad=False
            )

            # Hunter Distance Sensor & Compass
            dist_hunter = engine.maze.shortest_path_len(engine.player.position, engine.hunter.position)
            h_status = "STUNNED" if engine.hunter.is_stunned else (
                "AWAKENING" if engine.hunter.is_awakening else (
                    "PURSUIT" if engine.hunter.state == HunterState.PURSUIT else "PATROL"
                )
            )
            h_color = COLOR_HUNTER if dist_hunter <= 5 else (COLOR_TEXT_WARNING if dist_hunter <= 9 else COLOR_TEXT_SUCCESS)
            sensor_str = f"HUNTER: {dist_hunter} cells away [{h_status}]"
            s_surf = font_xs.render(sensor_str, True, h_color)
            screen.blit(s_surf, (CONSOLE_X + 16, 180))

            dist_exit = engine.maze.shortest_path_len(engine.player.position, engine.maze.exit)
            compass_str = f"EXIT: {dist_exit} steps"
            if config.route_guide and guide_path:
                next_cell = guide_path[0]
                step = next_cell[0] - engine.player_pos[0], next_cell[1] - engine.player_pos[1]
                heading = {(0, -1): "UP [W]", (0, 1): "DOWN [S]", (-1, 0): "LEFT [A]", (1, 0): "RIGHT [D]"}[step]
                compass_str += f" • NEXT: {heading}"
            comp_surf = font_xs.render(compass_str, True, COLOR_TEXT_MUTED)
            screen.blit(comp_surf, (CONSOLE_X + 16, 198))

            model_text = f"AI escape estimate: {int(wp * 100)}%" if wp is not None else "AI estimate: Model unavailable"
            screen.blit(font_xs.render(model_text, True, COLOR_TEXT_MUTED), (CONSOLE_X + 16, 218))

            # Card 2: Tactical Abilities Deck
            deck_rect = pygame.Rect(CONSOLE_X, 252, CONSOLE_WIDTH, 206)
            pygame.draw.rect(screen, COLOR_CARD, deck_rect, border_radius=10)
            pygame.draw.rect(screen, COLOR_CARD_BORDER, deck_rect, width=2, border_radius=10)

            c2_title = font_bold.render("TACTICAL ABILITIES", True, COLOR_TEXT_PRIMARY)
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

            c3_title = font_bold.render("NAVIGATION CONSOLE", True, COLOR_TEXT_PRIMARY)
            screen.blit(c3_title, (CONSOLE_X + 16, 478))

            for btn in arrow_buttons:
                btn.draw(screen)

            nav_hint = font_xs.render("Hold WASD / Arrows" + (" • Follow gold trail" if config.route_guide else " • R for route hint"), True, COLOR_TEXT_MUTED)
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
                screens.draw_pause(
                    banner_font=banner_font,
                    btn_history_pause=btn_history_pause,
                    btn_menu_pause=btn_menu_pause,
                    btn_quit_pause=btn_quit_pause,
                    btn_restart_pause=btn_restart_pause,
                    btn_resume_pause=btn_resume_pause,
                    config=config,
                    engine=engine,
                    font_sm=font_sm,
                    font_xs=font_xs,
                    pause_modal_h=pause_modal_h,
                    pause_modal_w=pause_modal_w,
                    pause_modal_x=pause_modal_x,
                    pause_modal_y=pause_modal_y,
                    screen=screen,
                )

            # 5. GAME OVER MODAL
            elif current_state == STATE_GAMEOVER:
                screens.draw_results(
                    banner_font=banner_font,
                    btn_history_go=btn_history_go,
                    btn_menu_go=btn_menu_go,
                    btn_replay=btn_replay,
                    config=config,
                    current_skill=current_skill,
                    engine=engine,
                    font=font,
                    gameover_modal_h=gameover_modal_h,
                    gameover_modal_w=gameover_modal_w,
                    go_modal_x=go_modal_x,
                    go_modal_y=go_modal_y,
                    screen=screen,
                    session=session,
                )

        pygame.display.flip()
        clock.tick(config.fps)

        frame_count += 1
        if headless_for_testing and frame_count >= frames_to_render:
            running = False

    pygame.quit()
    if requested_home:
        return NavigationAction.HOME
    return engine.outcome
