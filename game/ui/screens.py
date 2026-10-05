"""Draw classic menu, career history, pause, and result screens from supplied state. These
functions keep non-gameplay layouts separate from the main classic event loop.

Menu, history, pause, and results views.
"""

import pygame
from game.engine import Outcome
from game.session import tier_description
from game.ui.theme import (
    COLOR_CARD,
    COLOR_CARD_BORDER,
    COLOR_EXIT,
    COLOR_HUD_BG,
    COLOR_HUNTER,
    COLOR_TEXT_ACCENT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SUCCESS,
    COLOR_TEXT_WARNING,
    MARGIN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)


def draw_menu(
    *,
    base_config,
    big_banner_font,
    btn_history_menu,
    btn_quit_menu,
    btn_start_game,
    current_skill,
    font,
    font_bold,
    font_sm,
    menu_tier_buttons,
    player_tag,
    screen,
    selected_tier,
):
    """Explain controls and difficulty choices before classic play so a player can select
    an appropriate challenge.
    """
    title_surf = big_banner_font.render("MAZE MIND", True, COLOR_TEXT_PRIMARY)
    screen.blit(title_surf, title_surf.get_rect(center=(SCREEN_WIDTH // 2, 65)))

    sub_surf = font_bold.render("Find the exit. Outsmart the hunter. Build your next escape.", True, COLOR_TEXT_ACCENT)
    screen.blit(sub_surf, sub_surf.get_rect(center=(SCREEN_WIDTH // 2, 110)))

    # Instructions Card
    card_w, card_h = 680, 180
    card_x = (SCREEN_WIDTH - card_w) // 2
    card_y = 145
    pygame.draw.rect(screen, COLOR_CARD, (card_x, card_y, card_w, card_h), border_radius=12)
    pygame.draw.rect(screen, COLOR_CARD_BORDER, (card_x, card_y, card_w, card_h), width=2, border_radius=12)

    lines = [
        (f"OPERATOR: {player_tag}   •   ACTIVE SKILL TIER: {current_skill}", COLOR_TEXT_ACCENT, font_bold),
        ("FIRST ESCAPE: Recruit gives you a gold route guide and a slower hunter.", COLOR_TEXT_PRIMARY, font),
        ("GET PAST THE HUNTER: [F] Smoke lets you pass safely for 3 seconds.", COLOR_TEXT_WARNING, font),
        ("HOLD WASD / ARROWS to move • [Shift] Dash • [R] Route hint", COLOR_TEXT_MUTED, font_sm),
        ("PAUSE ANYTIME: Press [P] or click the top-right Pause button to freeze time.", COLOR_TEXT_SUCCESS, font_sm),
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
    tier_desc = tier_description(base_config, selected_tier, current_skill)
    desc_surf = font_sm.render(tier_desc, True, COLOR_TEXT_WARNING)
    screen.blit(desc_surf, desc_surf.get_rect(center=(SCREEN_WIDTH // 2, 430)))

    btn_start_game.draw(screen)
    btn_history_menu.draw(screen)
    btn_quit_menu.draw(screen)


def draw_history(
    *,
    banner_font,
    btn_back_history,
    current_skill,
    font,
    font_bold,
    font_sm,
    font_xs,
    player_summary,
    player_tag,
    screen,
    title_font,
):
    # Header
    """Display saved career summaries and recent rounds so players can review progress
    through the classic UI.
    """
    pygame.draw.rect(screen, COLOR_HUD_BG, (0, 0, SCREEN_WIDTH, 56))
    pygame.draw.line(screen, COLOR_CARD_BORDER, (0, 56), (SCREEN_WIDTH, 56), 2)
    btn_back_history.draw(screen)

    dossier_title = title_font.render("OPERATIVE DOSSIER & CAREER RECORDS", True, COLOR_TEXT_PRIMARY)
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
            sk_txt = font_sm.render(f"{r.get('predicted_skill') or r.get('skill_label') or 'Unrated'}", True, COLOR_TEXT_ACCENT)
            screen.blit(sk_txt, (col_x[7], row_y + 4))


def draw_pause(
    *,
    banner_font,
    btn_history_pause,
    btn_menu_pause,
    btn_quit_pause,
    btn_restart_pause,
    btn_resume_pause,
    config,
    engine,
    font_sm,
    font_xs,
    pause_modal_h,
    pause_modal_w,
    pause_modal_x,
    pause_modal_y,
    screen,
):
    """Show resume, restart, history, and exit actions above the frozen board so pausing
    does not lose the current round context.
    """
    dim_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    dim_surf.fill((8, 11, 18, 215))
    screen.blit(dim_surf, (0, 0))

    pygame.draw.rect(screen, COLOR_CARD, (pause_modal_x, pause_modal_y, pause_modal_w, pause_modal_h), border_radius=14)
    pygame.draw.rect(screen, (99, 102, 241), (pause_modal_x, pause_modal_y, pause_modal_w, pause_modal_h), width=2, border_radius=14)

    p_title = banner_font.render(" GAME PAUSED", True, COLOR_TEXT_PRIMARY)
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


def draw_results(
    *,
    banner_font,
    btn_history_go,
    btn_menu_go,
    btn_replay,
    config,
    current_skill,
    engine,
    font,
    gameover_modal_h,
    gameover_modal_w,
    go_modal_x,
    go_modal_y,
    screen,
    session,
):
    """Explain the completed outcome and navigation performance so the player receives
    useful feedback before replaying or changing level.
    """
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
    eff = f"{min(100, int(100 * optimal / path_len))}%" if engine.outcome == Outcome.ESCAPED else "N/A"

    remaining = engine.maze.shortest_path_len(engine.player_pos, engine.maze.exit)
    feedback = ("Nice escape! Try Runner when you want a tougher chase." if engine.outcome == Outcome.ESCAPED
                else f"{remaining} steps from the exit. Use F to pass the hunter, or try Recruit.")
    stats_lines = [
        f"Time Taken: {engine.elapsed_s:.1f}s   •   Path Efficiency: {eff}",
        f"Steps Taken: {path_len}   •   Optimal Route: {optimal}",
        f"Current Skill Tier: {current_skill} (Move: {config.hunter_move_interval / config.fps:.2f}s/step)",
        feedback,
        f"Playstyle: {session.style or 'Play a few rounds to find your style'}",
    ]
    for i, s in enumerate(stats_lines):
        s_surf = font.render(s, True, COLOR_TEXT_PRIMARY if i == 2 else COLOR_TEXT_MUTED)
        screen.blit(s_surf, s_surf.get_rect(center=(go_modal_x + gameover_modal_w // 2, go_modal_y + 90 + i * 28)))

    btn_replay.draw(screen)
    btn_history_go.draw(screen)
    btn_menu_go.draw(screen)
