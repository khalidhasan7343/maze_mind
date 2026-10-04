"""Maze, character, and portal rendering."""

import math
import time
import pygame
from game.ui.theme import (
    COLOR_EXIT,
    COLOR_FLOOR,
    COLOR_PLAYER,
    COLOR_PLAYER_GLOW,
    COLOR_RADAR_TRAIL,
    COLOR_WALL,
    COLOR_WALL_BORDER,
)

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
        centers = [(x * cell_size + cell_size // 2, y * cell_size + cell_size // 2) for x, y in radar_guide_path]
        for a, b in zip(centers, centers[1:]):
            pygame.draw.line(s_trail, (*COLOR_RADAR_TRAIL, radar_pulse_alpha), a, b, max(2, cell_size // 12))
        for idx, (gx, gy) in enumerate(radar_guide_path):
            cx = gx * cell_size + cell_size // 2
            cy = gy * cell_size + cell_size // 2
            r = max(4, cell_size // 4)
            alpha = max(30, min(230, radar_pulse_alpha - idx * 15))
            pygame.draw.circle(s_trail, (*COLOR_RADAR_TRAIL, alpha), (cx, cy), r)
            pygame.draw.circle(s_trail, (255, 255, 255, alpha), (cx, cy), max(2, r // 2))
            if idx == 0:
                pygame.draw.circle(s_trail, (255, 255, 255, alpha), (cx, cy), r + 3, width=2)
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

