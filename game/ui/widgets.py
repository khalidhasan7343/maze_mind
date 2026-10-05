"""Provide reusable classic buttons and status meters. The screens share these widgets so
hover behavior, disabled controls, and metric displays follow consistent presentation
rules.

Reusable buttons and telemetry meters.
"""

import pygame
from game.ui.theme import (
    COLOR_METER_BG,
    COLOR_METER_FILL,
    COLOR_METER_FILL_LOW,
    COLOR_METER_FILL_MID,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_WARNING,
)

class Button:
    """Reuse geometry, appearance, hover state, and click detection across classic screens
    so button behavior remains consistent.

    Clickable UI button supporting hover states, disabled styling, and badges.
    """

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
        """Store button geometry, colors, and labels so classic screens can reuse the same
        clickable control instead of reimplementing it.
        """
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
        """Refresh hover state from the mouse position so enabled buttons give visual
        feedback before a click.
        """
        self.is_hovered = self.rect.collidepoint(mouse_pos) and not self.disabled

    def draw(self, screen: pygame.Surface) -> None:
        """Render normal, hovered, or disabled styling and optional badges so button
        appearance communicates its current availability.
        """
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
        """Accept left clicks only inside an enabled button so mouse handlers cannot
        activate disabled or unrelated controls.
        """
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
    """Draw a bounded status ratio with optional warning colors so time and progress values
    are readable at a glance.

    Render a progress meter with gradient fill and labels.
    """
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


