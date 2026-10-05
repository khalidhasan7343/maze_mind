"""Project logical grid positions and height into an isometric playfield. Drawing geometry
and interpolating motion here gives the game a raised appearance while the engine keeps
simple grid-based collisions.

Orthographic isometric geometry, independent of round and screen state.

World coordinates stay on the existing maze grid. Height is projected into
screen space, so raised walls and characters need no extra graphics dependency.
"""

import math
from dataclasses import dataclass

import pygame

INK = (34, 43, 61)
MUTED = (113, 122, 137)
BG = (245, 246, 248)
LINE = (226, 230, 237)
BLUE = (91, 92, 226)
RED = (223, 104, 103)
GREEN = (43, 160, 139)


@dataclass(frozen=True)
class IsoProjection:
    """Hold the grid-to-screen transform in an immutable object so rendering and tile-click
    inversion use exactly the same scale and origin.
    """
    origin: tuple[float, float]
    half_width: float

    @classmethod
    def fit(cls, rect, width, height):
        """Choose a scale and origin for the available panel so square and rectangular
        mazes fit the same isometric view.
        """
        scale = max(1.0, min((rect.width - 64) / (width + height),
                             (rect.height - 90) / ((width + height) * .5 + .7)))
        return cls((rect.centerx - (width - height) * scale / 2,
                    rect.centery - (width + height) * scale / 4 + 16), scale)

    def project(self, x, y, z=0):
        """Map grid position and height into screen pixels so floor tiles, walls, and
        characters use one consistent projection.
        """
        return (self.origin[0] + (x - y) * self.half_width,
                self.origin[1] + (x + y) * self.half_width * .5 - z * self.half_width)

    def cell_at(self, point, width, height):
        """Map a screen click back to a floor cell so the isometric UI can request ordinary
        engine movement instead of changing game coordinates.

        Invert the floor projection; raised walls remain impassable in the engine.
        """
        dx = (point[0] - self.origin[0]) / self.half_width
        dy = (point[1] - self.origin[1]) / (self.half_width * .5)
        x, y = math.floor((dx + dy) / 2), math.floor((dy - dx) / 2)
        return (x, y) if 0 <= x < width and 0 <= y < height else None


def _polygon(surface, color, points):
    """Keep polygon filling behind a small drawing helper so the projected geometry uses a
    consistent surface operation.
    """
    pygame.draw.polygon(surface, color, points)


def _block(surface, projection, x, y, width, depth, height, colors, bottom=0):
    """Draw a prism from its projected faces so raised walls and character blocks share the
    same geometry and shading rules.

    Draw the two visible faces and the top of an axis-aligned prism.
    """
    p = projection.project
    a, b, c, d = [p(px, py, height) for px, py in
                  [(x, y), (x + width, y), (x + width, y + depth), (x, y + depth)]]
    _polygon(surface, colors[1], [b, p(x + width, y, bottom),
                                  p(x + width, y + depth, bottom), c])
    _polygon(surface, colors[2], [d, c, p(x + width, y + depth, bottom),
                                  p(x, y + depth, bottom)])
    _polygon(surface, colors[0], [a, b, c, d])


class Motion:
    """Keep visual interpolation separate from logical grid movement so smooth animation
    cannot change collisions or saved telemetry.

    Interpolate presentation only; collisions always use engine cell coordinates.
    """

    def __init__(self):
        """Allocate independent interpolation state so each displayed character can animate
        without changing its engine position.
        """
        self.target = None
        self.source = None
        self.changed_at = 0.0

    def position(self, target, now):
        """Ease between recent grid positions so discrete engine steps appear smooth while
        collision checks stay tied to real cells.
        """
        if self.target is None:
            self.source = self.target = target
        if target != self.target:
            self.source = self.position(self.target, now)
            self.target = target
            self.changed_at = now
        t = min(1.0, max(0.0, (now - self.changed_at) / .12))
        t = t * t * (3 - 2 * t)
        return tuple(a + (b - a) * t for a, b in zip(self.source, self.target))


class IsometricRenderer:
    """Own projection-based drawing and actor animations so the controller can render a
    board without implementing geometry itself.
    """
    def __init__(self):
        """Give player and hunter separate animation tracks so their movement can be
        interpolated independently.
        """
        self.player_motion = Motion()
        self.hunter_motion = Motion()

    def draw(self, surface, rect, maze, *, player=None, hunter=None, guide=(),
             now=0.0, stunned=False, font=None):
        """Render the maze and entities in depth order and return the projection so the UI
        can draw raised geometry and interpret tile clicks consistently.
        """
        projection = IsoProjection.fit(rect, maze.width, maze.height)
        p = projection.project
        scale = projection.half_width
        previous_clip = surface.get_clip()
        surface.set_clip(rect)
        shadow = pygame.Rect(0, 0, int((maze.width + maze.height) * scale * .84),
                             int((maze.width + maze.height) * scale * .38))
        shadow.center = (rect.centerx, int(rect.centery + scale * .9))
        pygame.draw.ellipse(surface, (235, 237, 242), shadow)
        _block(surface, projection, 0, 0, maze.width, maze.height, 0,
               ((231, 235, 242), (196, 203, 215), (211, 217, 228)), bottom=-.28)
        visited = set(guide)
        for y in range(maze.height):
            for x in range(maze.width):
                color = (239, 241, 247) if (x + y) % 2 else (246, 247, 251)
                if (x, y) in visited:
                    color = (224, 224, 252)
                if (x, y) == maze.exit:
                    color = (203, 238, 230)
                _polygon(surface, color, [p(x + .025, y + .025), p(x + .975, y + .025),
                                          p(x + .975, y + .975), p(x + .025, y + .975)])
        if guide and player is not None:
            points = [p(x + .5, y + .5, .015) for x, y in [player, *guide]]
            pygame.draw.lines(surface, (164, 161, 235), False, points, max(2, int(scale * .075)))
            for point in points[1:]:
                pygame.draw.circle(surface, BLUE, point, max(2, int(scale * .065)))

        # Unique wall edges plus characters share a back-to-front depth order.
        items = []
        thickness = .065
        for y in range(maze.height):
            for x in range(maze.width):
                walls = maze.walls[y][x]
                segments = []
                if walls["N"]:
                    segments.append((x, y - thickness / 2, 1, thickness))
                if walls["W"]:
                    segments.append((x - thickness / 2, y, thickness, 1))
                if x == maze.width - 1 and walls["E"]:
                    segments.append((x + 1 - thickness / 2, y, thickness, 1))
                if y == maze.height - 1 and walls["S"]:
                    segments.append((x, y + 1 - thickness / 2, 1, thickness))
                for sx, sy, w, d in segments:
                    items.append((sx + sy + (w + d) / 2, "wall", (sx, sy, w, d)))
        ex, ey = maze.exit
        items.append((ex + ey + 1, "exit", (ex + .36, ey + .36)))
        actors = []
        if player is not None:
            actors.append(("player", self.player_motion.position(player, now)))
        if hunter is not None:
            actors.append(("hunter", self.hunter_motion.position(hunter, now)))
        for kind, (x, y) in actors:
            items.append((x + y + 1, kind, (x + .33, y + .33)))
        for _, kind, coords in sorted(items, key=lambda item: item[0]):
            if kind == "wall":
                _block(surface, projection, *coords, .38,
                       ((255, 255, 255), (176, 186, 202), (201, 209, 221)))
            elif kind == "exit":
                _block(surface, projection, *coords, .28, .28, .52,
                       ((91, 205, 180), (31, 129, 113), GREEN))
            else:
                color = BLUE if kind == "player" else RED
                if kind == "hunter" and stunned:
                    color = (162, 166, 183)
                lighter = tuple(min(255, c + 28) for c in color)
                darker = tuple(max(0, c - 35) for c in color)
                _block(surface, projection, *coords, .34, .34, .65, (lighter, darker, color))
                # A white crest makes the player identifiable without color alone.
                if kind == "player":
                    pygame.draw.circle(surface, (255, 255, 255),
                                       p(coords[0] + .17, coords[1] + .17, .66), max(2, int(scale * .065)))
        if font:
            for kind, (x, y) in actors:
                label = font.render("YOU" if kind == "player" else "HUNTER", True,
                                    BLUE if kind == "player" else RED)
                at = p(x + .5, y + .5, .95)
                surface.blit(label, label.get_rect(midbottom=at))
        surface.set_clip(previous_clip)
        return projection
