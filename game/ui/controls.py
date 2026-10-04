"""Input mappings and grid navigation shared by keyboard and mouse controls."""

import pygame

KEY_TO_DIR = {
    pygame.K_w: "N", pygame.K_UP: "N",
    pygame.K_s: "S", pygame.K_DOWN: "S",
    pygame.K_a: "W", pygame.K_LEFT: "W",
    pygame.K_d: "E", pygame.K_RIGHT: "E",
}


class HeldMovement:
    """Immediate key presses plus predictable repeats, independent of OS repeat settings."""

    def __init__(self, interval_s=0.15, initial_delay_s=0.22):
        self.interval_s = interval_s
        self.initial_delay_s = initial_delay_s
        self.keys = []
        self.next_move_at = 0.0

    def press(self, key, now):
        if key not in KEY_TO_DIR or key in self.keys:
            return None
        self.keys.append(key)
        self.next_move_at = now + self.initial_delay_s
        return KEY_TO_DIR[key]

    def release(self, key):
        if key in self.keys:
            self.keys.remove(key)

    def update(self, now):
        if not self.keys or now < self.next_move_at:
            return None
        self.next_move_at = now + self.interval_s
        return KEY_TO_DIR[self.keys[-1]]

    def clear(self):
        self.keys.clear()


def move_toward_cell(engine, cell):
    px, py = engine.player_pos
    delta = cell[0] - px, cell[1] - py
    direction = next((d for d, step in engine.maze.DIRS.items() if step == delta), None)
    return engine.move_player(direction) if direction else False


def radar_guide(engine):
    info = engine.use_ability("radar")
    return (info.get("guide_path", []), 4.0) if isinstance(info, dict) else None
