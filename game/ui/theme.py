"""Define classic layout dimensions, colors, and screen-state names. Shared constants keep
the classic controller, widgets, and drawing helpers visually consistent.

Shared layout, colors, and screen states.
"""

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

