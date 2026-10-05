"""Share navigation requests between interfaces and the CLI launcher.

Returning to Home is a screen transition rather than a completed round. A
separate signal lets the launcher reopen view selection without inventing a
game outcome or saving an unfinished round.
"""

from enum import Enum


class NavigationAction(Enum):
    """Represent a request to leave a view so the launcher can handle navigation
    independently of escape, capture, or timeout results.
    """

    HOME = "home"
