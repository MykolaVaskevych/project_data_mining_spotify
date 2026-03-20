"""Shared constants and helpers for all animation scenes."""

from manim import (
    BLUE_C, BLUE_D, BLUE_E, TEAL_C, GREEN_C, GREEN_D, YELLOW_C, ORANGE, RED_C, RED_D,
    WHITE, GREY, GREY_A, DARK_GREY,
    UP, DOWN, LEFT, RIGHT, ORIGIN, OUT,
    DEGREES,
    Text, VGroup,
    interpolate_color, BLUE, RED, GREEN, YELLOW,
)

# ── Palette ──────────────────────────────────────────────────────────────
PALETTE = [BLUE_C, TEAL_C, GREEN_C, YELLOW_C, ORANGE, RED_C]
BAR_PALETTE = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]
BG_COLOR = "#1a1a2e"
ACCENT = YELLOW_C
TEXT_COLOR = WHITE
SUBTITLE_COLOR = GREY_A

# Genre-specific colors (consistent across all scenes)
GENRE_COLORS = {
    "pop": "#4e79a7",
    "indie-pop": "#59a14f",
    "synth-pop": "#9c755f",
    "hip-hop": "#f28e2b",
    "r-n-b": "#e15759",
}
GENRE_LIST = ["pop", "indie-pop", "synth-pop", "hip-hop", "r-n-b"]

# ── Camera defaults ──────────────────────────────────────────────────────
DEFAULT_PHI = 65 * DEGREES
DEFAULT_THETA = -40 * DEGREES

# ── Font sizes ───────────────────────────────────────────────────────────
TITLE_SIZE = 40
SUBTITLE_SIZE = 30
BODY_SIZE = 26
LABEL_SIZE = 22
NUMBER_SIZE = 30

# ── Spacing ──────────────────────────────────────────────────────────────
EDGE_BUFF = 0.5


def make_title(text, font_size=TITLE_SIZE):
    """Create a title positioned at the top with safe margin."""
    return Text(text, font_size=font_size, color=TEXT_COLOR).to_edge(UP, buff=EDGE_BUFF)


def make_subtitle(text, font_size=SUBTITLE_SIZE, ref=None):
    """Create a subtitle below the title area or a reference mobject."""
    t = Text(text, font_size=font_size, color=SUBTITLE_COLOR)
    if ref is not None:
        t.next_to(ref, DOWN, buff=0.25)
    else:
        t.to_edge(UP, buff=EDGE_BUFF + 0.65)
    return t


def make_note(text, font_size=LABEL_SIZE, color=ACCENT):
    """Create a bottom annotation."""
    return Text(text, font_size=font_size, color=color).to_edge(DOWN, buff=EDGE_BUFF)


def corr_color(value):
    """Map a correlation value (-1 to 1) to a blue-white-red color."""
    if value >= 0:
        return interpolate_color(WHITE, RED, min(abs(value), 1.0))
    else:
        return interpolate_color(WHITE, BLUE, min(abs(value), 1.0))


def safe_add_fixed(scene, *mobjects):
    """HUD overlay: titles, footnotes. Stays pinned to screen position."""
    for m in mobjects:
        scene.add_fixed_in_frame_mobjects(m)


def safe_add_orient(scene, *mobjects):
    """3D-attached label: faces camera but tracks world position during rotation."""
    for m in mobjects:
        scene.add_fixed_orientation_mobjects(m)
