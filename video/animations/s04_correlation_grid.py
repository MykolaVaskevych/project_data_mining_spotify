"""Scene 4: Correlation heatmap — 2D grid of colored squares with values."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, EDGE_BUFF, corr_color

# Subset of features for readability (8x8 grid)
FEATURES = ["dance", "energy", "loud", "speech", "acoust", "valence", "tempo", "pop"]

# Approximate correlation matrix (from notebook findings)
CORR = [
    [ 1.00,  0.15,  0.13,  0.14, -0.20,  0.42,  0.08, -0.04],
    [ 0.15,  1.00,  0.63, -0.06, -0.54, -0.01,  0.12,  0.02],
    [ 0.13,  0.63,  1.00, -0.05, -0.42, -0.01,  0.02, -0.02],
    [ 0.14, -0.06, -0.05,  1.00, -0.05,  0.02,  0.01, -0.06],
    [-0.20, -0.54, -0.42, -0.05,  1.00, -0.01, -0.08,  0.03],
    [ 0.42, -0.01, -0.01,  0.02, -0.01,  1.00, -0.05, -0.10],
    [ 0.08,  0.12,  0.02,  0.01, -0.08, -0.05,  1.00,  0.01],
    [-0.04,  0.02, -0.02, -0.06,  0.03, -0.10,  0.01,  1.00],
]


class CorrelationGridScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Correlation Matrix", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        n = len(FEATURES)
        cell_size = 0.7
        grid = VGroup()
        texts = VGroup()

        for i in range(n):
            for j in range(n):
                val = CORR[i][j]
                sq = Square(side_length=cell_size)
                sq.set_fill(color=corr_color(val), opacity=0.85)
                sq.set_stroke(WHITE, width=0.5)
                grid.add(sq)

                # Show value text for strong correlations
                if abs(val) >= 0.10 or i == j:
                    txt = Text(f"{val:.2f}", font_size=12, color=BLACK if abs(val) > 0.3 else WHITE)
                    texts.add(txt)
                else:
                    texts.add(VGroup())  # placeholder

        grid.arrange_in_grid(n, n, buff=0.04)
        # Position text on each square
        for k, txt in enumerate(texts):
            if isinstance(txt, Text):
                txt.move_to(grid[k])

        # Row labels (left side)
        row_labels = VGroup()
        for i, feat in enumerate(FEATURES):
            lbl = Text(feat, font_size=LABEL_SIZE - 4, color=TEXT_COLOR)
            lbl.next_to(grid[i * n], LEFT, buff=0.2)
            row_labels.add(lbl)

        # Column labels (top)
        col_labels = VGroup()
        for j, feat in enumerate(FEATURES):
            lbl = Text(feat, font_size=LABEL_SIZE - 6, color=TEXT_COLOR)
            lbl.next_to(grid[j], UP, buff=0.2)
            lbl.rotate(45 * DEGREES)
            col_labels.add(lbl)

        # Position everything below title
        full_chart = VGroup(grid, texts, row_labels, col_labels)
        full_chart.move_to(ORIGIN).shift(DOWN * 0.3)

        self.play(FadeIn(grid, lag_ratio=0.005), run_time=1.5)
        self.play(FadeIn(texts), FadeIn(row_labels), FadeIn(col_labels), run_time=0.8)

        # Highlight the energy-loudness cell (row 1, col 2 = index 1*8+2=10)
        el_idx = 1 * n + 2
        highlight = SurroundingRectangle(grid[el_idx], color=YELLOW, buff=0.02, stroke_width=3)
        el_note = Text("Energy–Loudness: 0.63", font_size=LABEL_SIZE, color=ACCENT)
        el_note.to_edge(DOWN, buff=EDGE_BUFF + 0.3)
        self.play(Create(highlight), FadeIn(el_note), run_time=0.6)

        self.wait(3)

        # Highlight popularity row (last row — near-zero values)
        pop_highlights = VGroup()
        for j in range(n - 1):
            idx = (n - 1) * n + j
            rect = SurroundingRectangle(grid[idx], color=RED, buff=0.02, stroke_width=2)
            pop_highlights.add(rect)

        pop_note = Text("Popularity row: all near zero", font_size=LABEL_SIZE, color=RED_C)
        pop_note.next_to(el_note, UP, buff=0.2)
        self.play(
            FadeOut(highlight), FadeOut(el_note),
            Create(pop_highlights, lag_ratio=0.05),
            FadeIn(pop_note),
            run_time=1.0,
        )

        self.wait(4)
        self.play(FadeOut(VGroup(title, full_chart, pop_highlights, pop_note)), run_time=0.8)
