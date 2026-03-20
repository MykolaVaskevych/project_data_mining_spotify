"""Scene 6: Silhouette scores for k=2..15 — flat profile using BarChart."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, EDGE_BUFF, PALETTE

# Approximate silhouette scores from notebook (k=2..15)
K_VALUES = list(range(2, 16))
SCORES = [0.155, 0.140, 0.130, 0.128, 0.127, 0.122, 0.118, 0.117,
          0.116, 0.115, 0.114, 0.113, 0.112, 0.111]


class SilhouetteBarsScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Silhouette Scores (k = 2–15)", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        bar_names = [f"k={k}" for k in K_VALUES]
        # Highlight k=5 (matching genre count)
        colors = [PALETTE[0]] * len(K_VALUES)
        colors[3] = ACCENT  # k=5 is index 3

        chart = BarChart(
            values=[0] * len(SCORES),
            bar_names=bar_names,
            y_range=[0, 0.20, 0.05],
            y_length=4.0,
            x_length=12,
            x_axis_config={"font_size": 14, "label_direction": DOWN},
            y_axis_config={"font_size": 16, "include_numbers": True},
            bar_colors=colors,
            bar_width=0.55,
            bar_fill_opacity=0.85,
        )
        chart.next_to(title, DOWN, buff=0.6)

        self.play(Create(chart), run_time=0.8)
        self.play(chart.animate.change_bar_values(SCORES), run_time=1.5, rate_func=smooth)

        bar_labels = chart.get_bar_labels(font_size=12, color=TEXT_COLOR)
        self.play(FadeIn(bar_labels), run_time=0.5)

        # Annotation
        note = Text("All scores below 0.2 — no well-separated clusters", font_size=LABEL_SIZE, color=ACCENT)
        note.to_edge(DOWN, buff=EDGE_BUFF)
        self.play(FadeIn(note), run_time=0.5)

        # Highlight k=5 bar
        highlight = SurroundingRectangle(chart.bars[3], color=ACCENT, buff=0.05, stroke_width=2)
        k5_note = Text("k=5 (matches 5 genres)", font_size=LABEL_SIZE - 2, color=ACCENT)
        k5_note.next_to(chart.bars[3], UP, buff=0.6)
        self.play(Create(highlight), FadeIn(k5_note), run_time=0.5)

        self.wait(5)
        self.play(FadeOut(VGroup(title, chart, bar_labels, note, highlight, k5_note)), run_time=0.8)
