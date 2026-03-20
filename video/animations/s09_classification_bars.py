"""Scene 9: Classification model comparison — BarChart with animated growth."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, NUMBER_SIZE, EDGE_BUFF, PALETTE


MODELS = ["Logistic Reg.", "Random Forest", "Grad. Boosting"]
ACCURACIES = [0.607, 0.702, 0.672]
COLORS = [PALETTE[0], ACCENT, PALETTE[2]]


class ClassificationBarsScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Classification Accuracy (5-Fold CV)", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        chart = BarChart(
            values=[0.5, 0.5, 0.5],  # baseline
            bar_names=MODELS,
            y_range=[0.5, 0.75, 0.05],
            y_length=4.5,
            x_length=9,
            x_axis_config={"font_size": 20, "label_direction": DOWN},
            y_axis_config={"font_size": 18, "include_numbers": True},
            bar_colors=COLORS,
            bar_width=0.6,
            bar_fill_opacity=0.85,
        )
        chart.next_to(title, DOWN, buff=0.7)

        # Y-axis label
        y_label = Text("Accuracy", font_size=LABEL_SIZE, color=TEXT_COLOR)
        y_label.next_to(chart.y_axis, UP, buff=0.2)

        self.play(Create(chart), FadeIn(y_label), run_time=0.8)

        # Animate bars growing from 0.5 baseline
        self.play(chart.animate.change_bar_values(ACCURACIES), run_time=1.5, rate_func=smooth)

        bar_labels = chart.get_bar_labels(font_size=22, color=TEXT_COLOR)
        self.play(FadeIn(bar_labels), run_time=0.5)

        # Highlight RF (best)
        highlight = SurroundingRectangle(chart.bars[1], color=ACCENT, buff=0.05, stroke_width=3)
        note = Text("+10pp from linear → non-linear", font_size=LABEL_SIZE, color=ACCENT)
        note.to_edge(DOWN, buff=EDGE_BUFF)

        # Draw arrow between LR and RF bars
        arrow = Arrow(
            chart.bars[0].get_top() + UP * 0.3,
            chart.bars[1].get_top() + UP * 0.3,
            color=ACCENT, stroke_width=3,
        )
        self.play(Create(highlight), Create(arrow), FadeIn(note), run_time=0.8)

        self.wait(5)
        self.play(FadeOut(VGroup(title, chart, bar_labels, y_label, highlight, note, arrow)), run_time=0.8)
