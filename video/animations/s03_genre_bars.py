"""Scene 3: Genre distribution using Manim's built-in BarChart."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, EDGE_BUFF, GENRE_COLORS, GENRE_LIST


GENRE_COUNTS = [490, 487, 399, 292, 292]  # pop, indie-pop, synth-pop, hip-hop, r-n-b


class GenreBarsScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Genre Distribution", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        chart = BarChart(
            values=[0] * 5,  # start from zero for animation
            bar_names=GENRE_LIST,
            y_range=[0, 550, 100],
            y_length=4.5,
            x_length=10,
            x_axis_config={"font_size": 20, "label_direction": DOWN},
            y_axis_config={"font_size": 18, "include_numbers": True},
            bar_colors=[GENRE_COLORS[g] for g in GENRE_LIST],
            bar_width=0.6,
            bar_fill_opacity=0.85,
        )
        chart.next_to(title, DOWN, buff=0.6)

        self.play(Create(chart), run_time=1.0)

        # Animate bars growing to actual values
        self.play(
            chart.animate.change_bar_values(GENRE_COUNTS),
            run_time=1.5,
            rate_func=smooth,
        )

        # Add value labels on top of bars
        bar_labels = chart.get_bar_labels(font_size=22, color=TEXT_COLOR)
        self.play(FadeIn(bar_labels), run_time=0.5)

        # Axis label
        y_label = Text("Tracks", font_size=LABEL_SIZE, color=TEXT_COLOR)
        y_label.next_to(chart.y_axis, UP, buff=0.2)
        self.play(FadeIn(y_label), run_time=0.3)

        self.wait(4)
        self.play(FadeOut(VGroup(title, chart, bar_labels, y_label)), run_time=0.8)
