"""Scene 13: Dramatic Ridge vs RF comparison with animated value change."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, NUMBER_SIZE, EDGE_BUFF, PALETTE


class RidgeDisasterScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Ridge vs Random Forest", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Start with both at the same level, then animate to show the gap
        chart = BarChart(
            values=[0.15, 0.15],
            bar_names=["Ridge Regression", "Random Forest"],
            y_range=[0, 0.30, 0.05],
            y_length=5.0,
            x_length=8,
            x_axis_config={"font_size": 22, "label_direction": DOWN},
            y_axis_config={"font_size": 18, "include_numbers": True},
            bar_colors=[PALETTE[5], ACCENT],
            bar_width=0.65,
            bar_fill_opacity=0.85,
        )
        chart.next_to(title, DOWN, buff=0.6)

        y_label = Text("R²", font_size=LABEL_SIZE + 2, color=TEXT_COLOR)
        y_label.next_to(chart.y_axis, UP, buff=0.2)

        self.play(Create(chart), FadeIn(y_label), run_time=0.8)

        # Dramatic reveal — Ridge drops, RF rises
        self.play(
            chart.animate.change_bar_values([0.018, 0.267]),
            run_time=2.0,
            rate_func=smooth,
        )

        bar_labels = chart.get_bar_labels(font_size=24, color=TEXT_COLOR)
        self.play(FadeIn(bar_labels), run_time=0.5)

        # "15x better" annotation with arrow
        mult_text = Text("15x", font_size=TITLE_SIZE + 8, color=ACCENT, weight=BOLD)
        mult_text.move_to(chart.bars[0].get_top() + RIGHT * 2.0 + UP * 1.5)
        arrow = Arrow(
            chart.bars[0].get_top() + UP * 0.3,
            chart.bars[1].get_top() + UP * 0.3,
            color=ACCENT, stroke_width=4, max_tip_length_to_length_ratio=0.15,
        )
        self.play(GrowArrow(arrow), FadeIn(mult_text, scale=1.3), run_time=0.8)

        # Bottom annotations
        ridge_note = Text("Ridge: no better than predicting the mean", font_size=LABEL_SIZE, color=PALETTE[5])
        ridge_note.to_edge(DOWN, buff=EDGE_BUFF + 0.3)
        rf_note = Text("RF: captures non-linear feature interactions", font_size=LABEL_SIZE, color=ACCENT)
        rf_note.next_to(ridge_note, UP, buff=0.15)
        self.play(FadeIn(ridge_note), FadeIn(rf_note), run_time=0.6)

        self.wait(5)
        self.play(FadeOut(VGroup(title, chart, bar_labels, y_label, mult_text, arrow, ridge_note, rf_note)), run_time=0.8)
