"""Scene 12: Regression R-squared comparison — BarChart with Ridge almost invisible."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, NUMBER_SIZE, EDGE_BUFF, PALETTE

MODELS = ["Ridge", "Grad. Boosting", "Random Forest"]
R2_SCORES = [0.018, 0.190, 0.267]
RMSE_SCORES = [29.0, 26.2, 25.0]
COLORS = [PALETTE[5], PALETTE[2], ACCENT]


class RegressionBarsScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Regression R² Comparison (5-Fold CV)", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        chart = BarChart(
            values=[0, 0, 0],
            bar_names=MODELS,
            y_range=[0, 0.30, 0.05],
            y_length=4.5,
            x_length=9,
            x_axis_config={"font_size": 20, "label_direction": DOWN},
            y_axis_config={"font_size": 18, "include_numbers": True},
            bar_colors=COLORS,
            bar_width=0.6,
            bar_fill_opacity=0.85,
        )
        chart.next_to(title, DOWN, buff=0.7)

        y_label = Text("R²", font_size=LABEL_SIZE, color=TEXT_COLOR)
        y_label.next_to(chart.y_axis, UP, buff=0.2)

        self.play(Create(chart), FadeIn(y_label), run_time=0.8)
        self.play(chart.animate.change_bar_values(R2_SCORES), run_time=1.5, rate_func=smooth)

        bar_labels = chart.get_bar_labels(font_size=20, color=TEXT_COLOR)
        self.play(FadeIn(bar_labels), run_time=0.5)

        # RMSE annotations below each bar
        rmse_labels = VGroup()
        for i, (rmse, bar) in enumerate(zip(RMSE_SCORES, chart.bars)):
            txt = Text(f"RMSE={rmse:.0f}", font_size=16, color=GREY_A)
            txt.next_to(bar, DOWN, buff=0.6)
            rmse_labels.add(txt)
        self.play(FadeIn(rmse_labels), run_time=0.5)

        # Highlight Ridge disaster
        note = Text("Ridge R²=0.018 ≈ predicting the mean", font_size=LABEL_SIZE, color=PALETTE[5])
        note.to_edge(DOWN, buff=EDGE_BUFF)
        self.play(FadeIn(note), run_time=0.5)

        self.wait(5)
        self.play(FadeOut(VGroup(title, chart, bar_labels, y_label, rmse_labels, note)), run_time=0.8)
