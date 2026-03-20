"""Scene 5: Popularity correlations — near-zero bars vs one tall energy-loudness bar."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, EDGE_BUFF, PALETTE


class PopularityFlatScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Popularity Correlations vs Feature-Feature", font_size=TITLE_SIZE - 4, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Popularity correlation with each feature (absolute values for bar height)
        features = [
            "dance", "energy", "loud", "speech", "acoust",
            "valence", "tempo", "instr", "liven", "durat",
        ]
        pop_corrs = [0.04, 0.02, 0.02, 0.06, 0.03, 0.10, 0.01, 0.02, 0.01, 0.04]

        # Add energy-loudness for contrast
        all_names = features + ["E–L"]
        all_vals = pop_corrs + [0.63]

        chart = BarChart(
            values=[0] * len(all_vals),
            bar_names=all_names,
            y_range=[0, 0.7, 0.1],
            y_length=4.0,
            x_length=11,
            x_axis_config={"font_size": 16, "label_direction": DOWN},
            y_axis_config={"font_size": 16, "include_numbers": True},
            bar_colors=[PALETTE[0]] * len(features) + [ACCENT],
            bar_width=0.55,
            bar_fill_opacity=0.85,
        )
        chart.next_to(title, DOWN, buff=0.6)

        self.play(Create(chart), run_time=0.8)
        self.play(chart.animate.change_bar_values(all_vals), run_time=1.5, rate_func=smooth)

        bar_labels = chart.get_bar_labels(font_size=16, color=TEXT_COLOR)
        self.play(FadeIn(bar_labels), run_time=0.5)

        # Annotations
        note1 = Text("Popularity correlations: all ≤ 0.10", font_size=LABEL_SIZE, color=PALETTE[0])
        note1.to_edge(DOWN, buff=EDGE_BUFF + 0.4)
        note2 = Text("Energy–Loudness: 0.63 (feature-feature)", font_size=LABEL_SIZE, color=ACCENT)
        note2.next_to(note1, DOWN, buff=0.15)

        self.play(FadeIn(note1), run_time=0.4)
        self.play(FadeIn(note2), run_time=0.4)

        self.wait(5)
        self.play(FadeOut(VGroup(title, chart, bar_labels, note1, note2)), run_time=0.8)
