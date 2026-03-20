"""Scene 11: Feature importance for classification — horizontal BarChart."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, EDGE_BUFF, PALETTE

# Top 10 features (approx from notebook — all 6-10%)
FEATURES = [
    "acousticness", "tempo", "danceability", "speechiness", "loudness",
    "energy", "valence", "duration_ms", "instrumentalness", "liveness",
]
IMPORTANCES = [0.098, 0.094, 0.091, 0.088, 0.085, 0.082, 0.080, 0.075, 0.070, 0.065]


class FeatureImportanceClfScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Feature Importance (Classification)", font_size=TITLE_SIZE - 2, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Build horizontal bars manually (BarChart is vertical-only)
        n = len(FEATURES)
        max_imp = max(IMPORTANCES)
        bar_height = 0.35
        bar_spacing = 0.48
        max_bar_width = 7.0

        bars = VGroup()
        labels = VGroup()
        val_texts = VGroup()

        for i, (feat, imp) in enumerate(zip(FEATURES, IMPORTANCES)):
            y = (n / 2 - 0.5 - i) * bar_spacing

            # Bar
            width = (imp / max_imp) * max_bar_width
            t = imp / max_imp
            color = interpolate_color(PALETTE[0], ACCENT, t)
            bar = Rectangle(width=width, height=bar_height)
            bar.set_fill(color=color, opacity=0.85)
            bar.set_stroke(WHITE, width=0.5)
            bar.move_to([-0.5 + width / 2, y, 0])
            bars.add(bar)

            # Feature name (left of bar)
            lbl = Text(feat, font_size=LABEL_SIZE - 4, color=TEXT_COLOR)
            lbl.next_to(bar, LEFT, buff=0.15)
            labels.add(lbl)

            # Value (right of bar)
            val = Text(f"{imp:.1%}", font_size=LABEL_SIZE - 4, color=TEXT_COLOR)
            val.next_to(bar, RIGHT, buff=0.15)
            val_texts.add(val)

        content = VGroup(bars, labels, val_texts)
        content.move_to(ORIGIN).shift(DOWN * 0.3)

        # Animate bars growing from left
        for bar in bars:
            bar.save_state()
            bar.stretch(0.01, 0)
            bar.move_to(bar.get_left(), aligned_edge=LEFT)

        self.play(
            *[Restore(bar) for bar in bars],
            FadeIn(labels), FadeIn(val_texts),
            run_time=1.5,
            rate_func=smooth,
        )

        # Annotation
        note = Text("No single feature dominates — all contribute 6–10%", font_size=LABEL_SIZE, color=ACCENT)
        note.to_edge(DOWN, buff=EDGE_BUFF)
        self.play(FadeIn(note), run_time=0.5)

        self.wait(5)
        self.play(FadeOut(VGroup(title, content, note)), run_time=0.8)
