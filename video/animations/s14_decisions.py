"""Scene 14: Key decisions — Improved (LR→RF) and Reduced (Ridge) side by side."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, BODY_SIZE, LABEL_SIZE, NUMBER_SIZE, EDGE_BUFF, PALETTE


class DecisionsScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Key Modelling Decisions", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # ── LEFT: IMPROVED ──
        imp_title = Text("IMPROVED", font_size=BODY_SIZE, color=GREEN_C, weight=BOLD)
        imp_sub = Text("LR → Random Forest", font_size=LABEL_SIZE, color=TEXT_COLOR)

        clf_before = Text("60.7%", font_size=NUMBER_SIZE + 4, color=PALETTE[5])
        clf_arrow = Text("→", font_size=NUMBER_SIZE - 4, color=TEXT_COLOR)
        clf_after = Text("70.2%", font_size=NUMBER_SIZE + 4, color=GREEN_C)
        clf_row = VGroup(clf_before, clf_arrow, clf_after).arrange(RIGHT, buff=0.25)
        clf_label = Text("Classification Accuracy", font_size=LABEL_SIZE - 2, color=GREY_A)

        r2_before = Text("0.018", font_size=NUMBER_SIZE + 4, color=PALETTE[5])
        r2_arrow = Text("→", font_size=NUMBER_SIZE - 4, color=TEXT_COLOR)
        r2_after = Text("0.267", font_size=NUMBER_SIZE + 4, color=GREEN_C)
        r2_row = VGroup(r2_before, r2_arrow, r2_after).arrange(RIGHT, buff=0.25)
        r2_label = Text("Regression R²", font_size=LABEL_SIZE - 2, color=GREY_A)

        imp_group = VGroup(imp_title, imp_sub, clf_label, clf_row, r2_label, r2_row)
        imp_group.arrange(DOWN, buff=0.25, aligned_edge=LEFT)

        # ── RIGHT: REDUCED ──
        red_title = Text("REDUCED", font_size=BODY_SIZE, color=PALETTE[5], weight=BOLD)
        red_sub = Text("Ridge for Popularity", font_size=LABEL_SIZE, color=TEXT_COLOR)

        ridge_r2 = Text("R² = 0.018", font_size=NUMBER_SIZE + 4, color=PALETTE[5])
        ridge_rmse = Text("RMSE = 29", font_size=NUMBER_SIZE, color=PALETTE[5])
        ridge_note = Text("≈ predicting the mean", font_size=LABEL_SIZE - 2, color=GREY_A)
        ridge_insight = Text("Most informative failure", font_size=LABEL_SIZE, color=ACCENT)

        red_group = VGroup(red_title, red_sub, ridge_r2, ridge_rmse, ridge_note, ridge_insight)
        red_group.arrange(DOWN, buff=0.25, aligned_edge=LEFT)

        # Divider
        divider = Line(UP * 2.5, DOWN * 2.5, color=GREY).set_opacity(0.3)

        # Position everything
        imp_group.move_to(LEFT * 3.3 + DOWN * 0.2)
        red_group.move_to(RIGHT * 3.3 + DOWN * 0.2)
        divider.move_to(ORIGIN + DOWN * 0.2)

        self.play(FadeIn(divider), run_time=0.3)
        self.play(FadeIn(imp_group, shift=RIGHT * 0.3), run_time=1.0)
        self.play(FadeIn(red_group, shift=LEFT * 0.3), run_time=1.0)

        # Checkmark and X icons
        check = Text("✓", font_size=40, color=GREEN_C)
        check.next_to(imp_title, RIGHT, buff=0.3)
        cross = Text("✗", font_size=40, color=PALETTE[5])
        cross.next_to(red_title, RIGHT, buff=0.3)
        self.play(FadeIn(check, scale=1.5), FadeIn(cross, scale=1.5), run_time=0.5)

        self.wait(7)
        self.play(FadeOut(VGroup(title, imp_group, red_group, divider, check, cross)), run_time=0.8)
