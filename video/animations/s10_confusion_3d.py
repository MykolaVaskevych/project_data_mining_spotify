"""Scene 10: Confusion matrix — 2D colored grid with highlighted cells."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, NUMBER_SIZE, EDGE_BUFF, PALETTE

# Approximate confusion matrix from test set (RF classifier)
CM = [[155, 44], [61, 132]]
CM_LABELS = [["TN", "FP"], ["FN", "TP"]]


class ConfusionMatrixScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Confusion Matrix — Random Forest", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Build 2x2 grid
        cell_size = 2.0
        max_val = 155
        grid = VGroup()
        cell_texts = VGroup()
        label_texts = VGroup()

        correct_color = GREEN_C
        wrong_color = RED_C
        color_map = [[correct_color, wrong_color], [wrong_color, correct_color]]

        for i in range(2):
            for j in range(2):
                val = CM[i][j]
                intensity = val / max_val
                color = interpolate_color(DARK_GREY, color_map[i][j], intensity * 0.8)

                sq = Square(side_length=cell_size)
                sq.set_fill(color=color, opacity=0.85)
                sq.set_stroke(WHITE, width=1.5)
                grid.add(sq)

                # Value + label
                val_txt = Text(str(val), font_size=NUMBER_SIZE + 4, color=TEXT_COLOR, weight=BOLD)
                lbl_txt = Text(CM_LABELS[i][j], font_size=LABEL_SIZE, color=GREY_A)
                lbl_txt.next_to(val_txt, DOWN, buff=0.15)
                cell_group = VGroup(val_txt, lbl_txt)
                cell_texts.add(cell_group)

        grid.arrange_in_grid(2, 2, buff=0.1)
        for k, txt in enumerate(cell_texts):
            txt.move_to(grid[k])

        # Row headers (left)
        row_headers = VGroup(
            Text("Actual\nLow", font_size=LABEL_SIZE, color=TEXT_COLOR),
            Text("Actual\nHigh", font_size=LABEL_SIZE, color=TEXT_COLOR),
        )
        row_headers[0].next_to(grid[0], LEFT, buff=0.4)
        row_headers[1].next_to(grid[2], LEFT, buff=0.4)

        # Column headers (top)
        col_headers = VGroup(
            Text("Pred Low", font_size=LABEL_SIZE, color=TEXT_COLOR),
            Text("Pred High", font_size=LABEL_SIZE, color=TEXT_COLOR),
        )
        col_headers[0].next_to(grid[0], UP, buff=0.3)
        col_headers[1].next_to(grid[1], UP, buff=0.3)

        full = VGroup(grid, cell_texts, row_headers, col_headers)
        full.move_to(ORIGIN).shift(DOWN * 0.2)

        self.play(FadeIn(grid, lag_ratio=0.1), run_time=1.0)
        self.play(FadeIn(cell_texts), FadeIn(row_headers), FadeIn(col_headers), run_time=0.8)

        # Accuracy annotation
        acc_text = Text("Test Accuracy: 73.2%", font_size=NUMBER_SIZE, color=ACCENT, weight=BOLD)
        acc_text.to_edge(DOWN, buff=EDGE_BUFF + 0.2)
        prec_text = Text("Balanced precision and recall across both classes", font_size=LABEL_SIZE, color=TEXT_COLOR)
        prec_text.next_to(acc_text, UP, buff=0.2)

        self.play(FadeIn(acc_text), FadeIn(prec_text), run_time=0.6)

        self.wait(5)
        self.play(FadeOut(VGroup(title, full, acc_text, prec_text)), run_time=0.8)
