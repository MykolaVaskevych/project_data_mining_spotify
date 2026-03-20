"""Scene 15: Conclusion — key numbers appearing one by one, final message."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, BODY_SIZE, NUMBER_SIZE, LABEL_SIZE, EDGE_BUFF, PALETTE


class ConclusionScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Key Takeaways", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Key stats in a clean grid layout
        stats = [
            ("≤ 0.10", "Max popularity\ncorrelation", PALETTE[0]),
            ("0.16", "Best silhouette\nscore", PALETTE[1]),
            ("70.2%", "Classification\naccuracy", PALETTE[2]),
            ("0.267", "Best regression\nR²", PALETTE[3]),
            ("27%", "Variance\nexplained", ACCENT),
        ]

        stat_boxes = VGroup()
        for val, desc, color in stats:
            num = Text(val, font_size=NUMBER_SIZE + 6, color=color, weight=BOLD)
            lbl = Text(desc, font_size=LABEL_SIZE - 2, color=TEXT_COLOR, line_spacing=1.1)
            lbl.next_to(num, DOWN, buff=0.2)
            box = VGroup(num, lbl)
            stat_boxes.add(box)

        # First row: 3 stats, second row: 2 stats centered
        row1 = VGroup(stat_boxes[0], stat_boxes[1], stat_boxes[2]).arrange(RIGHT, buff=1.5)
        row2 = VGroup(stat_boxes[3], stat_boxes[4]).arrange(RIGHT, buff=2.0)
        all_stats = VGroup(row1, row2).arrange(DOWN, buff=0.8)
        all_stats.next_to(title, DOWN, buff=0.6)

        for box in stat_boxes:
            self.play(FadeIn(box, scale=1.15), run_time=0.5)

        self.wait(1)

        # Final message
        final = Text(
            "Music popularity is a human phenomenon\nthat audio analysis alone cannot decode.",
            font_size=BODY_SIZE - 2,
            color=ACCENT,
            line_spacing=1.3,
        )
        final.to_edge(DOWN, buff=EDGE_BUFF + 0.1)
        self.play(FadeIn(final, shift=UP * 0.3), run_time=1.0)

        self.wait(5)

        # Fade everything to black
        self.play(
            *[FadeOut(m) for m in stat_boxes],
            FadeOut(title), FadeOut(final),
            run_time=2,
        )
        self.wait(0.5)
