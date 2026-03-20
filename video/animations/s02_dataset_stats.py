"""Scene 2: Animated counters for dataset statistics."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import BG_COLOR, ACCENT, TEXT_COLOR, TITLE_SIZE, BODY_SIZE, NUMBER_SIZE, EDGE_BUFF, PALETTE


class DatasetStatsScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Dataset Overview", font_size=TITLE_SIZE, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        stats = [
            ("2,000", "Total Tracks", PALETTE[0]),
            ("1,960", "After Cleaning", PALETTE[1]),
            ("5", "Genres", PALETTE[2]),
            ("14", "Audio Features", PALETTE[3]),
            ("422", "Zero Popularity", PALETTE[5]),
        ]

        boxes = VGroup()
        for val, label, color in stats:
            num = Text(val, font_size=NUMBER_SIZE + 10, color=color, weight=BOLD)
            lbl = Text(label, font_size=BODY_SIZE - 6, color=TEXT_COLOR)
            lbl.next_to(num, DOWN, buff=0.2)
            box = VGroup(num, lbl)
            boxes.add(box)

        boxes.arrange(RIGHT, buff=1.0)
        boxes.next_to(title, DOWN, buff=1.2)

        # Scale down if too wide
        if boxes.width > 13:
            boxes.scale_to_fit_width(13)

        for box in boxes:
            self.play(FadeIn(box, shift=UP * 0.3), run_time=0.35)

        # Highlight the zero-popularity stat
        highlight = SurroundingRectangle(boxes[-1], color=PALETTE[5], buff=0.2, corner_radius=0.1)
        self.play(Create(highlight), run_time=0.5)

        self.wait(4)
        self.play(FadeOut(VGroup(title, boxes, highlight)), run_time=0.8)
