"""Scene 1: Title card with 3D rotating dot cluster background."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from common import BG_COLOR, PALETTE, ACCENT, TEXT_COLOR, TITLE_SIZE, SUBTITLE_SIZE, EDGE_BUFF, safe_add_fixed


class TitleScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES, zoom=0.85)

        # Background dot cluster — positioned BELOW center so title has clear space
        np.random.seed(42)
        dots = VGroup()
        for i in range(40):
            pos = np.random.randn(3) * 1.5
            pos[2] -= 1.0  # shift dots down
            d = Dot3D(point=pos, radius=0.06, color=PALETTE[i % len(PALETTE)])
            d.set_opacity(0.65)
            dots.add(d)

        self.play(FadeIn(dots, lag_ratio=0.03), run_time=1.5)
        self.begin_ambient_camera_rotation(rate=0.025)

        # Title text — fixed in frame, positioned at TOP (clear of dots)
        title = Text("CS4168 Data Mining", font_size=TITLE_SIZE + 4, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF + 0.2)

        subtitle = Text("Spotify Tracks Analysis", font_size=SUBTITLE_SIZE, color=TEXT_COLOR)
        subtitle.next_to(title, DOWN, buff=0.35)

        byline = Text("What makes a track popular?", font_size=22, color=GREY_A)
        byline.next_to(subtitle, DOWN, buff=0.4)

        safe_add_fixed(self, title, subtitle, byline)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.8)
        self.play(FadeIn(subtitle, shift=DOWN * 0.2), run_time=0.6)
        self.play(FadeIn(byline, shift=DOWN * 0.2), run_time=0.5)

        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.play(*[FadeOut(m) for m in [title, subtitle, byline, dots]], run_time=1)
