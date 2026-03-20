"""Scene 7: PCA scatter — 3D dots with ThreeDAxes, colored by cluster."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
import numpy as np
from common import (
    BG_COLOR, TEXT_COLOR, TITLE_SIZE, LABEL_SIZE,
    EDGE_BUFF, DEFAULT_PHI, DEFAULT_THETA,
    PALETTE, safe_add_fixed, safe_add_orient,
)


class PCAScatterScene(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=DEFAULT_PHI, theta=DEFAULT_THETA, zoom=0.75)

        title = Text("PCA Scatter — K-Means (k=5)", font_size=TITLE_SIZE, color=TEXT_COLOR, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        safe_add_fixed(self, title)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # 3D Axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-2, 2, 1],
            x_length=6, y_length=6, z_length=4,
            axis_config={"color": GREY, "stroke_width": 1.5},
        )
        axis_labels = axes.get_axis_labels(
            Text("PC1", font_size=18, color=TEXT_COLOR),
            Text("PC2", font_size=18, color=TEXT_COLOR),
            Text("PC3", font_size=18, color=TEXT_COLOR),
        )
        self.play(Create(axes), FadeIn(axis_labels), run_time=1.0)

        # Generate synthetic PCA-like scatter (5 overlapping clusters)
        np.random.seed(42)
        centers = [
            [-1.5, 1.0, 0.5], [1.0, -0.5, -0.3], [0.0, 0.0, 0.0],
            [-0.5, -1.5, 0.8], [1.5, 1.0, -0.5],
        ]
        n_per_cluster = 30

        cluster_groups = []
        for cluster_id, center in enumerate(centers):
            points = np.random.randn(n_per_cluster, 3) * 0.6 + np.array(center)
            dots = VGroup()
            for pt in points:
                d = Dot3D(
                    point=axes.c2p(*pt),
                    radius=0.04,
                    color=PALETTE[cluster_id],
                )
                d.set_opacity(0.75)
                dots.add(d)
            cluster_groups.append(dots)

        all_dots = VGroup(*cluster_groups)
        self.play(FadeIn(all_dots, lag_ratio=0.003), run_time=1.5)

        # Legend — fixed in frame, positioned at right edge
        legend = VGroup()
        for i in range(5):
            dot = Dot(radius=0.06, color=PALETTE[i])
            lbl = Text(f"Cluster {i}", font_size=LABEL_SIZE - 6, color=TEXT_COLOR)
            lbl.next_to(dot, RIGHT, buff=0.1)
            legend.add(VGroup(dot, lbl))
        legend.arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        legend.to_corner(DR, buff=EDGE_BUFF)
        safe_add_fixed(self, legend)
        self.play(FadeIn(legend), run_time=0.4)

        # Orbit camera to show 3D depth
        self.begin_ambient_camera_rotation(rate=0.04)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        self.move_camera(phi=45 * DEGREES, theta=20 * DEGREES, run_time=2)
        self.wait(1)

        self.play(FadeOut(VGroup(all_dots, axes, axis_labels, legend, title)), run_time=0.8)
