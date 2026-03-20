"""Scene 8: K-Means cluster vs genre — grouped BarChart."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from manim import *
from common import (
    BG_COLOR, TEXT_COLOR, ACCENT, TITLE_SIZE, LABEL_SIZE, EDGE_BUFF,
    GENRE_COLORS, GENRE_LIST,
)

# Approximate cluster-genre counts (k=5): hip-hop concentrates in C2
CLUSTER_DATA = {
    "C0": {"pop": 120, "indie-pop": 95, "synth-pop": 50, "hip-hop": 30, "r-n-b": 60},
    "C1": {"pop": 100, "indie-pop": 110, "synth-pop": 80, "hip-hop": 40, "r-n-b": 55},
    "C2": {"pop": 90, "indie-pop": 80, "synth-pop": 60, "hip-hop": 144, "r-n-b": 50},
    "C3": {"pop": 100, "indie-pop": 102, "synth-pop": 130, "hip-hop": 38, "r-n-b": 62},
    "C4": {"pop": 80, "indie-pop": 100, "synth-pop": 79, "hip-hop": 40, "r-n-b": 65},
}


class KMeansGenreScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Cluster vs Genre Alignment (k=5)", font_size=TITLE_SIZE - 2, color=ACCENT, weight=BOLD)
        title.to_edge(UP, buff=EDGE_BUFF)
        self.play(FadeIn(title, shift=DOWN * 0.3), run_time=0.5)

        # Build a Table-like heatmap: rows=genres, cols=clusters
        clusters = list(CLUSTER_DATA.keys())
        n_rows = len(GENRE_LIST)
        n_cols = len(clusters)
        cell_size = 0.75
        max_val = 144

        grid = VGroup()
        val_texts = VGroup()

        for i, genre in enumerate(GENRE_LIST):
            for j, cluster in enumerate(clusters):
                val = CLUSTER_DATA[cluster][genre]
                intensity = val / max_val
                color = interpolate_color(BLACK, ManimColor(GENRE_COLORS[genre]), intensity)

                sq = Square(side_length=cell_size)
                sq.set_fill(color=color, opacity=0.9)
                sq.set_stroke(WHITE, width=0.5)
                grid.add(sq)

                txt = Text(str(val), font_size=14, color=WHITE if intensity > 0.3 else GREY)
                val_texts.add(txt)

        grid.arrange_in_grid(n_rows, n_cols, buff=0.05)
        for k, txt in enumerate(val_texts):
            txt.move_to(grid[k])

        # Row labels (genre names)
        row_labels = VGroup()
        for i, genre in enumerate(GENRE_LIST):
            lbl = Text(genre, font_size=LABEL_SIZE - 4, color=GENRE_COLORS[genre])
            lbl.next_to(grid[i * n_cols], LEFT, buff=0.25)
            row_labels.add(lbl)

        # Column labels
        col_labels = VGroup()
        for j, cluster in enumerate(clusters):
            lbl = Text(cluster, font_size=LABEL_SIZE - 2, color=TEXT_COLOR)
            lbl.next_to(grid[j], UP, buff=0.25)
            col_labels.add(lbl)

        full = VGroup(grid, val_texts, row_labels, col_labels)
        full.move_to(ORIGIN).shift(DOWN * 0.2)

        self.play(FadeIn(grid, lag_ratio=0.01), run_time=1.0)
        self.play(FadeIn(val_texts), FadeIn(row_labels), FadeIn(col_labels), run_time=0.8)

        # Highlight hip-hop in C2 (row 3, col 2 = index 3*5+2=17)
        hh_idx = 3 * n_cols + 2
        highlight = SurroundingRectangle(grid[hh_idx], color=YELLOW, buff=0.03, stroke_width=3)
        note = Text("Hip-hop concentrates in C2 (144/292)", font_size=LABEL_SIZE, color=ACCENT)
        note.to_edge(DOWN, buff=EDGE_BUFF)
        self.play(Create(highlight), FadeIn(note), run_time=0.6)

        self.wait(5)
        self.play(FadeOut(VGroup(title, full, highlight, note)), run_time=0.8)
