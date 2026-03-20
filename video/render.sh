#!/bin/bash
# Render all animation scenes at 1080p30 and combine into one video.
# Usage:
#   ./render.sh          # render 1080p + combine
#   ./render.sh -ql      # render 480p + combine (fast preview)
set -e

QUALITY="${1:--qh}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Scene files and their class names (order matters)
SCENES=(
    "s01_title.py:TitleScene"
    "s02_dataset_stats.py:DatasetStatsScene"
    "s03_genre_bars.py:GenreBarsScene"
    "s04_correlation_grid.py:CorrelationGridScene"
    "s05_popularity_flat.py:PopularityFlatScene"
    "s06_silhouette_bars.py:SilhouetteBarsScene"
    "s07_pca_scatter.py:PCAScatterScene"
    "s08_kmeans_genre.py:KMeansGenreScene"
    "s09_classification_bars.py:ClassificationBarsScene"
    "s10_confusion_3d.py:ConfusionMatrixScene"
    "s11_feat_imp_clf.py:FeatureImportanceClfScene"
    "s12_regression_bars.py:RegressionBarsScene"
    "s13_ridge_disaster.py:RidgeDisasterScene"
    "s14_decisions.py:DecisionsScene"
    "s15_conclusion.py:ConclusionScene"
)

# Determine resolution dir name from quality flag
if [ "$QUALITY" = "-ql" ]; then
    RES_DIR="480p15"
else
    RES_DIR="1080p30"
fi

# ── Step 1: Render all scenes ────────────────────────────────────────────
echo "═══ Rendering all scenes ($QUALITY) ═══"
echo ""

FAILED=0
for entry in "${SCENES[@]}"; do
    FILE="${entry%%:*}"
    CLASS="${entry##*:}"
    echo "▸ Rendering $FILE ($CLASS)..."
    if uv run manim "$QUALITY" --disable_caching "animations/$FILE" "$CLASS" --media_dir ./parts/ 2>&1 | tail -1; then
        echo "  ✓ Done"
    else
        echo "  ✗ FAILED: $FILE"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED scene(s) failed to render. Fix errors before combining."
    exit 1
fi

# ── Step 2: Build ffmpeg concat list ─────────────────────────────────────
echo "═══ Combining into final video ═══"
echo ""

CONCAT_FILE="$SCRIPT_DIR/parts/concat.txt"
> "$CONCAT_FILE"

for entry in "${SCENES[@]}"; do
    FILE="${entry%%:*}"
    CLASS="${entry##*:}"
    SCENE_DIR="${FILE%.py}"
    MP4_PATH="$SCRIPT_DIR/parts/videos/$SCENE_DIR/$RES_DIR/$CLASS.mp4"

    if [ ! -f "$MP4_PATH" ]; then
        echo "ERROR: Missing $MP4_PATH"
        exit 1
    fi
    echo "file '$MP4_PATH'" >> "$CONCAT_FILE"
done

echo "Concat list:"
cat "$CONCAT_FILE"
echo ""

# ── Step 3: Concatenate with ffmpeg ──────────────────────────────────────
OUTPUT="$SCRIPT_DIR/final_video.mp4"

ffmpeg -y -f concat -safe 0 -i "$CONCAT_FILE" \
    -c:v libx264 -preset medium -crf 18 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT" 2>&1 | tail -5

echo ""
echo "═══ Done ═══"
echo "Output: $OUTPUT"
DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$OUTPUT" 2>/dev/null)
echo "Duration: ${DURATION}s"
