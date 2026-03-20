#!/bin/bash
# Render all animation scenes in parallel at 1080p30 and combine into one video.
# Usage:
#   ./render.sh          # render 1080p + combine
#   ./render.sh -ql      # render 480p + combine (fast preview)
set -e

QUALITY="${1:--qh}"
JOBS="${2:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Scene files and their class names (order matters for concat)
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

mkdir -p parts/logs

# ── Step 1: Render all scenes in parallel ────────────────────────────────
echo "═══ Rendering ${#SCENES[@]} scenes ($QUALITY) with $JOBS parallel jobs ═══"
echo ""

render_scene() {
    local entry="$1"
    local FILE="${entry%%:*}"
    local CLASS="${entry##*:}"
    local LOG="parts/logs/${FILE%.py}.log"
    echo "▸ Starting $FILE..."
    if uv run manim "$QUALITY" --disable_caching "animations/$FILE" "$CLASS" --media_dir ./parts/ >"$LOG" 2>&1; then
        echo "  ✓ $FILE done"
    else
        echo "  ✗ $FILE FAILED (see $LOG)"
        return 1
    fi
}
export -f render_scene
export QUALITY

FAILED=0
# Run renders in parallel using background jobs, limited to $JOBS at a time
PIDS=()
IDX=0
for entry in "${SCENES[@]}"; do
    render_scene "$entry" &
    PIDS+=($!)
    IDX=$((IDX + 1))
    # Throttle: wait if we hit the job limit
    if [ "${#PIDS[@]}" -ge "$JOBS" ]; then
        wait "${PIDS[0]}" || FAILED=$((FAILED + 1))
        PIDS=("${PIDS[@]:1}")
    fi
done
# Wait for remaining jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED scene(s) failed. Check parts/logs/ for details."
    exit 1
fi
echo "All scenes rendered successfully."
echo ""

# ── Step 2: Build ffmpeg concat list ─────────────────────────────────────
echo "═══ Combining into final video ═══"

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

# ── Step 3: Concatenate with ffmpeg ──────────────────────────────────────
OUTPUT="$SCRIPT_DIR/final_video.mp4"

ffmpeg -y -f concat -safe 0 -i "$CONCAT_FILE" \
    -c:v libx264 -preset medium -crf 18 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT" 2>&1 | tail -3

echo ""
echo "═══ Done ═══"
echo "Output: $OUTPUT"
DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$OUTPUT" 2>/dev/null)
echo "Duration: ${DURATION}s"
