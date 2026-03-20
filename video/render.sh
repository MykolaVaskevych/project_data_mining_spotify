#!/bin/bash
# Render all scenes in parallel using GPU (OpenGL renderer) and combine into one video.
# Usage:
#   ./render.sh              # 1080p GPU render + combine
#   ./render.sh -ql          # 480p GPU render + combine (preview)
#   ./render.sh -qh 8        # 1080p, 8 parallel jobs
#   ./render.sh --cairo       # force CPU/Cairo renderer (fallback)
set -e

RENDERER="opengl"
QUALITY="-qh"
JOBS=""

# Parse args
for arg in "$@"; do
    case "$arg" in
        -ql|-qm|-qh|-qp|-qk) QUALITY="$arg" ;;
        --cairo) RENDERER="cairo" ;;
        [0-9]*) JOBS="$arg" ;;
    esac
done

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
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
case "$QUALITY" in
    -ql) RES_DIR="480p15" ;;
    -qm) RES_DIR="720p30" ;;
    -qh) RES_DIR="1080p60" ;;
    -qp) RES_DIR="1440p60" ;;
    -qk) RES_DIR="2160p60" ;;
    *)   RES_DIR="1080p60" ;;
esac

# OpenGL renderer outputs to different dir naming (1080p60 not 1080p30)
# but we also set frame_rate=30 in manim.cfg, so check both
RES_DIR_ALT="${RES_DIR/60/30}"

mkdir -p parts/logs

# Build renderer flags
if [ "$RENDERER" = "opengl" ]; then
    RENDER_FLAGS="--renderer=opengl --write_to_movie"
    echo "═══ GPU (OpenGL) rendering: ${#SCENES[@]} scenes ($QUALITY) with $JOBS parallel jobs ═══"
else
    RENDER_FLAGS=""
    echo "═══ CPU (Cairo) rendering: ${#SCENES[@]} scenes ($QUALITY) with $JOBS parallel jobs ═══"
fi
echo ""

render_scene() {
    local entry="$1"
    local FILE="${entry%%:*}"
    local CLASS="${entry##*:}"
    local LOG="parts/logs/${FILE%.py}.log"
    echo "▸ Starting $FILE..."
    if uv run manim "$QUALITY" $RENDER_FLAGS --disable_caching \
        "animations/$FILE" "$CLASS" --media_dir ./parts/ >"$LOG" 2>&1; then
        echo "  ✓ $FILE done"
    else
        echo "  ✗ $FILE FAILED (see $LOG)"
        return 1
    fi
}
export -f render_scene
export QUALITY RENDER_FLAGS

FAILED=0
PIDS=()
for entry in "${SCENES[@]}"; do
    render_scene "$entry" &
    PIDS+=($!)
    if [ "${#PIDS[@]}" -ge "$JOBS" ]; then
        wait "${PIDS[0]}" || FAILED=$((FAILED + 1))
        PIDS=("${PIDS[@]:1}")
    fi
done
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

echo ""
if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: $FAILED scene(s) failed. Check parts/logs/ for details."
    echo "Tip: try --cairo flag to fall back to CPU rendering."
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

    # Try both possible resolution dir names
    MP4_PATH="$SCRIPT_DIR/parts/videos/$SCENE_DIR/$RES_DIR/$CLASS.mp4"
    if [ ! -f "$MP4_PATH" ]; then
        MP4_PATH="$SCRIPT_DIR/parts/videos/$SCENE_DIR/$RES_DIR_ALT/$CLASS.mp4"
    fi
    if [ ! -f "$MP4_PATH" ]; then
        # Search for any mp4 in the scene dir
        MP4_PATH=$(find "$SCRIPT_DIR/parts/videos/$SCENE_DIR" -name "$CLASS.mp4" -not -path "*/partial_movie_files/*" | head -1)
    fi

    if [ -z "$MP4_PATH" ] || [ ! -f "$MP4_PATH" ]; then
        echo "ERROR: Missing render for $CLASS"
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
