#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ASSETS_DIR="${1:-$PROJECT_ROOT/robocasa/models/assets}"
OUTPUT_TAR="${2:-$PROJECT_ROOT/robocasa/models/assets.tar}"
DELETE_SOURCE="${DELETE_SOURCE_ASSETS:-0}"

if [ ! -d "$ASSETS_DIR" ]; then
    echo "Error: assets directory not found: $ASSETS_DIR"
    echo "Usage: bash scripts/pack_assets.sh [ASSETS_DIR] [OUTPUT_TAR]"
    exit 1
fi

ASSETS_PARENT="$(dirname "$ASSETS_DIR")"
ASSETS_BASENAME="$(basename "$ASSETS_DIR")"

mkdir -p "$(dirname "$OUTPUT_TAR")"

echo "Packing assets..."
echo "  source: $ASSETS_DIR"
echo "  output: $OUTPUT_TAR"

tar -cf "$OUTPUT_TAR" -C "$ASSETS_PARENT" "$ASSETS_BASENAME"

echo "Done: $OUTPUT_TAR"

after_size="$(du -sh "$OUTPUT_TAR" | awk '{print $1}')"
echo "Archive size: $after_size"

if [ "$DELETE_SOURCE" = "1" ]; then
    echo "DELETE_SOURCE_ASSETS=1 detected, deleting source assets directory..."
    rm -rf "$ASSETS_DIR"
    echo "Deleted: $ASSETS_DIR"
fi
