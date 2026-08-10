#!/bin/bash
set -euo pipefail

PROJECT_DIR="/Users/dnallusamy/Downloads/Lip_Reading_Viseme-main"
DOWNLOADS="/Users/dnallusamy/Downloads"
VIDEO_DIR="$PROJECT_DIR/data/videos"
TMP_ROOT="$PROJECT_DIR/.unzip_tmp"

mkdir -p "$VIDEO_DIR" "$TMP_ROOT"

for zip_path in "$DOWNLOADS"/s*.mpg_vcd.zip; do
    subject=$(basename "$zip_path" .mpg_vcd.zip)
    echo "=== Processing $subject ($zip_path) ==="

    dest_dir="$VIDEO_DIR/$subject"
    mkdir -p "$dest_dir"

    tmp_dir="$TMP_ROOT/$subject"
    rm -rf "$tmp_dir"
    mkdir -p "$tmp_dir"

    unzip -q "$zip_path" -d "$tmp_dir"

    # Move all .mpg/.mp4 video files found anywhere in the extracted tree
    find "$tmp_dir" -type f \( -iname "*.mpg" -o -iname "*.mp4" \) -exec mv -n {} "$dest_dir"/ \;

    video_count=$(find "$dest_dir" -type f \( -iname "*.mpg" -o -iname "*.mp4" \) | wc -l | tr -d ' ')
    echo "$subject: $video_count videos moved to $dest_dir"

    rm -rf "$tmp_dir"
    rm -f "$zip_path"

    echo "Deleted $zip_path"
    df -h "$DOWNLOADS" | tail -1
    echo
done

rm -rf "$TMP_ROOT"
echo "All done."
