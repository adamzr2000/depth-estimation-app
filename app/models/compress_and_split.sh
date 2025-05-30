#!/bin/bash

FOLDER="monocular_deployed"
ARCHIVE="monocular_deployed.tar.gz"
CHUNK_PREFIX="monocular_split_"
CHUNK_SIZE="50M"  # Adjust as needed

# Compress folder
echo "[*] Compressing $FOLDER..."
tar -czvf "$ARCHIVE" "$FOLDER"

# Split into parts
echo "[*] Splitting into chunks of $CHUNK_SIZE..."
split -b "$CHUNK_SIZE" "$ARCHIVE" "$CHUNK_PREFIX"

# Delete original archive
echo "[*] Deleting original archive: $ARCHIVE"
rm "$ARCHIVE"

echo "[✓] Done. Use 'cat ${CHUNK_PREFIX}* > $ARCHIVE' to recombine and extract."

