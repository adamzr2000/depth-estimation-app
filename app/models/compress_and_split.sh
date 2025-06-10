#!/bin/bash

CHUNK_SIZE="50M"  # Adjust as needed

for FOLDER in monocular_deployed monocular_deployed_gpu; do
  ARCHIVE="${FOLDER}.tar.gz"
  CHUNK_PREFIX="${FOLDER}_split_"

  echo "[*] Compressing $FOLDER..."
  tar -czvf "$ARCHIVE" "$FOLDER"

  echo "[*] Splitting $ARCHIVE into chunks of $CHUNK_SIZE..."
  split -b "$CHUNK_SIZE" "$ARCHIVE" "$CHUNK_PREFIX"

  echo "[*] Deleting original archive: $ARCHIVE"
  rm "$ARCHIVE"

  echo "[✓] Done with $FOLDER. Use 'cat ${CHUNK_PREFIX}* > $ARCHIVE' to recombine and extract."
done

