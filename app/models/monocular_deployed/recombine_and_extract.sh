#!/bin/bash

for FOLDER in lib_cpu lib_gpu; do
  ARCHIVE="${FOLDER}.tar.gz"
  CHUNK_PREFIX="${FOLDER}_split_"

  echo "[*] Recombining split files for $FOLDER into $ARCHIVE..."
  cat ${CHUNK_PREFIX}* > "$ARCHIVE"

  echo "[*] Extracting $ARCHIVE..."
  tar -xzvf "$ARCHIVE"

  echo "[*] Cleaning up split parts and archive for $FOLDER..."
  rm -f "$ARCHIVE"
  rm -f ${CHUNK_PREFIX}*
done

echo "[✓] All done. Folder 'monocular_deployed' is restored."

