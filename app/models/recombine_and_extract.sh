#!/bin/bash

PYTHON_SCRIPT="download_keras_model.py"

for FOLDER in monocular_deployed monocular_deployed_gpu; do
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

# Ensure huggingface_hub is installed
echo "[*] Checking Python dependencies..."
python3 -m pip show huggingface_hub > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[*] huggingface_hub not found. Installing..."
    python3 -m pip install huggingface_hub
else
    echo "[✓] huggingface_hub already installed."
fi

# Run the Python download script
echo "[*] Downloading Keras model into ./monocular_keras..."
python3 "$PYTHON_SCRIPT"

echo "[✓] All done. Folder 'monocular_deployed' is restored and Keras model is downloaded."

