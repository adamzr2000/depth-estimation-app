#!/bin/bash

ARCHIVE="monocular_deployed.tar.gz"
CHUNK_PREFIX="monocular_split_"
PYTHON_SCRIPT="download_keras_model.py"

# Recombine split parts
echo "[*] Recombining split files into $ARCHIVE..."
cat ${CHUNK_PREFIX}* > "$ARCHIVE"

# Extract the archive
echo "[*] Extracting $ARCHIVE..."
tar -xzvf "$ARCHIVE"

# Clean up split parts and tar.gz
echo "[*] Cleaning up split parts and archive..."
rm -f "$ARCHIVE"
rm -f ${CHUNK_PREFIX}*

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

