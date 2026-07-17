#!/usr/bin/env bash
# Download and verify the released Beyond Images enriched datasets.
#
# Primary source : Zenodo (https://doi.org/10.5281/zenodo.14847095)
# Fallback mirror: GitHub Release asset (same file, same checksum)
#
# Usage:
#   bash scripts/prepare_data.sh            # download + verify + extract
#   bash scripts/prepare_data.sh --sample   # only extract the bundled sample
set -euo pipefail
cd "$(dirname "$0")/.."

ZENODO_URL="https://zenodo.org/records/14847095/files/img_text_summary.zip?download=1"
MIRROR_URL="https://github.com/pengyu-zhang/Beyond-Images/releases/download/v1.0/img_text_summary.zip"
FULL_MD5="e5464dcb76501c10b692afd309139a50"
SAMPLE_MD5="946951db0c58930d1108d50c8ab2d1e1"

md5_of() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    else
        # macOS
        md5 -q "$1"
    fi
}

extract_sample() {
    echo "-> Extracting bundled sample (MKG-W BLIP captions, 15k entities)"
    actual=$(md5_of data/sample/img_text_summary.zip)
    if [ "$actual" != "$SAMPLE_MD5" ]; then
        echo "ERROR: sample zip checksum mismatch ($actual)"; exit 1
    fi
    mkdir -p data/sample/extracted
    python - <<'EOF'
import zipfile
zipfile.ZipFile("data/sample/img_text_summary.zip").extractall("data/sample/extracted")
print("   extracted to data/sample/extracted/img_text_summary/")
EOF
}

if [ "${1:-}" = "--sample" ]; then
    extract_sample
    exit 0
fi

mkdir -p data/raw data/processed
TARGET="data/raw/img_text_summary.zip"

if [ -f "$TARGET" ] && [ "$(md5_of "$TARGET")" = "$FULL_MD5" ]; then
    echo "-> $TARGET already present and verified"
else
    echo "-> Downloading enriched datasets from Zenodo (~128 MB)"
    if ! curl -fL --retry 3 -o "$TARGET" "$ZENODO_URL"; then
        echo "-> Zenodo failed, trying GitHub Release mirror"
        curl -fL --retry 3 -o "$TARGET" "$MIRROR_URL"
    fi
    actual=$(md5_of "$TARGET")
    if [ "$actual" != "$FULL_MD5" ]; then
        echo "ERROR: checksum mismatch: expected $FULL_MD5, got $actual"; exit 1
    fi
    echo "-> Checksum OK"
fi

echo "-> Extracting to data/processed/"
python - <<'EOF'
import zipfile
zipfile.ZipFile("data/raw/img_text_summary.zip").extractall("data/processed")
print("   extracted to data/processed/img_text_summary/")
EOF

extract_sample
echo "Done. See data/README.md for the source MMKG datasets (MKG-W, MKG-Y, DB15K)."
