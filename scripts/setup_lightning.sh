#!/usr/bin/env bash
# One-time setup for a Lightning AI Studio. Run it once; the Studio home is persistent, so it
# survives session restarts and you never pay for these minutes twice.
#
#   bash scripts/setup_lightning.sh
#
# Lightning does not allow creating virtualenvs inside a Studio, so this installs into the
# Studio's own environment. That is the supported pattern there, not laziness.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="/teamspace/studios/this_studio"
[ -d "$HOME_DIR" ] || HOME_DIR="$REPO"
WORK="$HOME_DIR/waxal-work"

echo "repo = $REPO"
echo "work = $WORK   (persistent — every stage reads and writes here)"
mkdir -p "$WORK"

# --- 1. python deps ------------------------------------------------------------------------
# torch first, from the CUDA index. A Studio may already have a good torch; only install if the
# import fails, because reinstalling torch is a multi-GB download billed against your credits.
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "torch present with CUDA: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "installing torch (cu121)..."
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
fi

pip install -q -r "$REPO/requirements-gpu.txt"

# --- 2. system deps ---------------------------------------------------------------------------
# Stage 3 builds KenLM from source into $WORK/kenlm. Doing the apt part here means the GPU
# machine isn't sitting idle during an apt-get when the meter is running.
#
# ffmpeg is here for soundfile/librosa's benefit on compressed inputs. It is no longer load-
# bearing for HF audio: requirements-gpu.txt pins datasets < 4 precisely so that decoding goes
# through soundfile rather than torchcodec. Read the note at the bottom of that file before
# changing either. Keeping ffmpeg costs one apt line and removes a class of codec surprises.
sudo apt-get -qq update
sudo apt-get -qq install -y build-essential cmake wget ffmpeg \
    libboost-system-dev libboost-thread-dev libboost-program-options-dev \
    libboost-test-dev libboost-filesystem-dev libeigen3-dev zlib1g-dev

# --- 3. sanity -------------------------------------------------------------------------------
python - <<'PY'
import torch
print(f"torch      {torch.__version__}")
print(f"cuda       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    bf16 = torch.cuda.is_bf16_supported()
    print(f"gpu        {name}  x{torch.cuda.device_count()}")
    print(f"bf16       {bf16}")
    if not bf16:
        print("           ^ T4 (Turing) has no bf16. Training falls back to fp16, which is "
              "fine but less stable. An L4 costs about the same per hour on Lightning and "
              "has bf16 — prefer it.")
import transformers, datasets
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")
PY

echo
echo "setup done. Next: python kaggle/00_build_lm_corpus.py   (CPU studio is enough)"
