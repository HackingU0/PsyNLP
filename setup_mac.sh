#!/bin/zsh
#!/bin/zsh
# PsyNLP macOS deployment script
set -euo pipefail

rm -rf nlp_models/GGUFS
rm -rf bert-emotion
rm -rf deberta-illness
# 0. check git and Homebrew
if ! command -v git &>/dev/null; then
  echo "[INFO] Git is not installed. Triggering Xcode command line tools installer..."
  xcode-select --install || true
  echo "[INFO] If the installer opened, re-run this script after installation finishes."
fi

if ! command -v brew &>/dev/null; then
  echo "[INFO] Homebrew is not installed. Attempting automatic installation..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 1. Check Python (require == 3.12)
if ! command -v python3 &>/dev/null; then
  echo "[ERROR] No Python3 detected. Installing python@3.12 via Homebrew..."
  brew install python@3.12
fi

PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
  echo "[ERROR] Detected Python version ${PY_MAJOR}.${PY_MINOR}. Please use Python 3.12."
  exit 1
fi

# 2. Ensure pip is available
if ! command -v pip3 &>/dev/null; then
  echo "[INFO] pip3 not found. Bootstrapping pip..."
  python3 -m ensurepip --upgrade
fi

# 3. Check CMake
if ! command -v cmake &>/dev/null; then
  echo "[WARNING] No CMake detected. Installing cmake via Homebrew..."
  brew install cmake
fi

#!/bin/zsh
# PsyNLP macOS deployment script
set -euo pipefail

# 0. check git and Homebrew
if ! command -v git &>/dev/null; then
  echo "[INFO] Git is not installed. Triggering Xcode command line tools installer..."
  xcode-select --install || true
  echo "[INFO] If the installer opened, re-run this script after installation finishes."
fi

if ! command -v brew &>/dev/null; then
  echo "[INFO] Homebrew is not installed. Attempting automatic installation..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 1. Check Python (require >= 3.12)
if ! command -v python3 &>/dev/null; then
  echo "[ERROR] No Python3 detected. Installing python@3.12 via Homebrew..."
  brew install python@3.12
fi

PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
  echo "[ERROR] Detected Python version ${PY_MAJOR}.${PY_MINOR}. Please use Python >= 3.12."
  exit 1
fi

# 2. Ensure pip is available
if ! command -v pip3 &>/dev/null; then
  echo "[INFO] pip3 not found. Bootstrapping pip..."
  python3 -m ensurepip --upgrade
fi

# 3. Check CMake
if ! command -v cmake &>/dev/null; then
  echo "[WARNING] No CMake detected. Installing cmake via Homebrew..."
  brew install cmake
fi

# 4. Create virtual environment
if [ ! -d "venv" ]; then
  echo "[INFO] Creating Python virtual environment venv..."
  python3 -m venv venv
fi
source venv/bin/activate

# 5. Upgrade pip/tooling inside venv
echo "[INFO] Upgrading pip, setuptools and wheel inside venv..."
python3 -m pip install --upgrade pip setuptools wheel

# 6. Install requirements.txt dependencies
if [ -f "requirements.txt" ]; then
  echo "[INFO] Installing requirements.txt dependencies..."
  python3 -m pip install -r requirements.txt
else
  echo "[WARNING] requirements.txt not found, skipping dependency installation."
fi

# 7. Install llama-cpp-python (Metal acceleration)
echo "[INFO] Installing llama-cpp-python (Metal acceleration)..."
env CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 python3 -m pip install llama-cpp-python --force-reinstall --no-binary :all: --no-cache-dir

# 8. Model Download
echo "[INFO] Downloading model files and optional resources..."
python3 -m spacy download en_core_web_sm || true

mkdir -p nlp_models/GGUFS
cd nlp_models/GGUFS

download() {
  local url="$1"; local out="$2"
  if [ -f "$out" ]; then
    echo "[INFO] $out already exists, skipping."
    return 0
  fi
  if command -v wget &>/dev/null; then
    wget -O "$out" "$url"
  else
    curl -L -o "$out" "$url"
  fi
}

download "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_0.gguf" "Qwen3-4B-Instruct-2507-Q4_0.gguf"
download "https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf" "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

cd ../

# clone or update other model folders
if [ -d "bert-emotion" ]; then
  echo "[INFO] bert-emotion already exists, pulling latest..."
  (cd bert-emotion && git pull --ff-only) || true
else
  git clone https://huggingface.co/boltuix/bert-emotion
fi

if [ -d "deberta-illness" ]; then
  echo "[INFO] deberta-illness already exists, pulling latest..."
  (cd deberta-illness && git pull --ff-only) || true
else
  git clone https://huggingface.co/boltuix/deberta-illness
fi

# 9. Completion message
echo "[SUCCESS] Deployment complete! Please use 'source venv/bin/activate' to activate the environment."
