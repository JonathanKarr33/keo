#!/usr/bin/env bash
# Platform detection, sourced by the serving and sweep scripts.
#
#   source "$(dirname "$0")/platform.sh"
#   keo_detect_platform
#
# Sets: KEO_OS, KEO_ARCH, KEO_ACCEL (mlx|cuda|cpu), KEO_SERVE_PYTHON, KEO_SERVE_CMD
#
# Exists because the serving path differs by platform in ways that used to surface as an
# unrelated-looking failure: vLLM needs CUDA, mlx-lm needs python >=3.10 while the repo venv is
# 3.9, and several GNU utilities the scripts reached for are simply absent on macOS. Each of those
# was diagnosed as a model or model-name problem before anyone checked the platform.

KEO_DEFAULT_LOCAL_MODEL="${KEO_DEFAULT_LOCAL_MODEL:-mlx-community/phi-4-4bit}"

# Python >=3.10 for mlx-lm, preferring the dedicated venv. Echoes nothing if none is usable.
_keo_find_serve_python() {
  local repo_root="$1" candidate
  for candidate in \
    "$repo_root/.venv-mlx/bin/python" \
    "$(command -v python3.13 || true)" \
    "$(command -v python3.12 || true)" \
    "$(command -v python3.11 || true)" \
    "$(command -v python3.10 || true)"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

keo_detect_platform() {
  local repo_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  KEO_OS="$(uname -s)"
  KEO_ARCH="$(uname -m)"
  KEO_SERVE_PYTHON=""
  KEO_SERVE_CMD=""

  if [[ "$KEO_OS" == "Darwin" && "$KEO_ARCH" == "arm64" ]]; then
    KEO_ACCEL="mlx"
    KEO_SERVE_PYTHON="$(_keo_find_serve_python "$repo_root" || true)"
    if [[ -n "$KEO_SERVE_PYTHON" ]]; then
      KEO_SERVE_CMD="$KEO_SERVE_PYTHON -m mlx_lm server --model $KEO_DEFAULT_LOCAL_MODEL --port ${PORT:-8000}"
    fi
  elif command -v nvidia-smi >/dev/null 2>&1; then
    KEO_ACCEL="cuda"
    KEO_SERVE_CMD="bash run_local_vllm.sh --serve-only"
  else
    KEO_ACCEL="cpu"
  fi
  export KEO_OS KEO_ARCH KEO_ACCEL KEO_SERVE_PYTHON KEO_SERVE_CMD
}

# Print how to start a server on *this* machine, rather than a command for someone else's.
keo_serve_hint() {
  case "$KEO_ACCEL" in
    mlx)
      if [[ -n "$KEO_SERVE_PYTHON" ]]; then
        echo "  $KEO_SERVE_CMD"
      else
        echo "  No python >=3.10 found, which mlx-lm requires (the repo venv is 3.9). Create one:"
        echo "    brew install python@3.12"
        echo "    python3.12 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx-lm"
      fi
      ;;
    cuda) echo "  $KEO_SERVE_CMD" ;;
    *)    echo "  No GPU or Apple Silicon detected; local generation will be impractically slow." ;;
  esac
}

# Abort with a platform-specific explanation unless a server is answering.
keo_require_local_server() {
  local base_url="${1:-http://127.0.0.1:${PORT:-8000}/v1}"
  if curl -s -m 10 "${base_url}/models" >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: no local model server answering at ${base_url}" >&2
  echo "Detected ${KEO_OS}/${KEO_ARCH} (accel=${KEO_ACCEL}). Start one with:" >&2
  keo_serve_hint >&2
  return 1
}

# Run a command detached. setsid does not exist on macOS, so a launch that relied on it died the
# instant its parent shell exited, leaving a log holding only "setsid: No such file or directory".
keo_detach() {
  local log="$1"; shift
  if command -v setsid >/dev/null 2>&1; then
    setsid nohup "$@" > "$log" 2>&1 < /dev/null &
  else
    nohup "$@" > "$log" 2>&1 < /dev/null &
  fi
  echo $!
}
