#!/bin/bash
# Общий выбор бэкенда для NVIDIA (source из *-nvidia.sh). НЕ ПРОВЕРЯЛОСЬ НА ЖЕЛЕЗЕ.
#
#   compute capability >= 7.5 (RTX 20 и новее) -> cuda: vLLM на CUDA/тензорных ядрах
#   compute capability <  7.5 (GTX 10xx, Pascal) -> ollama: llama.cpp, как раньше
# Переопределение: GLMOCR_BACKEND=cuda|ollama
# Задаёт: GPU_NAME VRAM_MIB FREE_MIB COMPUTE_CAP GLMOCR_BACKEND

GPU_NAME=""; VRAM_MIB=0; FREE_MIB=0; COMPUTE_CAP="0.0"
if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -r GPU_NAME VRAM_MIB FREE_MIB COMPUTE_CAP < <(
    nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits | head -1
  ) || true
  GPU_NAME="$(echo "$GPU_NAME" | sed 's/^ *//;s/ *$//')"
  VRAM_MIB="$(echo "$VRAM_MIB" | tr -cd '0-9')"; VRAM_MIB="${VRAM_MIB:-0}"
  FREE_MIB="$(echo "$FREE_MIB" | tr -cd '0-9')"; FREE_MIB="${FREE_MIB:-0}"
  COMPUTE_CAP="$(echo "$COMPUTE_CAP" | tr -cd '0-9.')"; COMPUTE_CAP="${COMPUTE_CAP:-0.0}"
fi

compute_at_least_75() {
  local major="${COMPUTE_CAP%%.*}" minor="${COMPUTE_CAP#*.}"
  minor="${minor%%.*}"
  [[ "$major" =~ ^[0-9]+$ ]] || return 1
  [[ "$minor" =~ ^[0-9]+$ ]] || minor=0
  (( major > 7 || (major == 7 && minor >= 5) ))
}

case "${GLMOCR_BACKEND:-auto}" in
  cuda|ollama) ;;
  auto|"")
    if compute_at_least_75; then GLMOCR_BACKEND=cuda; else GLMOCR_BACKEND=ollama; fi ;;
  *)
    echo "GLMOCR_BACKEND=$GLMOCR_BACKEND не понят. Допустимо: cuda, ollama." >&2
    return 2 2>/dev/null || exit 2 ;;
esac
if [[ "$GLMOCR_BACKEND" == cuda ]] && ! compute_at_least_75; then
  echo "Внимание: GLMOCR_BACKEND=cuda, но compute $COMPUTE_CAP < 7.5 — vLLM на этой карте, скорее всего, не запустится." >&2
fi
export GLMOCR_BACKEND
