#!/usr/bin/env bash
set -euxo pipefail

BASE_DIR="$(pwd)"
SRC_DIR="${BASE_DIR}/src"
DEST_DIR="${BASE_DIR}/src_minified"

echo "▶ Source directory : ${SRC_DIR}"
echo "▶ Destination dir  : ${DEST_DIR}"

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"

# ────────────────────────────────────────────────────────────
# find 와 while 은 반드시 같은 논리 라인에 파이프로 연결
find "${SRC_DIR}" -type f -print0 | while IFS= read -r -d '' src_file; do
  rel_path="${src_file#${SRC_DIR}/}"
  dest_file="${DEST_DIR}/${rel_path}"
  mkdir -p "$(dirname "${dest_file}")"

  if [[ "${src_file##*.}" == "py" ]]; then
    # 최대 압축: 주석·docstring·assert·debug 코드 제거, 전역 심볼 축약, shebang 제거
    python3 -m python_minifier \
      --remove-literal-statements \
      --remove-asserts \
      --remove-debug \
      --remove-class-attribute-annotations \
      --rename-globals \
      --no-preserve-shebang \
      --output "${dest_file}" \
      "${src_file}"
  else
    cp -p "${src_file}" "${dest_file}"
  fi
done
# ────────────────────────────────────────────────────────────

echo "✅ Minification complete: ${DEST_DIR}"
