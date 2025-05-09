#!/usr/bin/env bash
# run_pytest.sh

# 인자 검증
if [ $# -ne 1 ]; then
  echo "Usage: $0 <test_path>"
  exit 1
fi

TEST_PATH="$1"
export RUN_BENCHMARKS=1
export PYTHONPATH=.

# 테스트 옵션 1
# pytest -qq --disable-warnings -rN "$TEST_PATH"
# pytest -qq --disable-warnings -rN "$TEST_PATH" > ~/Desktop/pytestlog.txt

# 테스트 옵션
pytest -qq --disable-warnings -v
pytest -qq --disable-warnings -v --maxfail=1
pytest -qq --disable-warnings -rN -p no:logging
