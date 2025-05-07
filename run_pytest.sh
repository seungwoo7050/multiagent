#!/usr/bin/env bash
# run_pytest.sh

# 인자 검증
if [ $# -ne 1 ]; then
  echo "Usage: $0 <test_path>"
  exit 1
fi

TEST_PATH="$1"
export RUN_BENCHMARKS=1

# PYTHONPATH 설정
export PYTHONPATH=.

# 1) 콘솔에 실패한 테스트와 트레이스백만 출력
# pytest -qq --disable-warnings -rN --maxfail=1 "$TEST_PATH"
# pytest -qq --disable-warnings -rN "$TEST_PATH" > ~/Desktop/pytestlog.txt

# 3) skipped reason
# pytest -q -r s "$TEST_PATH"  > ~/Desktop/pytestlog.txt

# 4)
# pytest -q --benchmark-only "$TEST_PATH" > ~/Desktop/pytestlog.txt

# 5) 로그 제거
# pytest -q --disable-warnings -p no:logging --tb=short "$TEST_PATH"

# 6) maxfail
pytest --disable-warnings "$TEST_PATH" # > ~/Desktop/pytestlog.txt
# pytest -qq --disable-warnings -rN --maxfail=1 "$TEST_PATH" > ~/Desktop/pytestlog.txt