#!/usr/bin/env bash
# run_pytest.sh

# 인자 검증
if [ $# -ne 1 ]; then
  echo "Usage: $0 <test_path>"
  exit 1
fi

TEST_PATH="$1"

# PYTHONPATH 설정
export PYTHONPATH=.

# 1) 콘솔에 실패한 테스트와 트레이스백만 출력
pytest -qq --disable-warnings -rN "$TEST_PATH"

# 2) 같은 결과를 ~/Desktop/pytest_log.txt 파일로 저장
pytest -qq --disable-warnings -rN "$TEST_PATH" > ~/Desktop/pytest_log.txt
