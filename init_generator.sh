# 모든 하위 디렉토리에 __init__.py 파일 생성
find src -type d -not -path "*/\.*" -exec touch {}/__init__.py \;