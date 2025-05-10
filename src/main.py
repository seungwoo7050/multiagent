#!/usr/bin/env python3
# src/main.py - Redis 연결 테스트 스크립트

import asyncio
import os
import json
from datetime import datetime
import sys

# 경로 추가 (import 오류 방지)
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    # 설정 모듈 가져오기 시도
    from src.config.settings import get_settings
    from src.config.connections import setup_connection_pools, get_redis_async_connection
    settings_available = True
except ImportError as e:
    print(f"설정 모듈 가져오기 실패: {e}")
    settings_available = False

# 기본 설정
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_REDIS_PASSWORD = None

async def test_redis_connection():
    """Redis 연결을 테스트하고 상태를 보고합니다."""
    print("\n===== Redis 연결 테스트 =====")
    
    # 1. 환경 설정 확인
    print("\n----- 환경 설정 -----")
    redis_host = os.getenv("REDIS_HOST", DEFAULT_REDIS_HOST)
    redis_port = int(os.getenv("REDIS_PORT", DEFAULT_REDIS_PORT))
    redis_db = int(os.getenv("REDIS_DB", DEFAULT_REDIS_DB))
    redis_password = os.getenv("REDIS_PASSWORD", DEFAULT_REDIS_PASSWORD)
    
    print(f"Redis 호스트: {redis_host}")
    print(f"Redis 포트: {redis_port}")
    print(f"Redis DB: {redis_db}")
    print(f"Redis 비밀번호: {'설정됨' if redis_password else '없음'}")
    
    # 2. 애플리케이션 설정 확인 (사용 가능한 경우)
    if settings_available:
        print("\n----- 애플리케이션 설정 -----")
        try:
            settings = get_settings()
            print(f"설정 로드 성공: {settings.__class__.__name__}")
            
            # Settings 객체에서 Redis 설정 속성 찾기 시도
            redis_settings = {}
            for attr in dir(settings):
                if 'REDIS' in attr and not attr.startswith('__'):
                    redis_settings[attr] = getattr(settings, attr)
            
            if redis_settings:
                print("설정에서 찾은 Redis 관련 값:")
                for key, value in redis_settings.items():
                    print(f"  {key}: {value}")
            else:
                print("설정에서 Redis 관련 값을 찾을 수 없습니다.")
        except Exception as e:
            print(f"설정 로드 실패: {e}")
    
    # 3. 자체 Redis 연결 테스트
    print("\n----- 직접 Redis 연결 테스트 -----")
    try:
        import redis.asyncio as aioredis
        
        # Redis 클라이언트 생성
        redis_url = f"redis://{':'+redis_password+'@' if redis_password else ''}{redis_host}:{redis_port}/{redis_db}"
        print(f"연결 URL: {redis_url}")
        
        direct_redis = aioredis.from_url(redis_url)
        
        # PING 테스트
        ping_result = await direct_redis.ping()
        print(f"PING 테스트: {'성공' if ping_result else '실패'}")
        
        # 읽기/쓰기 테스트
        test_key = "redis_test_key"
        test_value = {
            "timestamp": datetime.now().isoformat(),
            "message": "Redis 연결 테스트"
        }
        
        # 쓰기 테스트
        set_result = await direct_redis.set(test_key, json.dumps(test_value))
        print(f"쓰기 테스트: {'성공' if set_result else '실패'}")
        
        # 읽기 테스트
        get_result = await direct_redis.get(test_key)
        if get_result:
            print(f"읽기 테스트: 성공 - {get_result.decode('utf-8')}")
        else:
            print("읽기 테스트: 실패 - 값을 읽을 수 없음")
        
        # 연결 종료
        await direct_redis.close()
        
    except ImportError:
        print("Redis 클라이언트 라이브러리를 가져올 수 없습니다. 'pip install redis' 실행 필요")
    except Exception as e:
        print(f"직접 Redis 연결 실패: {e}")
    
    # 4. 애플리케이션 Redis 풀 테스트 (사용 가능한 경우)
    if settings_available:
        print("\n----- 애플리케이션 Redis 풀 테스트 -----")
        try:
            # Redis 풀 초기화 시도
            print("Redis 연결 풀 초기화 시도...")
            await setup_connection_pools()
            print("Redis 연결 풀 초기화 성공")
            
            # Redis 연결 가져오기 시도
            print("Redis 연결 가져오기 시도...")
            redis = await get_redis_async_connection()
            print(f"Redis 연결 가져오기 성공: {redis}")
            
            # 앱 Redis로 PING 테스트
            ping_result = await redis.ping()
            print(f"PING 테스트: {'성공' if ping_result else '실패'}")
            
            # 앱 Redis로 읽기/쓰기 테스트
            app_test_key = "app_redis_test_key"
            app_test_value = {
                "timestamp": datetime.now().isoformat(),
                "source": "애플리케이션 Redis 풀"
            }
            
            # 쓰기 테스트
            set_result = await redis.set(app_test_key, json.dumps(app_test_value))
            print(f"쓰기 테스트: {'성공' if set_result else '실패'}")
            
            # 읽기 테스트
            get_result = await redis.get(app_test_key)
            if get_result:
                print(f"읽기 테스트: 성공 - {get_result.decode('utf-8')}")
            else:
                print("읽기 테스트: 실패 - 값을 읽을 수 없음")
                
            # Redis 풀 저장하는지 테스트
            test_key = f"memory:test-task-{datetime.now().timestamp()}:workflow_final_state"
            test_data = {
                "status": "completed",
                "task_id": f"test-task-{datetime.now().timestamp()}",
                "final_answer": "Redis 연결 테스트 성공"
            }
            
            print(f"\n메모리 저장소 테스트 키: {test_key}")
            set_result = await redis.set(test_key, json.dumps(test_data))
            print(f"메모리 저장 테스트: {'성공' if set_result else '실패'}")
            
            # 연결 종료
            await redis.close()
            print("Redis 연결 종료됨")
            
        except Exception as e:
            print(f"애플리케이션 Redis 풀 테스트 실패: {e}")
    
    print("\n===== Redis 연결 테스트 완료 =====")

if __name__ == "__main__":
    # 메인 함수 실행
    asyncio.run(test_redis_connection())