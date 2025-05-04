# src/api/routes/config.py

import os
# 프로젝트 루트 경로 설정 (app.py와 동일하게)
import sys
from typing import Any, Dict, Set

from fastapi import APIRouter, Depends, HTTPException, status

from src.config.logger import get_logger
from src.config.settings import Settings, get_settings

logger = get_logger(__name__)

# APIRouter 인스턴스 생성
router = APIRouter(
    prefix="/config",
    tags=["System Configuration"]
)

# 민감 정보 필드 목록 (환경 변수 이름 기준 또는 Settings 모델 필드 이름 기준)
# 실제 프로젝트에 맞게 수정/추가해야 합니다.
SENSITIVE_FIELDS: Set[str] = {
    "REDIS_PASSWORD",
    "Google Search_API_KEY",
    "Google Search_ENGINE_ID",
    # LLM_PROVIDERS_CONFIG 내부의 api_key 필드도 제외해야 함
}

# Settings 모델 필드 이름 기준 민감 정보 (소문자)
SENSITIVE_MODEL_FIELDS: Set[str] = {
    "redis_password",
    "Google Search_api_key",
    "Google Search_engine_id",
}

# 의존성 주입 함수
def get_settings_dependency() -> Settings:
    # get_settings()는 이미 캐싱되므로 여기서는 간단히 호출
    return get_settings()

# /config (GET): 현재 시스템 설정 반환 (민감 정보 제외)
@router.get(
    "",
    # response_model을 명시하기보다, 필터링된 dict를 직접 반환
    # response_model=Dict[str, Any], # 필요하다면 정의 가능
    summary="Get System Configuration",
    description="Retrieves the current system configuration settings, excluding sensitive information like API keys and passwords."
)
async def get_system_configuration(
    settings: Settings = Depends(get_settings_dependency)
):
    """
    현재 활성화된 시스템 설정을 반환합니다.
    API 키, 비밀번호 등 민감한 정보는 `***REDACTED***`로 마스킹 처리됩니다.
    """
    logger.info("Request received to get system configuration")
    try:
        # Safely serialize to handle complex objects
        from src.utils.serialization import serialize_to_json, deserialize_from_json
        
        # First get dict representation
        config_dict = settings.model_dump(mode='json')
        
        # Process and filter with serialization for complex objects
        filtered_config = {}
        for key, value in config_dict.items():
            if key.lower() in SENSITIVE_MODEL_FIELDS:
                filtered_config[key] = "***REDACTED***"
            elif key == "llm_providers_config" and isinstance(value, dict):
                # Serialize this section to handle nested complex objects
                serialized = serialize_to_json(value)
                providers_dict = deserialize_from_json(serialized)
                
                # Apply masking for API keys
                provider_configs_filtered = {}
                for provider, provider_config in providers_dict.items():
                    if isinstance(provider_config, dict):
                        provider_configs_filtered[provider] = {
                            k: ("***REDACTED***" if k == "api_key" else v)
                            for k, v in provider_config.items()
                        }
                    else:
                        provider_configs_filtered[provider] = provider_config
                filtered_config[key] = provider_configs_filtered
            else:
                # Serialize each value individually to handle potential complex objects
                serialized = serialize_to_json(value)
                filtered_config[key] = deserialize_from_json(serialized)
                
        return filtered_config

    except Exception as e:
        logger.exception("Error retrieving system configuration")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system configuration: {str(e)}"
        )