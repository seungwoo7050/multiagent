# src/api/routes/metrics.py

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

# 프로젝트 루트 경로 설정 (app.py와 동일하게)
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.logger import get_logger
from src.config.metrics import get_metrics_manager

# Prometheus 클라이언트 라이브러리 import 시도
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    prometheus_client_available = True
except ImportError:
    prometheus_client_available = False
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain" # Fallback
    REGISTRY = None
    logger = get_logger(__name__)
    logger.error("prometheus-client library not installed. Metrics endpoint will not function.")
    logger.error("Please install it using: pip install prometheus-client")


logger = get_logger(__name__)

# APIRouter 인스턴스 생성
# metrics 엔드포인트는 보통 API prefix 없이 루트에 위치하므로 prefix 설정 안 함
router = APIRouter(
    tags=["System Metrics"]
)

# /metrics (GET): Prometheus 형식 메트릭 반환
@router.get(
    "/metrics",
    response_class=PlainTextResponse, # 응답 형식을 PlainText로 지정
    summary="Get System Metrics",
    description="Exposes system metrics in Prometheus format.",
    responses={
        200: {
            "description": "Prometheus metrics",
            "content": {
                "text/plain; version=0.0.4; charset=utf-8": {}
            }
        },
        503: {"description": "Metrics collection is disabled or prometheus-client library is unavailable"}
    }
)
async def get_metrics():
    """
    Prometheus가 수집할 수 있도록 시스템 메트릭을 텍스트 형식으로 반환합니다.
    `prometheus-client` 라이브러리가 설치되어 있어야 합니다.
    """
    metrics_manager = get_metrics_manager()
    if not metrics_manager.enabled or not prometheus_client_available:
        logger.warning("Metrics requested but metrics collection is disabled or prometheus-client is unavailable.")
        return PlainTextResponse("Metrics collection is disabled or unavailable.", status_code=503)

    logger.debug("Generating Prometheus metrics")
    try:
        # generate_latest 함수를 호출하여 현재 메트릭 데이터 생성
        metrics_data = generate_latest(REGISTRY)
        return PlainTextResponse(metrics_data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.exception("Error generating Prometheus metrics")
        return PlainTextResponse(f"Error generating metrics: {str(e)}", status_code=500)