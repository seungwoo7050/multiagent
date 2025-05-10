# src/utils/telemetry.py
import logging
from typing import Optional, List

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION # <--- 표준 속성 이름 사용
# 테스트 환경을 위한 InMemorySpanExporter 추가 (선택적이지만 유용)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_tracer_provider: Optional[TracerProvider] = None
_is_telemetry_setup_attempted: bool = False # <--- 추가: 설정 시도 여부 플래그

# 테스트용 InMemorySpanExporter (선택 사항)
_test_in_memory_exporter: Optional[InMemorySpanExporter] = None

def setup_telemetry(force_setup: bool = False, for_testing: bool = False) -> None:
    global _tracer_provider, _is_telemetry_setup_attempted, _test_in_memory_exporter
    if not force_setup and _is_telemetry_setup_attempted:
        logger.debug("Telemetry already initialised – skipping.")
        return
    
    # 테스트 모드에서는 항상 새로 설정
    if for_testing:
        # 이전 exporter 초기화
        if _test_in_memory_exporter:
            _test_in_memory_exporter.clear()
        else:
            _test_in_memory_exporter = InMemorySpanExporter()
            
        # 테스트용 provider 생성
        resource = Resource.create({
            SERVICE_NAME: "test-service",
            SERVICE_VERSION: "test-version"
        })
        
        provider = TracerProvider(resource=resource)
        processor = SimpleSpanProcessor(_test_in_memory_exporter)
        provider.add_span_processor(processor)
        
        # 중요: 전역 provider로 설정하여 모든 트레이서가 사용하도록 함
        trace.set_tracer_provider(provider)
        _tracer_provider = provider
        _is_telemetry_setup_attempted = True
        logger.info("OpenTelemetry configured with InMemorySpanExporter for testing.")
        return

    try:
        # Resource 표준 속성 이름 사용
        resource_attributes = {
            SERVICE_NAME: settings.APP_NAME,
            SERVICE_VERSION: settings.APP_VERSION,
            "environment": settings.ENVIRONMENT, # 환경 정보도 추가하면 유용
        }
        resource = Resource.create(resource_attributes)

        provider = TracerProvider(resource=resource)
        
        span_processor: Optional[SpanProcessor] = None

        if for_testing:
            # 테스트 환경에서는 InMemorySpanExporter 또는 ConsoleSpanExporter 사용
            _test_in_memory_exporter = InMemorySpanExporter()
            # 테스트에서는 즉시 Span을 확인해야 하므로 SimpleSpanProcessor 사용 권장
            span_processor = SimpleSpanProcessor(_test_in_memory_exporter)
            logger.info("OpenTelemetry configured with InMemorySpanExporter for testing.")
        else:
            otlp_endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT
            if otlp_endpoint:
                # insecure=True는 settings.ENVIRONMENT == "development" 일 때만 고려
                insecure_otlp = settings.ENVIRONMENT == "development"
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=insecure_otlp)
                span_processor = BatchSpanProcessor(otlp_exporter)
                logger.info(f"OpenTelemetry configured with OTLP exporter to: {otlp_endpoint} (insecure: {insecure_otlp})")
            else:
                console_exporter = ConsoleSpanExporter()
                span_processor = BatchSpanProcessor(console_exporter) # 또는 SimpleSpanProcessor
                logger.info("OpenTelemetry configured with ConsoleSpanExporter as OTEL_EXPORTER_OTLP_ENDPOINT is not set.")

        if span_processor:
            provider.add_span_processor(span_processor)
        else:
            # 이 경우는 거의 없지만, 만약 span_processor가 None이면 로깅
            logger.warning("No span processor was configured for OpenTelemetry.")


        # set_tracer_provider는 provider 객체가 완전히 설정된 후에 호출
        trace.set_tracer_provider(provider)
        _tracer_provider = provider # 성공적으로 설정된 경우에만 할당

        # LangChain/LangSmith 통합 (기존 로직 유지)
        if not for_testing and settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
            logger.info("LangSmith tracing appears to be enabled via environment variables.")
            # from langchain_core.tracers.context import tracing_v2_enabled
            # if not tracing_v2_enabled():
            #    logger.warning("LangSmith tracing enabled in settings, but LangChain context shows it as disabled.")
        
        logger.info(f"OpenTelemetry TracerProvider setup complete. Current global provider: {trace.get_tracer_provider()}")

    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry TracerProvider: {e}", exc_info=True)
        _tracer_provider = None # 실패 시 None으로 명확히 설정
        # 실패 시, NoOpTracerProvider가 전역으로 설정될 수 있도록 여기서 set_tracer_provider 호출하지 않음
        # 또는 명시적으로 NoOpTracerProvider로 설정할 수도 있음
        # trace.set_tracer_provider(trace.NoOpTracerProvider())


def get_tracer(name: str) -> trace.Tracer:
    """지정된 이름으로 OpenTelemetry Tracer를 가져옵니다."""
    global _tracer_provider, _is_telemetry_setup_attempted

    # setup_telemetry가 시도되지 않았다면 (예: 테스트 환경에서 명시적 호출 누락)
    # 여기서 기본 설정을 시도하거나 경고 후 NoOpTracer 반환
    if not _is_telemetry_setup_attempted:
        logger.warning(
            "OpenTelemetry setup not attempted yet. Attempting basic setup with ConsoleExporter now. "
            "It's recommended to call setup_telemetry() explicitly at application/test start."
        )
        # 테스트가 아닌 일반 실행 중이라면 for_testing=False
        # 다만 이 경우는 app.py의 lifespan에서 이미 호출되었어야 함
        setup_telemetry(for_testing=False) # 또는 True로 하여 InMemory 사용 유도 가능

    # _tracer_provider가 여전히 None이면 (설정 실패 또는 의도적으로 NoOp 사용)
    # 전역 프로바이더에서 Tracer를 가져옴 (NoOpTracerProvider가 기본)
    # 이 경우 NoOpTracer가 반환되어 NonRecordingSpan을 생성할 수 있음
    # 테스트 실패 시나리오에서 NonRecordingSpan이 나온다면, setup_telemetry(for_testing=True)가
    # 테스트 시작 시점에 제대로 호출되었는지 확인하는 것이 중요함.
    current_global_provider = trace.get_tracer_provider()
    if _tracer_provider is None or not isinstance(current_global_provider, TracerProvider) or isinstance(current_global_provider, trace.NoOpTracerProvider):
        if _tracer_provider is None:
             logger.debug(f"Local _tracer_provider is None. Global provider is {type(current_global_provider)}. Returning tracer from global provider.")
        else: # _tracer_provider는 설정되었으나, 전역이 NoOp인 경우 (일관성 문제)
             logger.warning(f"Local _tracer_provider is set, but global provider is {type(current_global_provider)}. This might indicate an issue. Using global provider's tracer.")


    # 어떤 경우든, trace.get_tracer()는 현재 전역 프로바이더로부터 Tracer를 가져옴
    # setup_telemetry()에서 trace.set_tracer_provider(provider)가 성공했다면,
    # trace.get_tracer_provider()는 설정된 SDK Provider를 반환해야 함.
    return trace.get_tracer(name)


def get_finished_test_spans() -> List[ReadableSpan]:
    """테스트용 InMemorySpanExporter에 저장된 완료된 Span들을 반환합니다."""
    if _test_in_memory_exporter:
        spans = _test_in_memory_exporter.get_finished_spans()
        logger.info(f"Retrieved {len(spans)} spans from test exporter")
        return spans
    logger.warning("Test span exporter not initialized, returning empty list")
    return []  # None 대신 빈 리스트 반환


def clear_test_spans() -> None:
    """테스트용 InMemorySpanExporter의 Span들을 초기화합니다."""
    if _test_in_memory_exporter:
        _test_in_memory_exporter.clear()