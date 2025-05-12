import logging
from typing import Optional, List

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.export import (
  BatchSpanProcessor,
  ConsoleSpanExporter,
  SimpleSpanProcessor,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
_tracer_provider: Optional[TracerProvider] = None
_is_telemetry_setup_attempted: bool = False
_test_in_memory_exporter: Optional[InMemorySpanExporter] = None


def setup_telemetry(force_setup: bool = False, for_testing: bool = False) -> None:
  global _tracer_provider, _is_telemetry_setup_attempted, _test_in_memory_exporter
  if not force_setup and _is_telemetry_setup_attempted:
    logger.debug("Telemetry already initialised – skipping.")
    return

  if for_testing:
    if _test_in_memory_exporter:
      _test_in_memory_exporter.clear()
    else:
      _test_in_memory_exporter = InMemorySpanExporter()

    resource = Resource.create(
      {SERVICE_NAME: "test-service", SERVICE_VERSION: "test-version"}
    )

    provider = TracerProvider(resource=resource)
    processor = SimpleSpanProcessor(_test_in_memory_exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    _is_telemetry_setup_attempted = True
    logger.info("OpenTelemetry configured with InMemorySpanExporter for testing.")
    return
  try:
    resource_attributes = {
      SERVICE_NAME: settings.APP_NAME,
      SERVICE_VERSION: settings.APP_VERSION,
      "environment": settings.ENVIRONMENT,
    }
    resource = Resource.create(resource_attributes)
    provider = TracerProvider(resource=resource)
    span_processor: Optional[SpanProcessor] = None

    if for_testing:
      _test_in_memory_exporter = InMemorySpanExporter()
      span_processor = SimpleSpanProcessor(_test_in_memory_exporter)
      logger.info(
        "OpenTelemetry configured with InMemorySpanExporter for testing."
      )

    else:
      otlp_endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT

      if otlp_endpoint:
        insecure_otlp = settings.ENVIRONMENT == "development"
        otlp_exporter = OTLPSpanExporter(
          endpoint=otlp_endpoint, insecure=insecure_otlp
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        logger.info(
          f"OpenTelemetry configured with OTLP exporter to: {otlp_endpoint} (insecure: {insecure_otlp})"
        )

      else:
        console_exporter = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(console_exporter)
        logger.info(
          "OpenTelemetry configured with ConsoleSpanExporter as OTEL_EXPORTER_OTLP_ENDPOINT is not set."
        )

    if span_processor:
      provider.add_span_processor(span_processor)

    else:
      logger.warning("No span processor was configured for OpenTelemetry.")

    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    if (
      not for_testing
      and settings.LANGCHAIN_TRACING_V2
      and settings.LANGCHAIN_API_KEY
    ):
      logger.info(
        "LangSmith tracing appears to be enabled via environment variables."
      )

    logger.info(
      f"OpenTelemetry TracerProvider setup complete. Current global provider: {trace.get_tracer_provider()}"
    )

  except Exception as e:
    logger.error(
      f"Failed to setup OpenTelemetry TracerProvider: {e}", exc_info=True
    )
    _tracer_provider = None


def get_tracer(name: str) -> trace.Tracer:
  """지정된 이름으로 OpenTelemetry Tracer를 가져옵니다."""
  global _tracer_provider, _is_telemetry_setup_attempted

  if not _is_telemetry_setup_attempted:
    logger.warning(
      "OpenTelemetry setup not attempted yet. Attempting basic setup with ConsoleExporter now. "
      "It's recommended to call setup_telemetry() explicitly at application/test start."
    )
    setup_telemetry(for_testing=False)

  current_global_provider = trace.get_tracer_provider()

  if (
    _tracer_provider is None
    or not isinstance(current_global_provider, TracerProvider)
    or isinstance(current_global_provider, trace.NoOpTracerProvider)
  ):
    if _tracer_provider is None:
      logger.debug(
        f"Local _tracer_provider is None. Global provider is {type(current_global_provider)}. Returning tracer from global provider."
      )

    else:
      logger.warning(
        f"Local _tracer_provider is set, but global provider is {type(current_global_provider)}. This might indicate an issue. Using global provider's tracer."
      )

  return trace.get_tracer(name)


def get_finished_test_spans() -> List[ReadableSpan]:
  """테스트용 InMemorySpanExporter에 저장된 완료된 Span들을 반환합니다."""
  if _test_in_memory_exporter:
    spans = _test_in_memory_exporter.get_finished_spans()
    logger.info(f"Retrieved {len(spans)} spans from test exporter")
    return spans
  logger.warning("Test span exporter not initialized, returning empty list")
  return []


def clear_test_spans() -> None:
  """테스트용 InMemorySpanExporter의 Span들을 초기화합니다."""
  if _test_in_memory_exporter:
    _test_in_memory_exporter.clear()
