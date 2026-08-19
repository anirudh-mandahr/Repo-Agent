"""OpenTelemetry tracing with correlation_id as the shared trace identity."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    SpanKind,
    Status,
    StatusCode,
    TraceFlags,
    get_tracer_provider,
    set_span_in_context,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from core.logging import correlation_id_from_mcp_meta

_TRACER_NAME = "repochat"
_propagator = TraceContextTextMapPropagator()
_configured = False


def _trace_id_from_correlation(correlation_id: str) -> int:
    try:
        value = uuid.UUID(correlation_id).int
    except ValueError:
        value = uuid.uuid5(uuid.NAMESPACE_URL, correlation_id).int
    return value & ((1 << 128) - 1)


def _new_span_id() -> int:
    value = uuid.uuid4().int & ((1 << 64) - 1)
    return value or 1


def configure_tracing(service_name: str) -> None:
    """Install a process-wide TracerProvider once.

    Spans export over OTLP HTTP when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.

    Args:
        service_name: ``service.name`` resource attribute.
    """
    global _configured
    if _configured:
        return
    provider = get_tracer_provider()
    if isinstance(provider, TracerProvider):
        _configured = True
        return
    resource = Resource.create({"service.name": service_name})
    sdk_provider = TracerProvider(resource=resource)
    exporter = _build_exporter()
    if exporter is not None:
        sdk_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(sdk_provider)
    _configured = True


def _build_exporter() -> SpanExporter | None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=endpoint)


def _seed_context(correlation_id: str) -> otel_context.Context:
    span_context = SpanContext(
        trace_id=_trace_id_from_correlation(correlation_id),
        span_id=_new_span_id(),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return set_span_in_context(NonRecordingSpan(span_context))


@contextmanager
def start_span(
    name: str,
    *,
    correlation_id: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    **attributes: Any,
) -> Iterator[Span]:
    """Start a span, seeding the trace id from ``correlation_id`` when needed.

    Args:
        name: Span name.
        correlation_id: Request id shared across MCP hops.
        kind: OpenTelemetry span kind.
        **attributes: Extra span attributes.

    Yields:
        The active span.
    """
    tracer = trace.get_tracer(_TRACER_NAME)
    parent = otel_context.get_current()
    current = trace.get_current_span()
    if (
        correlation_id
        and correlation_id not in {"", "-"}
        and not current.get_span_context().is_valid
    ):
        parent = _seed_context(correlation_id)
    attrs: dict[str, Any] = {"correlation_id": correlation_id or "-"}
    attrs.update(attributes)
    with tracer.start_as_current_span(name, context=parent, kind=kind, attributes=attrs) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def inject_trace_carrier(correlation_id: str) -> dict[str, str]:
    """Build MCP ``meta`` with correlation_id plus W3C ``traceparent``.

    Args:
        correlation_id: Request id to propagate.

    Returns:
        Carrier mapping safe to pass as MCP tool metadata.
    """
    carrier: dict[str, str] = {"correlation_id": correlation_id}
    _propagator.inject(carrier)
    return carrier


def attach_from_mcp_meta(meta: object | None) -> object:
    """Extract W3C context from MCP request meta and attach it.

    Args:
        meta: FastMCP ``request_context.meta`` value.

    Returns:
        Attachment token for :func:`opentelemetry.context.detach`.
    """
    carrier = _meta_as_carrier(meta)
    extracted = _propagator.extract(carrier)
    correlation_id = correlation_id_from_mcp_meta(meta) or carrier.get("correlation_id")
    current = trace.get_current_span(extracted)
    if (
        correlation_id
        and correlation_id not in {"", "-"}
        and not current.get_span_context().is_valid
    ):
        extracted = _seed_context(correlation_id)
    return otel_context.attach(extracted)


def detach_trace(token: object) -> None:
    """Undo :func:`attach_from_mcp_meta`.

    Args:
        token: Value returned by :func:`attach_from_mcp_meta`.
    """
    otel_context.detach(token)  # type: ignore[arg-type]


def _meta_as_carrier(meta: object | None) -> dict[str, str]:
    if meta is None:
        return {}
    carrier: dict[str, str] = {}
    extra = getattr(meta, "model_extra", None)
    mapping = extra if isinstance(extra, dict) else {}
    for key in ("correlation_id", "traceparent", "tracestate"):
        raw = getattr(meta, key, None)
        if raw is None:
            raw = mapping.get(key)
        if raw is not None:
            carrier[key] = str(raw)
    if isinstance(meta, dict):
        for key in ("correlation_id", "traceparent", "tracestate"):
            raw = meta.get(key)
            if raw is not None:
                carrier[key] = str(raw)
    return carrier
