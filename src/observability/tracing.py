from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_tracer_provider_configured = False


def configure_tracing(service_name: str = "waf-proxy") -> None:
    global _tracer_provider_configured
    if _tracer_provider_configured:
        return
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        # No OTLP endpoint -> no-op tracer
        trace.set_tracer_provider(
            TracerProvider(resource=Resource.create({"service.name": service_name}))
        )
        _tracer_provider_configured = True
        return

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer_provider_configured = True
    except Exception:
        # Fail silently — tracing is optional
        trace.set_tracer_provider(
            TracerProvider(resource=Resource.create({"service.name": service_name}))
        )
        _tracer_provider_configured = True


def get_tracer(name: str = "waf"):
    return trace.get_tracer(name)
