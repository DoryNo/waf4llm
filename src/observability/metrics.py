from __future__ import annotations

from prometheus_client import Counter, Histogram

# Use try to avoid duplicate registration on reload
try:
    REQUESTS_TOTAL = Counter(
        "waf_requests_total",
        "Total proxy requests",
        labelnames=("route", "method", "status"),
    )
    REQUEST_DURATION = Histogram(
        "waf_request_duration_seconds",
        "Request duration seconds",
        labelnames=("route",),
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    )
    PIPELINE_LAYER_DURATION = Histogram(
        "waf_pipeline_layer_duration_seconds",
        "Per-layer pipeline duration",
        labelnames=("layer",),
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
    )
    DETECTIONS_TOTAL = Counter(
        "waf_detections_total",
        "Detections by decision level",
        labelnames=("level", "layer"),
    )
    UPSTREAM_ERRORS = Counter(
        "waf_upstream_errors_total",
        "Upstream errors",
        labelnames=("code",),
    )
    CANARY_HITS = Counter(
        "waf_canary_hits_total",
        "Canary token hits in output",
    )
except ValueError:
    # already registered (reload)
    from prometheus_client import REGISTRY

    def _get(name: str):
        collector = REGISTRY._names_to_collectors.get(name)
        if collector is not None:
            return collector
        # prometheus_client stores Counter collectors without the _total suffix.
        collector = REGISTRY._names_to_collectors.get(name.removesuffix("_total"))
        if collector is not None:
            return collector
        raise KeyError(name)

    REQUESTS_TOTAL = _get("waf_requests_total")
    REQUEST_DURATION = _get("waf_request_duration_seconds")
    PIPELINE_LAYER_DURATION = _get("waf_pipeline_layer_duration_seconds")
    DETECTIONS_TOTAL = _get("waf_detections_total")
    UPSTREAM_ERRORS = _get("waf_upstream_errors_total")
    CANARY_HITS = _get("waf_canary_hits_total")


class _Metrics:
    requests_total = REQUESTS_TOTAL
    request_duration = REQUEST_DURATION
    pipeline_layer_duration = PIPELINE_LAYER_DURATION
    detections_total = DETECTIONS_TOTAL
    upstream_errors = UPSTREAM_ERRORS
    canary_hits = CANARY_HITS


metrics = _Metrics()
