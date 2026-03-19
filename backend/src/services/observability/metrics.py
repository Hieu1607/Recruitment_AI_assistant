from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class _RouteAggregate:
    count: int = 0
    error_count: int = 0
    sum_latency_ms: float = 0.0


class MetricsRegistry:
    """In-memory request metrics useful for lightweight operational dashboards."""

    def __init__(self, max_samples: int = 2048) -> None:
        self._started_at = time.time()
        self._max_samples = max_samples
        self._recent_latencies: deque[float] = deque(maxlen=max_samples)
        self._route_aggregates: dict[str, _RouteAggregate] = defaultdict(_RouteAggregate)

    def record_http_request(self, path: str, method: str, status_code: int, elapsed_ms: float) -> None:
        route_key = f"{method.upper()} {path}"
        aggregate = self._route_aggregates[route_key]
        aggregate.count += 1
        aggregate.sum_latency_ms += elapsed_ms
        if status_code >= 400:
            aggregate.error_count += 1
        self._recent_latencies.append(elapsed_ms)

    def snapshot(self) -> dict:
        p95 = self._percentile(list(self._recent_latencies), 95)
        uptime_seconds = int(max(0, time.time() - self._started_at))
        routes = {}
        for route_key, aggregate in sorted(self._route_aggregates.items()):
            average_latency = round(
                aggregate.sum_latency_ms / aggregate.count, 2
            ) if aggregate.count else 0.0
            routes[route_key] = {
                "count": aggregate.count,
                "errorCount": aggregate.error_count,
                "averageLatencyMs": average_latency,
            }
        return {
            "uptimeSeconds": uptime_seconds,
            "requestCount": sum(item.count for item in self._route_aggregates.values()),
            "p95LatencyMs": p95,
            "routes": routes,
        }

    @staticmethod
    def _percentile(values: list[float], percentile: int) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int(round((percentile / 100) * (len(ordered) - 1)))
        index = max(0, min(index, len(ordered) - 1))
        return round(ordered[index], 2)


metrics_registry = MetricsRegistry()
