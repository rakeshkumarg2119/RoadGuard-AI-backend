"""
RoadGuard AI - Standalone Performance & Benchmark Test Suite

Measures latency (mean, p50, p95, p99) and throughput (RPS) for:
1. Physics Engine Calculation Speed
2. Camera Calibration & YOLO Detector Inference Benchmark
3. API Endpoints Latency & Concurrency (/health, /simulate/info, /simulate/step, /alert)

Run directly with Python:
    python tests/test_performance.py
Or with pytest:
    pytest tests/test_performance.py
"""

import io
import os
import sys
import time
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from PIL import Image
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import app
from core.physics_engine import PotholePhysicsEngine, PotholeGeometry, RoadCondition
from core.camera_calibration import CameraCalibrator, BoundingBox
from core.pothole_detector import PotholeDetector


# =========================================================
# STATS CALCULATOR HELPER
# =========================================================

def calculate_stats(latencies_ms: list[float], total_duration_sec: float) -> dict:
    """Computes mean, p50, p95, p99 latencies (ms) and Requests Per Second (RPS)."""
    if not latencies_ms:
        return {"count": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "rps": 0}

    sorted_lat = sorted(latencies_ms)
    count = len(sorted_lat)

    p50_idx = int(0.50 * count)
    p95_idx = min(int(0.95 * count), count - 1)
    p99_idx = min(int(0.99 * count), count - 1)

    mean_ms = sum(sorted_lat) / count
    rps = count / total_duration_sec if total_duration_sec > 0 else 0

    return {
        "count": count,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(sorted_lat[p50_idx], 3),
        "p95_ms": round(sorted_lat[p95_idx], 3),
        "p99_ms": round(sorted_lat[p99_idx], 3),
        "rps": round(rps, 2),
    }


def print_stats(name: str, stats: dict):
    """Prints formatted performance benchmark metrics."""
    divider = "=" * 60
    print(f"\n{divider}")
    print(f" BENCHMARK: {name}")
    print(f"{divider}")
    print(f"  Total Operations : {stats['count']}")
    print(f"  Throughput (RPS) : {stats['rps']} ops/sec")
    print(f"  Mean Latency     : {stats['mean_ms']} ms")
    print(f"  p50 Latency      : {stats['p50_ms']} ms")
    print(f"  p95 Latency      : {stats['p95_ms']} ms")
    print(f"  p99 Latency      : {stats['p99_ms']} ms")
    print(f"{divider}")


# =========================================================
# 1. CORE PHYSICS BENCHMARK
# =========================================================

def benchmark_physics_engine(iterations: int = 5000) -> dict:
    """Benchmark PotholePhysicsEngine calculation performance."""
    engine = PotholePhysicsEngine()
    pothole = PotholeGeometry(width_m=0.45, depth_m=0.08)

    latencies = []
    t_start = time.perf_counter()

    for i in range(iterations):
        speed = 10.0 + (i % 80)
        t0 = time.perf_counter()
        engine.calculate(speed, pothole, RoadCondition.DRY)
        latencies.append((time.perf_counter() - t0) * 1000)

    total_duration = time.perf_counter() - t_start
    stats = calculate_stats(latencies, total_duration)
    print_stats(f"Physics Engine ({iterations} iterations)", stats)
    return stats


# =========================================================
# 2. DETECTOR & CAMERA CALIBRATION BENCHMARK
# =========================================================

def benchmark_detector_and_calibration(iterations: int = 50) -> dict:
    """Benchmark YOLO PotholeDetector and Camera Calibration inference."""
    detector = PotholeDetector(min_confidence=0.10)
    frame = np.full((720, 1280, 3), fill_value=80, dtype=np.uint8)
    frame[500:600, 500:700] = 20

    latencies = []
    t_start = time.perf_counter()

    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.process_frame(frame, speed_kmh=40.0, road_condition=RoadCondition.DRY)
        latencies.append((time.perf_counter() - t0) * 1000)

    total_duration = time.perf_counter() - t_start
    stats = calculate_stats(latencies, total_duration)
    print_stats(f"Pothole Detector Frame Processing ({iterations} frames)", stats)
    return stats


# =========================================================
# 3. FASTAPI API CONCURRENCY BENCHMARK
# =========================================================

async def benchmark_api_endpoints(concurrency: int = 20, total_requests: int = 200) -> dict:
    """Benchmark API endpoints under concurrent load using httpx AsyncClient."""

    # Mock DB/Store dependencies for API benchmarking
    mock_col = AsyncMock()
    mock_col.find_one.return_value = {
        "_id": "507f1f77bcf86cd799439011",
        "gps_readable": {"lat": 9.9252, "lon": 78.1198},
        "pothole": {"width_m": 0.4, "depth_m": 0.08},
        "seen_count": 1,
    }

    async def async_gen(items):
        for item in items:
            yield item

    mock_cursor = MagicMock()
    mock_cursor.__aiter__.side_effect = lambda: async_gen([{
        "_id": "507f1f77bcf86cd799439011",
        "gps_readable": {"lat": 9.9252, "lon": 78.1198},
        "seen_count": 1,
        "speed_simulation": {
            "30": {
                "d_alert_m": 35.0,
                "d_stop_m": 10.0,
                "d_react_m": 25.0,
                "severity": "medium",
                "fall_type": "controlled",
            }
        },
    }])
    mock_cursor.limit.return_value = mock_cursor
    mock_col.find.return_value = mock_cursor

    mock_store = AsyncMock()
    mock_store.collection = mock_col
    mock_store.get_alert_for_speed.return_value = {
        "pothole_id": "507f1f77bcf86cd799439011",
        "distance_m": 15.0,
        "d_alert_m": 30.0,
        "d_stop_m": 10.0,
        "zone": "stage2",
        "severity": "medium",
        "fall_type": "controlled",
    }

    mock_weather = MagicMock()
    mock_weather.get_current_sync.return_value = None

    with patch("routes.alert.get_store", return_value=mock_store), \
         patch("routes.alert.get_weather", return_value=mock_weather), \
         patch("routes.simulate.get_store", return_value=mock_store):

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            semaphore = asyncio.Semaphore(concurrency)
            latencies = []

            async def worker(url: str):
                async with semaphore:
                    t0 = time.perf_counter()
                    resp = await client.get(url)
                    assert resp.status_code == 200
                    latencies.append((time.perf_counter() - t0) * 1000)

            urls = [
                "/health",
                "/simulate/info?speed_kmh=40",
                "/simulate/step?pothole_id=507f1f77bcf86cd799439011&step=3",
                "/alert?lat=9.9252&lon=78.1198&speed_kmh=30",
            ]

            tasks = [worker(urls[i % len(urls)]) for i in range(total_requests)]

            t_start = time.perf_counter()
            await asyncio.gather(*tasks)
            total_duration = time.perf_counter() - t_start

            stats = calculate_stats(latencies, total_duration)
            print_stats(f"FastAPI Endpoints ({total_requests} reqs, {concurrency} concurrent)", stats)
            return stats


# =========================================================
# MAIN ENTRYPOINT & PYTEST HOOK
# =========================================================

def test_run_all_benchmarks():
    """Pytest wrapper function to execute all benchmarks."""
    stats_physics = benchmark_physics_engine(iterations=2000)
    assert stats_physics["count"] == 2000
    assert stats_physics["mean_ms"] < 1.0  # Should be sub-millisecond

    stats_detector = benchmark_detector_and_calibration(iterations=10)
    assert stats_detector["count"] == 10

    stats_api = asyncio.run(benchmark_api_endpoints(concurrency=10, total_requests=100))
    assert stats_api["count"] == 100


if __name__ == "__main__":
    divider = "=" * 60
    print(f"\n{divider}")
    print(" ROADGUARD AI BACKEND — PERFORMANCE BENCHMARK SUITE")
    print(divider)

    benchmark_physics_engine(iterations=5000)
    benchmark_detector_and_calibration(iterations=20)
    asyncio.run(benchmark_api_endpoints(concurrency=20, total_requests=200))

    print("\nBenchmark Suite Completed Successfully.\n")
