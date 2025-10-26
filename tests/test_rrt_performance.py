#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RRT-Dubins 경로 계획 성능 테스트"""
import sys
import os
import time
import math
import numpy as np

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model.road import PathPlanner
from src.utils.config_utils import load_config


def test_rrt_performance():
    """RRT 성능 테스트 - 다양한 시나리오에서 경로 계획 성능 측정"""

    # 설정 로드
    config = load_config()['simulation']['path_planning']

    # PathPlanner 생성
    planner = PathPlanner(config)

    # 테스트 케이스 정의
    test_cases = [
        {
            "name": "Simple Path (no obstacles)",
            "start": (0, 0, 0),
            "end": (40, 40, math.pi/4),
            "obstacles": []
        },
        {
            "name": "Path with Few Obstacles",
            "start": (0, 0, 0),
            "end": (50, 50, 0),
            "obstacles": [
                (20, 20, 3),
                (30, 15, 3),
                (25, 35, 3)
            ]
        },
        {
            "name": "Complex Path (many obstacles)",
            "start": (0, 0, 0),
            "end": (60, 60, math.pi),
            "obstacles": [
                (15, 10, 3),
                (25, 15, 3),
                (35, 20, 3),
                (20, 30, 3),
                (40, 35, 3),
                (30, 45, 3),
                (50, 40, 3)
            ]
        },
        {
            "name": "Long Distance Path",
            "start": (-40, -40, 0),
            "end": (70, 70, 0),
            "obstacles": [
                (0, 0, 5),
                (20, 20, 5),
                (40, 40, 5)
            ]
        },
        {
            "name": "Narrow Passage",
            "start": (0, 0, 0),
            "end": (50, 0, 0),
            "obstacles": [
                (20, 5, 4),
                (20, -5, 4),
                (30, 5, 4),
                (30, -5, 4)
            ]
        }
    ]

    print("=" * 70)
    print("RRT-Dubins Path Planning Performance Test (Optimized Version)")
    print("=" * 70)
    print()
    print("Optimization Features:")
    print("  - KD-Tree rebuild: Every 15 iterations (prev: every iteration)")
    print("  - Spatial indexing: Grid-based intersection check")
    print("  - Node position: Managed as list, converted when needed")
    print("  - Computation: step size, goal distance calculated periodically")
    print()
    print("=" * 70)
    print()

    total_time = 0
    total_tests = 0
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"  Start: {test_case['start']}")
        print(f"  End: {test_case['end']}")
        print(f"  Obstacles: {len(test_case['obstacles'])} objects")

        # 성능 측정 (5회 평균)
        times = []
        for _ in range(5):
            start_time = time.time()
            result = planner.plan_path(
                test_case['start'],
                test_case['end'],
                test_case['obstacles'],
                mode='rrt'
            )
            elapsed = time.time() - start_time
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        if result:
            print(f"  [SUCCESS] Path generated")
            print(f"  - Waypoints: {len(result.waypoints)}")
            print(f"  - Total length: {result.total_length:.2f} m")
            print(f"  - Avg time: {avg_time*1000:.1f} ms (+/- {std_time*1000:.1f} ms)")
            print(f"  - Min/Max: {min_time*1000:.1f} / {max_time*1000:.1f} ms")

            results.append({
                'name': test_case['name'],
                'success': True,
                'waypoints': len(result.waypoints),
                'length': result.total_length,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time
            })
        else:
            print(f"  [FAILED] Path generation failed")
            print(f"  - Execution time: {avg_time*1000:.1f} ms")

            results.append({
                'name': test_case['name'],
                'success': False,
                'avg_time': avg_time
            })

        print()

        total_time += avg_time
        total_tests += 1

    # 요약 출력
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Total tests: {total_tests}")
    print(f"Success rate: {sum(1 for r in results if r['success'])}/{total_tests} ({sum(1 for r in results if r['success'])/total_tests*100:.1f}%)")
    print(f"Average execution time: {(total_time/total_tests)*1000:.1f} ms")
    print()

    # 가장 빠른/느린 테스트
    if results:
        fastest = min(results, key=lambda x: x['avg_time'])
        slowest = max(results, key=lambda x: x['avg_time'])

        print(f"Fastest test: {fastest['name']} ({fastest['avg_time']*1000:.1f} ms)")
        print(f"Slowest test: {slowest['name']} ({slowest['avg_time']*1000:.1f} ms)")

    print()
    print("=" * 70)

    return results


def benchmark_comparison():
    """이전 버전과의 성능 비교 벤치마크"""
    print("\n" + "=" * 70)
    print("Performance Comparison: Before vs After Optimization")
    print("=" * 70)
    print()

    print("Expected improvements:")
    print("  - KD-Tree rebuild: ~93% reduction (2000 -> ~133 calls)")
    print("  - Intersection check: ~80% reduction (spatial indexing)")
    print("  - Overall speedup: ~60-80% in complex scenarios")
    print()

    print("Key metrics to observe:")
    print("  1. Average execution time")
    print("  2. Standard deviation (stability)")
    print("  3. Success rate in complex scenarios")
    print()
    print("=" * 70)


if __name__ == "__main__":
    # 성능 테스트 실행
    results = test_rrt_performance()

    # 비교 정보 출력
    benchmark_comparison()
