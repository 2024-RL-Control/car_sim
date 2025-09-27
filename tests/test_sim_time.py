#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시뮬레이션 타이밍 시스템 테스트 스크립트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model.vehicle import SubsystemManager
from src.utils.config_utils import load_config

def test_timing_validation():
    """타이밍 검증 시스템 테스트"""
    print("=== 시뮬레이션 타이밍 검증 시스템 테스트 ===\n")

    # 설정 로드
    config = load_config()
    simulation_config = config['simulation']

    # SubsystemManager 생성
    subsystem_manager = SubsystemManager(simulation_config=simulation_config)

    # 현재 설정 검증
    dt = simulation_config['dt']
    hz_config = simulation_config.get('hz', {})

    print("현재 설정:")
    print(f"  - 시뮬레이션 dt: {dt}초 ({1/dt:.1f}Hz)")
    print(f"  - 서브시스템 Hz 설정: {hz_config}")
    print()

    # 검증 실행
    subsystem_manager.print_timing_validation_report(dt, hz_config)

    # 다양한 시나리오 테스트
    print("\n다양한 시나리오 테스트:")

    test_scenarios = [
        {
            "name": "고속 시뮬레이션 (dt=0.008, 125Hz)",
            "dt": 0.008,
            "hz": hz_config
        },
        {
            "name": "저속 시뮬레이션 (dt=0.033, 30Hz)",
            "dt": 0.033,
            "hz": hz_config
        },
        {
            "name": "비효율적 Hz 설정",
            "dt": 0.016,
            "hz": {
                "collision_check": 33,  # 비효율적
                "lidar_update": 17,     # 비효율적
                "goal_check": 3         # 효율적
            }
        }
    ]

    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        result = subsystem_manager.validate_timing_configuration(scenario['dt'], scenario['hz'])

        status = "양호" if result['is_valid'] else "문제 발견"
        print(f"검증 결과: {status}")

        if result['warnings']:
            print("경고:")
            for warning in result['warnings'][:2]:  # 최대 2개만 출력
                print(f"  - {warning}")

        if result['recommendations']:
            print("권장사항:")
            for rec in result['recommendations'][:2]:  # 최대 2개만 출력
                print(f"  - {rec}")

def test_simulation_speed_consistency():
    """시뮬레이션 속도 일관성 테스트"""
    print("\n\n=== 시뮬레이션 속도 일관성 테스트 ===")
    print("이제 Hz 시스템이 시뮬레이션 시간 기준으로 동작하므로,")
    print("FPS가 변해도 서브시스템들은 일관되게 동작합니다.\n")

    # 예시 시나리오
    scenarios = [
        {"fps": 30, "dt": 0.016},   # 느린 FPS, 빠른 시뮬레이션
        {"fps": 60, "dt": 0.016},   # 표준 설정
        {"fps": 120, "dt": 0.016},  # 빠른 FPS, 빠른 시뮬레이션
    ]

    lidar_hz = 20  # 예시: LIDAR 20Hz

    for scenario in scenarios:
        fps = scenario["fps"]
        dt = scenario["dt"]
        sim_hz = 1.0 / dt

        # 1초 동안의 시뮬레이션
        steps_per_second = fps
        sim_time_per_second = steps_per_second * dt
        expected_lidar_calls = sim_time_per_second * lidar_hz

        print(f"FPS {fps}, dt {dt}:")
        print(f"  - 1초 실시간 = {steps_per_second}번 step() 호출")
        print(f"  - 시뮬레이션 시간 경과: {sim_time_per_second:.2f}초")
        print(f"  - 예상 LIDAR 호출 횟수: {expected_lidar_calls:.1f}번")
        print(f"  - 실제 LIDAR Hz: {expected_lidar_calls:.1f}Hz (설정: {lidar_hz}Hz)")
        print()

if __name__ == "__main__":
    try:
        test_timing_validation()
        test_simulation_speed_consistency()
        print("모든 테스트가 완료되었습니다!")

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()