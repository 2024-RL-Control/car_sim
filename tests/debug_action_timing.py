#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ActionController 디버그 테스트 - 타이밍 정확성 확인
"""

import sys
import os
import time
import numpy as np

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.env.env_rl import BasicRLDrivingEnv

def debug_action_timing():
    """ActionController 타이밍 디버그"""
    print("=== ActionController 타이밍 디버그 ===\n")

    # RL 환경 초기화
    env = BasicRLDrivingEnv()

    print(f"ActionController 디버그 모드: {env.debug_action_timing}")
    print(f"설정된 Hz: {env.action_hz}")
    print(f"시뮬레이션 dt: {env.env.fixed_dt}")
    print(f"예상 행동 간격: {1.0/env.action_hz:.4f}초")
    print(f"시뮬레이션 스텝당 예상 간격: {1.0/env.action_hz / env.env.fixed_dt:.1f}스텝\n")

    # 짧은 테스트 실행
    observations, _ = env.reset()

    print("첫 20스텝의 행동 선택 패턴:")
    for step in range(20):
        # 랜덤 행동 생성
        actions = np.random.uniform(-0.2, 0.2, size=(env.num_vehicles, env.env.action_space.shape[0]))

        # 환경 스텝 실행
        observations, reward, done, truncated, info = env.step(actions)

        # 현재 시뮬레이션 시간과 행동 통계 출력
        sim_time = env.env._time_elapsed
        stats = info.get('action_controller_stats', {})

        print(f"Step {step+1:2d}: 시뮬시간 {sim_time:.4f}초, "
              f"행동선택 {stats.get('action_selection_count', 0)}회")

        if done:
            break

    env.close()
    print("\n디버그 테스트 완료!")

if __name__ == "__main__":
    try:
        debug_action_timing()
    except Exception as e:
        print(f"디버그 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()