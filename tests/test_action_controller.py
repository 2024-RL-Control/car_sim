#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ActionController 테스트 스크립트
RL 환경에서 행동 선택 빈도 제어가 제대로 작동하는지 확인
"""

import sys
import os
import time
import numpy as np
import pygame

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.env.env_rl import BasicRLDrivingEnv

def test_action_controller():
    """ActionController 기능 테스트"""
    print("=== ActionController 테스트 시작 ===\n")

    # RL 환경 초기화
    env = BasicRLDrivingEnv()

    print(f"ActionController 상태: {'활성화' if env.action_controller else '비활성화'}")
    if env.action_controller:
        print(f"설정된 Hz: {env.action_hz}")
        print(f"행동 유지 동작: {env.action_hold_behavior}")
        print(f"디버그 모드: {env.debug_action_timing}")
        print()

    # 테스트 실행
    max_episodes = 3
    max_steps_per_episode = 100

    for episode in range(max_episodes):
        print(f"--- Episode {episode + 1} ---")

        observations, _ = env.reset()
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            # 키보드 입력 처리 (ESC로 종료 가능)
            env.handle_keyboard_input()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False

            # 랜덤 행동 생성 (테스트용)
            actions = np.random.uniform(-0.5, 0.5, size=(env.num_vehicles, env.env.action_space.shape[0]))

            # 환경 스텝 실행
            observations, reward, done, truncated, info = env.step(actions)
            step_count += 1

            # 시각화
            env.render()

            # ActionController 통계 출력 (10스텝마다)
            if env.action_controller and step_count % 10 == 0:
                stats = info.get('action_controller_stats', {})
                print(f"Step {step_count}: 행동선택 {stats.get('action_selection_count', 0)}회 "
                      f"/ 총 {stats.get('step_count', 0)}스텝 "
                      f"({stats.get('actual_selection_rate', 0):.2%})")

        # 에피소드 종료 시 최종 통계
        if env.action_controller:
            final_stats = env.action_controller.get_statistics()
            expected_selections = step_count / (1/env.action_hz * (1/env.env.fixed_dt))
            actual_selections = final_stats['action_selection_count']

            print(f"에피소드 완료:")
            print(f"  총 스텝: {step_count}")
            print(f"  행동 선택: {actual_selections}회")
            print(f"  예상 선택: {expected_selections:.1f}회")
            print(f"  선택 비율: {final_stats['actual_selection_rate']:.2%}")
            print(f"  목표 Hz: {env.action_hz}Hz")

            if actual_selections > 0:
                actual_hz = actual_selections / (step_count * env.env.fixed_dt)
                print(f"  실제 Hz: {actual_hz:.2f}Hz")

            print()

    env.close()
    print("테스트 완료!")

def test_action_controller_configurations():
    """다양한 ActionController 설정 테스트"""
    print("=== 다양한 설정 테스트 ===\n")

    # 임시 설정 파일로 다른 Hz 값 테스트
    test_configs = [
        {'hz': 5, 'name': '5Hz (느린 반응)'},
        {'hz': 15, 'name': '15Hz (기본값)'},
        {'hz': 30, 'name': '30Hz (빠른 반응)'},
    ]

    for config in test_configs:
        print(f"--- {config['name']} 테스트 ---")

        # config 파일을 직접 수정하지 않고 환경 변수 등을 통해 테스트할 수 있지만
        # 여기서는 간단히 설명만 출력
        hz = config['hz']
        dt = 0.016  # 기본 dt
        expected_interval = 1.0 / hz
        sim_steps_per_action = expected_interval / dt

        print(f"  설정 Hz: {hz}Hz")
        print(f"  행동 간격: {expected_interval:.4f}초")
        print(f"  시뮬레이션 스텝당 행동: {sim_steps_per_action:.1f}스텝에 1번")
        print()

if __name__ == "__main__":
    try:
        # ActionController 기능 테스트
        test_action_controller()

        # 다양한 설정 정보 출력
        test_action_controller_configurations()

    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()