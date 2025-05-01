# -*- coding: utf-8 -*-
import pygame
import numpy as np

class KeyboardHandler:
    """키보드 입력 처리 클래스"""

    def __init__(self, config):
        """
        키보드 핸들러 초기화

        Args:
            config: 시뮬레이터 설정 정보
        """
        self.config = config

        # 키보드 상태 (연속 입력 처리)
        self._keys_pressed = {
            'up': False,     # 가속
            'down': False,   # 제동
            'left': False,   # 좌회전
            'right': False,  # 우회전
            'brake': False,  # 수동 브레이크 (Space)
        }

        # 키 상태 추적 (이전 프레임에 눌렸는지 여부)
        self._keys_state = {
            pygame.K_F1: False,  # 학습 모드 토글
            pygame.K_F2: False,  # 전체 렌더링 토글
            pygame.K_F3: False,  # HUD 렌더링 토글
            pygame.K_F4: False,  # 디버그 모드 토글
            pygame.K_F5: False,  # 상태 저장
            pygame.K_F9: False,  # 상태 로드
            pygame.K_TAB: False,  # 차량 전환
        }

    def handle_keyboard_input(self, env, camera):
        """
        키보드 입력 처리 (상태 업데이트)

        Args:
            env: 시뮬레이터 환경 객체
            camera: 카메라 객체

        Returns:
            액션 배열 또는 다중 차량일 경우 액션 리스트
        """
        pressed = pygame.key.get_pressed()

        # 메인 키 상태 업데이트
        self._keys_pressed['up'] = pressed[pygame.K_UP] or pressed[pygame.K_w]
        self._keys_pressed['down'] = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
        self._keys_pressed['left'] = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
        self._keys_pressed['right'] = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
        self._keys_pressed['brake'] = pressed[pygame.K_SPACE]

        # 차량 전환 (다중 차량 모드일 때만)
        if env.multi_vehicle and pressed[pygame.K_TAB] and not self._keys_state[pygame.K_TAB]:
            # 다음 차량으로 전환
            env.vehicle_manager.cycle_active_vehicle()
            print(f"현재 차량: {env.vehicle_manager.get_active_vehicle_index() + 1}/{env.vehicle_manager.get_vehicle_count()}")
        self._keys_state[pygame.K_TAB] = pressed[pygame.K_TAB]

        # 카메라 컨트롤 처리
        camera.handle_camera_controls(pressed)

        # 토글 키 처리 - 키가 눌렸다가 떼어질 때 한 번만 실행되도록 함
        # F1: 학습 모드 토글
        if pressed[pygame.K_F1] and not self._keys_state[pygame.K_F1]:
            env.config['visualization']['training_mode'] = not env.config['visualization']['training_mode']
            print(f"학습 모드: {'켜짐' if env.config['visualization']['training_mode'] else '꺼짐'}")
        self._keys_state[pygame.K_F1] = pressed[pygame.K_F1]

        # F2: 전체 렌더링 토글
        if pressed[pygame.K_F2] and not self._keys_state[pygame.K_F2]:
            env.config['visualization']['visualize'] = not env.config['visualization']['visualize']
            print(f"전체 렌더링: {'켜짐' if env.config['visualization']['visualize'] else '꺼짐'}")
        self._keys_state[pygame.K_F2] = pressed[pygame.K_F2]

        # F3: HUD 렌더링 토글
        if pressed[pygame.K_F3] and not self._keys_state[pygame.K_F3]:
            env.config['visualization']['visualize_hud'] = not env.config['visualization']['visualize_hud']
            print(f"HUD 렌더링: {'켜짐' if env.config['visualization']['visualize_hud'] else '꺼짐'}")
        self._keys_state[pygame.K_F3] = pressed[pygame.K_F3]

        # F4: 디버그 모드 토글
        if pressed[pygame.K_F4] and not self._keys_state[pygame.K_F4]:
            env.config['visualization']['debug_mode'] = not env.config['visualization']['debug_mode']
            print(f"디버그 모드: {'켜짐' if env.config['visualization']['debug_mode'] else '꺼짐'}")
        self._keys_state[pygame.K_F4] = pressed[pygame.K_F4]

        # 저장 및 로드
        if pressed[pygame.K_F5] and not self._keys_state[pygame.K_F5]:  # 상태 저장
            env._save_state()
        self._keys_state[pygame.K_F5] = pressed[pygame.K_F5]

        if pressed[pygame.K_F9] and not self._keys_state[pygame.K_F9]:  # 상태 로드
            env._load_state()
        self._keys_state[pygame.K_F9] = pressed[pygame.K_F9]

        # 입력에서 액션 결정
        action = np.zeros(2)

        # 학습 모드 아닐 때, 차량 제어 (조작 모드)
        if not env.config['visualization']['training_mode']:
            # 가속/제동 입력
            if self._keys_pressed['up']:
                action[0] = 1.0  # 가속
            if self._keys_pressed['down']:
                action[0] = -5.0  # 제동

            # 브레이크 키가 눌려있으면 최대 제동
            if self._keys_pressed['brake']:
                action[0] = -1.0

            # 조향 입력
            if self._keys_pressed['left']:
                action[1] = -1.0  # 좌회전
            if self._keys_pressed['right']:
                action[1] = 1.0  # 우회전

        # 다중 차량 모드에서는 활성 차량만 입력으로 제어
        if env.multi_vehicle:
            actions = [np.zeros(2) for _ in range(env.vehicle_manager.get_vehicle_count())]
            active_idx = env.vehicle_manager.get_active_vehicle_index()
            actions[active_idx] = action
            return actions
        else:
            return action

    def reset(self):
        """키보드 상태 초기화"""
        self._keys_pressed = {
            'up': False,
            'down': False,
            'left': False,
            'right': False,
            'brake': False,
        }

        self._keys_state = {
            pygame.K_F1: False,
            pygame.K_F2: False,
            pygame.K_F3: False,
            pygame.K_F4: False,
            pygame.K_F5: False,
            pygame.K_F9: False,
            pygame.K_TAB: False,
        }
