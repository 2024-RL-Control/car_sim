# -*- coding: utf-8 -*-
import numpy as np
import pygame

class Camera:
    """차량 시뮬레이터의 카메라 관리 클래스"""

    def __init__(self, config):
        """
        카메라 초기화

        Args:
            config: 시뮬레이터 설정 정보
        """
        self.config = config
        self._camera_pos = (0, 0)    # 카메라 위치 (카메라가 차량을 따라가지 않을 때 사용)
        self._camera_offset = (0, 0)  # 카메라 오프셋 (팬/줌 기능용)

        self._init_camera_follow = self.config['visualization']['camera_follow']
        self._init_camera_zoom = self.config['visualization']['camera_zoom']

        self._keys_state = {
            pygame.K_r: False,
            pygame.K_c: False
        }

    def world_to_screen(self, world_points, vehicle=None):
        """
        월드 좌표 리스트 → 화면 좌표 리스트 변환
        월드: 원점 중앙, Y축 위로 증가
        화면: 원점 좌상단, Y축 아래로 증가

        카메라 추적 모드시 활성 차량을 화면 중앙에 오도록 변환
        정적 카메라 모드시 카메라 위치를 원점으로 설정

        Args:
            world_points: 변환할 월드 좌표들의 리스트 또는 NumPy 배열. 각 요소는 (x, y) 튜플 또는 [x, y] 형태
            vehicle: 카메라 위치를 가져올 차량 객체
        """
        if isinstance(world_points, np.ndarray):
            world_points_arr = world_points
        else:
            world_points_arr = np.array(world_points)

        flag = False
        if world_points_arr.ndim == 1: # 단일 좌표가 [x,y] 형태로 들어온 경우 2D 배열로 만듦: [x,y] -> [[x,y]]
            flag = True
            world_points_arr = world_points_arr.reshape(-1, 2)

        xs = world_points_arr[:, 0]
        ys = world_points_arr[:, 1]

        # 카메라 위치 결정
        if self.config['visualization']['camera_follow'] and vehicle is not None:
            # 카메라 추적 모드가 켜져 있고 차량이 주어진 경우, 차량 위치 사용
            cam_x = vehicle.state.x
            cam_y = vehicle.state.y
            self._camera_pos = (cam_x, cam_y)
        else:
            # 카메라 추적 모드가 꺼져 있거나 차량이 없는 경우, 기존 카메라 위치 사용
            cam_x, cam_y = self._camera_pos

        # 줌 처리
        scale = self.config['visualization']['scale_factor'] * self.config['visualization']['camera_zoom']

        # 월드 좌표를 카메라 기준으로 상대화
        rel_xs = xs - cam_x
        rel_ys = ys - cam_y

        # 스케일 적용 및 화면 중앙 기준으로 변환
        screen_xs = rel_xs * scale + self.config['visualization']['window_width']/2
        screen_ys = self.config['visualization']['window_height']/2 - rel_ys * scale

        # 카메라 오프셋 적용 (수동 카메라 조작용)
        screen_xs += self._camera_offset[0]
        screen_ys += self._camera_offset[1]

        # (int, int) 튜플의 배열로 변환
        screen_points_arr = np.vstack((screen_xs, screen_ys)).T.astype(int)

        if flag:
            return tuple(screen_points_arr[0]) # 단일 입력 시 단일 (x,y) 튜플 반환
        else:
            return screen_points_arr.tolist() # 리스트 입력 시 [(x1,y1), (x2,y2), ...] 반환

    def handle_camera_controls(self, pressed_keys):
        """
        카메라 컨트롤 처리

        Args:
            pressed_keys: pygame.key.get_pressed() 결과
        """
        # 줌 컨트롤
        if pressed_keys[pygame.K_PLUS] or pressed_keys[pygame.K_EQUALS]:
            self.config['visualization']['camera_zoom'] = min(7.0, self.config['visualization']['camera_zoom'] * 1.05)
        if pressed_keys[pygame.K_MINUS]:
            self.config['visualization']['camera_zoom'] = max(0.25, self.config['visualization']['camera_zoom'] / 1.05)

        # 카메라 리셋
        if pressed_keys[pygame.K_r] and not self._keys_state[pygame.K_r]:
            self.config['visualization']['camera_zoom'] = self._init_camera_zoom
            self._camera_offset = (0, 0)
        self._keys_state[pygame.K_r] = pressed_keys[pygame.K_r]

        # 카메라 팬
        pan_speed = 5

        # 카메라 오프셋 변경
        if pressed_keys[pygame.K_i]:  # 위로 이동
            self._camera_offset = (self._camera_offset[0], self._camera_offset[1] + pan_speed)
        if pressed_keys[pygame.K_k]:  # 아래로 이동
            self._camera_offset = (self._camera_offset[0], self._camera_offset[1] - pan_speed)
        if pressed_keys[pygame.K_j]:  # 왼쪽으로 이동
            self._camera_offset = (self._camera_offset[0] + pan_speed, self._camera_offset[1])
        if pressed_keys[pygame.K_l]:  # 오른쪽으로 이동
            self._camera_offset = (self._camera_offset[0] - pan_speed, self._camera_offset[1])

        # 카메라 추적 토글
        if pressed_keys[pygame.K_c] and not self._keys_state[pygame.K_c]:
            self.config['visualization']['camera_follow'] = not self.config['visualization']['camera_follow']
            if self.config['visualization']['camera_follow']:
                self.config['visualization']['camera_zoom'] = 1.5
            else:
                self.config['visualization']['camera_zoom'] = self._init_camera_zoom
            self._camera_pos = (0, 0)
            self._camera_offset = (0, 0)
        self._keys_state[pygame.K_c] = pressed_keys[pygame.K_c]

    def reset(self):
        """카메라 설정 초기화"""
        self._camera_pos = (0, 0)
        self._camera_offset = (0, 0)
        # self.config['visualization']['camera_follow'] = self._init_camera_follow
        # self.config['visualization']['camera_zoom'] = self._init_camera_zoom
        self._keys_state = {
            pygame.K_r: False,
            pygame.K_c: False
        }

    def get_offset(self):
        """카메라 오프셋 반환"""
        return self._camera_offset

    def get_position(self):
        """카메라 위치 반환"""
        return self._camera_pos

    def get_serializable_camera(self):
        """
        카메라 설정을 직렬화 가능한 형태로 반환 (저장용)

        Returns:
            직렬화 가능한 카메라 설정 딕셔너리
        """
        return {
            'offset_x': self._camera_offset[0],
            'offset_y': self._camera_offset[1],
            'pos_x': self._camera_pos[0],
            'pos_y': self._camera_pos[1],
            'zoom': self.config['visualization']['camera_zoom'],
            'follow_mode': self.config['visualization']['camera_follow']
        }

    def load_from_serialized(self, serialized_data):
        """
        직렬화된 카메라 설정으로부터 카메라 상태 복원 (로드용)

        Args:
            serialized_data: 직렬화된 카메라 설정 딕셔너리
        """
        offset_x = serialized_data.get('offset_x', 0)
        offset_y = serialized_data.get('offset_y', 0)
        self._camera_offset = (offset_x, offset_y)

        pos_x = serialized_data.get('pos_x', 0)
        pos_y = serialized_data.get('pos_y', 0)
        self._camera_pos = (pos_x, pos_y)

        zoom = serialized_data.get('zoom', 1.0)
        follow_mode = serialized_data.get('follow_mode', True)

        self.config['visualization']['camera_zoom'] = zoom
        self.config['visualization']['camera_follow'] = follow_mode
