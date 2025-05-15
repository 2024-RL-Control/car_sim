# -*- coding: utf-8 -*-
import pygame
import numpy as np
import time
import os
import ctypes
from collections import deque

class Renderer:
    """차량 시뮬레이터 렌더링 클래스"""

    def __init__(self, config):
        """
        렌더러 초기화

        Args:
            config: 시뮬레이터 설정 정보
        """
        self.config = config
        self.screen = None
        self.clock = None
        self.font = None
        self.large_font = None

        # 그리드 렌더링 최적화용 캐시
        self._grid_cache = None
        self._grid_cache_params = None

        # 성능 측정 변수
        self._performance_metrics = {
            'fps': 0,
            'physics_time': 0,
            'render_time': 0
        }

        # 시간 관리 변수
        self._last_update_time = time.time()
        self._frame_times = []

    def init_pygame(self):
        """Pygame 초기화 및 그래픽 리소스 로드"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.config['visualization']['window_size'])
        pygame.display.set_caption("Vehicle Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)  # 큰 글꼴 추가

        # 프레임 시간 관리 초기화
        self._frame_times = deque(maxlen=60)  # 최근 60프레임 시간 (FPS 계산용)
        self._last_update_time = time.time()

        return self.screen, self.clock

    def draw_grid(self, camera):
        """
        고정 그리드 그리기

        Args:
            camera: 카메라 객체
        """
        scale = self.config['visualization']['scale_factor']
        grid_size = 10 * scale   # 그리드 단위 (픽셀 수) * 스케일

        # cam_x, cam_y = camera.get_position()
        cam_x = 0
        cam_y = 0

        # 화면 영역(viewport)에서 보이는 월드 좌표 범위 계산
        # screen_width = self.config['visualization']['window_width']
        # screen_height = self.config['visualization']['window_height']
        screen_width = 3000
        screen_height = 3000

        # 화면 경계에 해당하는 월드 좌표 계산
        world_left = cam_x - screen_width / (2 * scale)
        world_right = cam_x + screen_width / (2 * scale)
        world_top = cam_y + screen_height / (2 * scale)
        world_bottom = cam_y - screen_height / (2 * scale)

        # 화면에 표시될 그리드 라인의 시작/끝 인덱스 계산
        grid_start_x = int(world_left / grid_size)
        grid_end_x = int(world_right / grid_size)
        grid_start_y = int(world_bottom / grid_size)
        grid_end_y = int(world_top / grid_size)

        # 수직 그리드 라인 (X축과 평행한 선)
        for i in range(grid_start_x, grid_end_x + 1):
            # 그리드 라인의 월드 좌표
            grid_x = i * grid_size

            # 화면 좌표로 변환하여 그리기 (카메라 위치 전달)
            start_v = camera.world_to_screen(grid_x, world_bottom)
            end_v = camera.world_to_screen(grid_x, world_top)
            pygame.draw.line(self.screen, self.config['visualization']['grid_color'], start_v, end_v, 1)

            # 좌표 표시 (원점 제외)
            if i != 0:
                x_coord = self.font.render(f"{i*grid_size}m", True, self.config['visualization']['grid_color'])
                # 카메라 좌표를 명시적으로 전달하여 그리드에 고정
                x_pos = camera.world_to_screen(grid_x, 0)
                self.screen.blit(x_coord, (x_pos[0] - 20, x_pos[1] + 5))

        # 수평 그리드 라인 (Y축과 평행한 선)
        for i in range(grid_start_y, grid_end_y + 1):
            # 그리드 라인의 월드 좌표
            grid_y = i * grid_size

            # 화면 좌표로 변환하여 그리기 (카메라 위치 전달)
            start_h = camera.world_to_screen(world_left, grid_y)
            end_h = camera.world_to_screen(world_right, grid_y)
            pygame.draw.line(self.screen, self.config['visualization']['grid_color'], start_h, end_h, 1)

            # 좌표 표시 (원점 제외)
            if i != 0:
                y_coord = self.font.render(f"{i*grid_size}m", True, self.config['visualization']['grid_color'])
                # 카메라 좌표를 명시적으로 전달하여 그리드에 고정
                y_pos = camera.world_to_screen(0, grid_y)
                self.screen.blit(y_coord, (y_pos[0] + 5, y_pos[1] - 20))

        # 원점 표시
        origin = self.font.render("(0,0)", True, self.config['visualization']['grid_color'])
        # 카메라 좌표를 명시적으로 전달하여 그리드에 고정
        origin_pos = camera.world_to_screen(0, 0)
        self.screen.blit(origin, (origin_pos[0] + 5, origin_pos[1] + 5))

        # 그리드 교차점에 좌표 표시 (2칸 간격으로 표시하여 화면이 복잡해지지 않게 함)
        # 좀 더 작은 폰트를 사용하여 텍스트 크기 줄임
        small_font = pygame.font.Font(None, 18)

        # 교차점 좌표 표시
        for i in range(grid_start_x, grid_end_x + 1, 1):
            for j in range(grid_start_y, grid_end_y + 1, 1):
                # 원점은 이미 표시했으므로 제외
                if i == 0 or j == 0:
                    continue

                # 그리드 교차점 월드 좌표
                grid_x = i * grid_size
                grid_y = j * grid_size

                # 교차점 좌표 (x,y) 형식으로 렌더링
                coord_text = f"({int(grid_x)},{int(grid_y)})"
                coord_label = small_font.render(coord_text, True, self.config['visualization']['grid_color'])

                # 화면 좌표로 변환하여 표시
                coord_pos = camera.world_to_screen(grid_x, grid_y)
                # 텍스트가 교차점 주위에 적절하게 배치되도록 약간의 오프셋 적용
                self.screen.blit(coord_label, (coord_pos[0] + 5, coord_pos[1] - 15))

    def render(self, env, camera, hud):
        """
        환경 렌더링

        Args:
            env: 시뮬레이터 환경 객체
            camera: 카메라 객체
            hud: HUD 객체
        """
        # 렌더링 시작 시간
        render_start = time.time()

        # 배경 지우기
        self.screen.fill(self.config['visualization']['background_color'])

        # 현재 활성화된 차량 가져오기
        active_vehicle = env.vehicle_manager.get_active_vehicle()
        if active_vehicle is None:
            # 활성화된 차량이 없으면 렌더링 중지
            pygame.display.flip()
            return

        # 월드 좌표를 화면 좌표로 변환하는 함수를 생성
        if self.config['visualization']['camera_follow']:
            # 카메라 위치를 활성 차량 중심으로 설정
            world_to_screen = lambda x, y: camera.world_to_screen(x, y, vehicle=active_vehicle)
        else:
            # 카메라 위치를 카메라 객체의 위치로 설정
            world_to_screen = lambda x, y: camera.world_to_screen(x, y)

        # 전체 렌더링 설정이 켜져 있을 때만 렌더링을 수행
        if self.config['visualization']['visualize']:
            # 고정 그리드 그리기
            self.draw_grid(camera)

            # 도로 네트워크 그리기
            env.road_manager.draw(self.screen, world_to_screen, self.config['visualization']['debug_mode'])

            # 장애물 그리기
            env.obstacle_manager.draw(self.screen, world_to_screen, self.config['visualization']['debug_mode'])

            # 모든 차량 그리기
            env.vehicle_manager.draw_vehicles(self.screen, world_to_screen, self.config['visualization']['debug_mode'])

        # HUD 표시 (visualize_hud 설정 체크)
        if self.config['visualization']['visualize_hud']:
            active_vehicle_idx = env.vehicle_manager.get_active_vehicle_index()
            vehicle_count = env.vehicle_manager.get_vehicle_count()

            hud.draw_hud(self.screen, active_vehicle, active_vehicle_idx, vehicle_count, self._performance_metrics)

            if self.config['visualization']['debug_mode']:
                # 차량 목표 방향 화살표 그리기
                hud.draw_target_direction_arrows(self.screen, env.vehicle_manager.get_all_vehicles(), active_vehicle_idx, world_to_screen)

        # 렌더링 시간 측정
        self._performance_metrics['render_time'] = time.time() - render_start

        # 화면 업데이트
        pygame.display.flip()

        if not self.config['visualization']['training_mode']:
            self.clock.tick(self.config['simulation']['fps'])  # FPS 제한

        self.update_performance_metrics()

    def update_performance_metrics(self):
        """성능 측정 지표 업데이트"""
        current_time = time.time()
        frame_time = current_time - self._last_update_time
        self._frame_times.append(frame_time)
        self._last_update_time = current_time

        # FPS 계산 (이동 평균)
        if len(self._frame_times) > 0:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            self._performance_metrics['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    def get_performance_metrics(self):
        """성능 측정 지표 반환"""
        return self._performance_metrics

    def set_physics_time(self, physics_time):
        """물리 시뮬레이션 시간 설정"""
        self._performance_metrics['physics_time'] = physics_time
