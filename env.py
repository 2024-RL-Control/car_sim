# -*- coding: utf-8 -*-
import pygame
import gym
from gym import spaces
import numpy as np
import time
import pickle
import os
from collections import deque
from math import radians, degrees, pi, cos, sin
from config import VehicleConfig, SimConfig
from state import VehicleState
from physics import DynamicModel
from path_planning import PathPlanner

# ======================
# Simulation Environment
# ======================
class CarSimulatorEnv(gym.Env):
    """2D Top-Down 시점 차량 시뮬레이터(Gym) 환경"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        시뮬레이터 환경 초기화
        """
        # 설정 객체 초기화
        self.sim_config = SimConfig()
        self.vehicle_config = VehicleConfig()
        self.state = VehicleState()

        # 캐시 및 렌더링 최적화를 위한 변수
        self._track_marks = []          # 타이어 자국 [(x,y,width,color), ...]
        self._camera_offset = (0, 0)    # 카메라 오프셋 (팬/줌 기능용)
        self._camera_zoom = 1.0         # 카메라 줌 수준
        self._performance_metrics = {   # 성능 측정 데이터
            'fps': 0,
            'physics_time': 0,
            'render_time': 0
        }

        # 차량 상태 기록 (리플레이용)
        self._state_history = deque(maxlen=100)

        # Pygame 초기화
        self._init_pygame()

        # Gym 인터페이스
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),  # [가속, 조향]
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # 관측 공간 확장: [x, y, cos(yaw), sin(yaw), vel, vel_lateral, drift_angle, g_forces[0], g_forces[1]]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -1, -1, -20, -10, -np.pi, -5, -5]),
            high=np.array([np.inf, np.inf, 1, 1, 60, 10, np.pi, 5, 5]),
            dtype=np.float32
        )

        # 키보드 상태 (연속 입력 처리)
        self._keys_pressed = {
            'up': False,     # 가속
            'down': False,   # 제동
            'left': False,   # 좌회전
            'right': False,  # 우회전
            'brake': False,  # 수동 브레이크 (Space)
        }

        # 시간 관리 변수
        self._last_update_time = time.time()
        self._frame_times = deque(maxlen=60)  # 최근 60프레임 시간 (FPS 계산용)
    def set_global_path(self, start, goal, obstacles):
        """
        RRT + Dubins 기반 글로벌 경로 설정
        장애물과 경로를 한 번 변환하여 고정
        """
        print(f"📌 글로벌 경로 탐색 시작: Start {start} → Goal {goal}")

        self.obstacles = obstacles  # 장애물 저장 (월드 좌표)
        
        # 🔥 장애물 화면 좌표 변환하여 저장
        self.obstacle_screen_positions = [
            (self._world_to_screen(ox, oy), int(radius * self.sim_config.SCALE))
            for ox, oy, radius in obstacles
        ]

        planner = PathPlanner(start, goal, obstacles, x_range=(0, self.sim_config.SIM_WIDTH), y_range=(0, self.sim_config.SIM_HEIGHT))
        path = planner.plan()

        if path:
            print("✅ 경로 탐색 성공!")
            self.global_path = path

            # 🔥 경로를 한 번 변환하여 저장
            self.global_path_screen = [(self._world_to_screen(p[0], p[1])) for p in self.global_path]
        else:
            print("❌ 경로 탐색 실패! 장애물 회피 불가능")
            self.global_path = []
            self.global_path_screen = []
    
    def _draw_obstacles(self):
        """장애물을 화면에 그리는 함수 (자연스럽게 가려지는 효과)"""
        if not hasattr(self, 'obstacles') or not self.obstacles:
            print("⚠ 장애물 데이터 없음.")
            return  

        for ox, oy, radius in self.obstacles:
            # 🚀 월드 좌표 → 화면 좌표 변환 (출발점, 목표점과 동일한 방식)
            screen_x, screen_y = self._world_to_screen(ox, oy)
            screen_radius = max(3, int(radius * self.sim_config.SCALE * 0.5))  # 반지름 보정

            # ✅ 원의 경계 계산 (실제 그릴 수 있는 범위 확인)
            left_bound = screen_x - screen_radius
            right_bound = screen_x + screen_radius
            top_bound = screen_y - screen_radius
            bottom_bound = screen_y + screen_radius

            # ✅ 화면 범위 내에 있는 경우에만 원을 그림
            if right_bound > 0 and left_bound < self.sim_config.SIM_WIDTH and \
            bottom_bound > 0 and top_bound < self.sim_config.SIM_HEIGHT:

                # 🚀 원을 화면 크기 안에서만 그리도록 처리
                pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), screen_radius)




    def _init_pygame(self):
        """Pygame 초기화 및 그래픽 리소스 로드"""
        pygame.init()
        self.screen = pygame.display.set_mode(self.sim_config.SIM_SIZE)
        pygame.display.set_caption("Vehicle Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)  # 큰 글꼴 추가

        # 그래픽 리소스 로드
        self._load_assets()

        # 카메라 기본값 설정 - 자동 추적 활성화
        self.sim_config.CAMERA_FOLLOW = True

    def _load_assets(self):
        """차량 및 타이어 그래픽 리소스 생성"""
        # 차체
        car_length = self.vehicle_config.LENGTH * self.sim_config.SCALE
        car_width = self.vehicle_config.WIDTH * self.sim_config.SCALE
        self.car_surf = pygame.Surface((car_length, car_width), pygame.SRCALPHA)
        pygame.draw.rect(self.car_surf, self.sim_config.VEHICLE_COLOR,
                        (0, 0, car_length, car_width), 2, border_radius=int(car_width * 0.2))

        # 타이어
        tire_w = self.vehicle_config.TIRE_WIDTH * self.sim_config.SCALE
        tire_h = self.vehicle_config.TIRE_HEIGHT * self.sim_config.SCALE
        self.tire_surf = pygame.Surface((tire_h, tire_w), pygame.SRCALPHA)
        pygame.draw.rect(self.tire_surf, self.sim_config.TIRE_COLOR,
                        (0, 0, tire_h, tire_w), 2)

        # G-force 표시기
        self.g_force_surf = pygame.Surface((100, 100), pygame.SRCALPHA)

    def _world_to_screen(self, x, y):
        """
        월드 좌표 → 화면 좌표 변환
        월드: 원점 중앙, Y축 위로 증가
        화면: 원점 좌상단, Y축 아래로 증가

        항상 차량이 화면 중앙에 오도록 변환
        """
        # 차량의 현재 위치를 화면 중앙으로
        cam_x = self.state.x
        cam_y = self.state.y

        # 줌 처리
        scale = self.sim_config.SCALE * self._camera_zoom

        # 월드 좌표를 차량 기준으로 상대화
        rel_x = x - cam_x
        rel_y = y - cam_y

        # 스케일 적용 및 화면 중앙 기준으로 변환
        screen_x = rel_x * scale + self.sim_config.SIM_WIDTH/2
        screen_y = self.sim_config.SIM_HEIGHT/2 - rel_y * scale

        # 카메라 오프셋 적용 (수동 카메라 조작용)
        screen_x += self._camera_offset[0]
        screen_y += self._camera_offset[1]

        return (int(screen_x), int(screen_y))

    def _draw_grid(self):
        """고정 그리드 그리기"""
        scale = self.sim_config.SCALE * self._camera_zoom
        grid_size = 10 * self.sim_config.SCALE   # 그리드 단위 (100 픽셀) * 스케일

        # 카메라 위치 계산 (차량 중심)
        cam_x = self.state.x
        cam_y = self.state.y

        # 화면 영역(viewport)에서 보이는 월드 좌표 범위 계산
        screen_width = self.sim_config.SIM_WIDTH
        screen_height = self.sim_config.SIM_HEIGHT

        # 화면 경계에 해당하는 월드 좌표 계산
        world_left = cam_x - screen_width / (2 * scale)
        world_right = cam_x + screen_width / (2 * scale)
        world_top = cam_y + screen_height / (2 * scale)
        world_bottom = cam_y - screen_height / (2 * scale)

        # 화면에 표시될 그리드 라인의 시작/끝 인덱스 계산
        grid_start_x = int(world_left / grid_size) - 1
        grid_end_x = int(world_right / grid_size) + 1
        grid_start_y = int(world_bottom / grid_size) - 1
        grid_end_y = int(world_top / grid_size) + 1

        # 수직 그리드 라인 (X축과 평행한 선)
        for i in range(grid_start_x, grid_end_x + 1):
            # 그리드 라인의 월드 좌표
            grid_x = i * grid_size

            # 화면 좌표로 변환하여 그리기
            start_v = self._world_to_screen(grid_x, world_bottom)
            end_v = self._world_to_screen(grid_x, world_top)
            pygame.draw.line(self.screen, self.sim_config.GRID_COLOR, start_v, end_v, 1)

            # 좌표 표시 (원점 제외)
            if i != 0:
                x_coord = self.font.render(f"{i*grid_size}m", True, self.sim_config.GRID_COLOR)
                x_pos = self._world_to_screen(grid_x, 0)
                self.screen.blit(x_coord, (x_pos[0] - 20, x_pos[1] + 5))

        # 수평 그리드 라인 (Y축과 평행한 선)
        for i in range(grid_start_y, grid_end_y + 1):
            # 그리드 라인의 월드 좌표
            grid_y = i * grid_size

            # 화면 좌표로 변환하여 그리기
            start_h = self._world_to_screen(world_left, grid_y)
            end_h = self._world_to_screen(world_right, grid_y)
            pygame.draw.line(self.screen, self.sim_config.GRID_COLOR, start_h, end_h, 1)

            # 좌표 표시 (원점 제외)
            if i != 0:
                y_coord = self.font.render(f"{i*grid_size}m", True, self.sim_config.GRID_COLOR)
                y_pos = self._world_to_screen(0, grid_y)
                self.screen.blit(y_coord, (y_pos[0] + 5, y_pos[1] - 20))

        # 원점 표시
        origin = self.font.render("(0,0)", True, self.sim_config.GRID_COLOR)
        origin_pos = self._world_to_screen(0, 0)
        self.screen.blit(origin, (origin_pos[0] + 5, origin_pos[1] + 5))

    def _update_camera(self):
        """카메라 위치 업데이트 (시선 위치 조정)"""
        # 기본적으로 차량은 항상 화면 중심에 위치 (월드-스크린 변환에서 처리)

        if self.sim_config.CAMERA_FOLLOW:
            # 전방 주시 효과 (시선 약간 앞쪽으로)
            look_ahead_distance = min(self.state.vel * 0.5, 50)  # 속도에 비례하여 전방 주시

            if look_ahead_distance > 5:  # 일정 속도 이상일 때만 전방 주시
                # 전방 주시 방향으로 약간의 오프셋 적용
                target_offset_x = look_ahead_distance * np.cos(self.state.yaw) * self.sim_config.SCALE * self._camera_zoom * 0.3
                target_offset_y = -look_ahead_distance * np.sin(self.state.yaw) * self.sim_config.SCALE * self._camera_zoom * 0.3

                # 부드러운 카메라 이동
                smoothing = 0.05  # 낮을수록 부드럽게

                self._camera_offset = (
                    self._camera_offset[0] * (1 - smoothing) + target_offset_x * smoothing,
                    self._camera_offset[1] * (1 - smoothing) + target_offset_y * smoothing
                )
            else:
                # 저속에서는 서서히 중앙으로 복귀
                smoothing = 0.05
                self._camera_offset = (
                    self._camera_offset[0] * (1 - smoothing),
                    self._camera_offset[1] * (1 - smoothing)
                )

    def _draw_tire_marks(self):
        """타이어 자국 그리기"""
        if not self.sim_config.ENABLE_TRACK_MARKS:
            return

        # 드리프트 중일 때만 타이어 자국 추가
        if abs(self.state.drift_angle) > radians(5) and abs(self.state.vel) > 5.0:
            # 각 타이어 위치 계산
            wheelbase = self.vehicle_config.WHEELBASE
            track = self.vehicle_config.TRACK

            # 타이어 상대 위치 계산 (차량 좌표계 기준)
            tire_positions = [
                ( wheelbase/2,  track/2),  # Front Right
                ( wheelbase/2, -track/2),  # Front Left
                (-wheelbase/2,  track/2),  # Rear Right
                (-wheelbase/2, -track/2)   # Rear Left
            ]

            for dx, dy in tire_positions:
                # 차량 회전 변환
                rotated_x = dx * cos(self.state.yaw) - dy * sin(self.state.yaw)
                rotated_y = dx * sin(self.state.yaw) + dy * cos(self.state.yaw)

                # 월드 좌표 계산
                world_x = self.state.x + rotated_x
                world_y = self.state.y + rotated_y

                # 마크 너비는 드리프트 각도에 비례
                mark_width = max(1, min(5, abs(self.state.drift_angle) * 10))

                # 타이어 자국 추가
                self._track_marks.append((world_x, world_y, mark_width))

        # 타이어 자국 렌더링 (모든 자국)
        for x, y, width in self._track_marks:
            pos = self._world_to_screen(x, y)
            pygame.draw.circle(self.screen, self.sim_config.MARK_COLOR, pos, width)

        # 타이어 자국 개수 제한 (너무 많으면 성능 저하)
        max_marks = 2000
        if len(self._track_marks) > max_marks:
            self._track_marks = self._track_marks[-max_marks:]

    def _draw_tires(self):
        """4개의 타이어 렌더링 (아커만 조향 적용)"""
        wheelbase = self.vehicle_config.WHEELBASE
        track = self.vehicle_config.TRACK

        # 타이어 상대 위치 계산 (차량 좌표계 기준)
        tire_positions = [
            ( wheelbase/2,  track/2),  # Front Right
            ( wheelbase/2, -track/2),  # Front Left
            (-wheelbase/2,  track/2),  # Rear Right
            (-wheelbase/2, -track/2)   # Rear Left
        ]

        # 아커만 조향각 계산 (좌/우 앞바퀴 각도 차이 계산)
        steer = self.state.steer
        if abs(steer) > 0.001:  # 조향 중일 때만 아커만 계산
            # 회전 반경 계산 (자전거 모델 기준)
            R = wheelbase / np.tan(abs(steer))

            # 좌/우 조향각 계산 (아커만 공식)
            if steer > 0:  # 좌회전
                steer_inner = np.arctan(wheelbase / (R - track/2))  # 왼쪽 바퀴 (안쪽)
                steer_outer = np.arctan(wheelbase / (R + track/2))  # 오른쪽 바퀴 (바깥쪽)
            else:  # 우회전
                steer_inner = -np.arctan(wheelbase / (R - track/2))  # 오른쪽 바퀴 (안쪽)
                steer_outer = -np.arctan(wheelbase / (R + track/2))  # 왼쪽 바퀴 (바깥쪽)

            # 조향각 배열 [FR, FL, RR, RL]
            steer_angles = [
                steer_outer if steer < 0 else steer_inner,  # 우회전시 바깥쪽, 좌회전시 안쪽
                steer_inner if steer < 0 else steer_outer,  # 우회전시 안쪽, 좌회전시 바깥쪽
                0.0,  # 뒷바퀴는 조향 없음
                0.0   # 뒷바퀴는 조향 없음
            ]
        else:
            # 직진 시 모든 바퀴 조향각 0
            steer_angles = [0.0, 0.0, 0.0, 0.0]

        for i, (dx, dy) in enumerate(tire_positions):
            # 차량 회전 변환
            rotated_x = dx * cos(self.state.yaw) - dy * sin(self.state.yaw)
            rotated_y = dx * sin(self.state.yaw) + dy * cos(self.state.yaw)

            # 월드 좌표 계산
            world_x = self.state.x + rotated_x
            world_y = self.state.y + rotated_y

            # 타이어 각도 계산 (앞바퀴는 아커만 스티어링 적용)
            tire_angle = steer_angles[i]

            total_angle = self.state.yaw + tire_angle

            # 타이어 회전 및 위치 변환
            rotated_tire = pygame.transform.rotate(self.tire_surf, degrees(total_angle))
            tire_rect = rotated_tire.get_rect(center=self._world_to_screen(world_x, world_y))

            self.screen.blit(rotated_tire, tire_rect.topleft)

    def _draw_trajectory(self):
        """차량 궤적 그리기"""
        if len(self.state.trajectory) < 2:
            return

        # 지난 궤적을 점선으로 표시
        trajectory_points = [self._world_to_screen(x, y) for x, y in self.state.trajectory]

        # 부드러운 곡선 대신 선분으로 그림 (성능)
        if len(trajectory_points) > 1:
            trajectory_color = (0, 150, 255, 100)  # 반투명 파란색
            pygame.draw.lines(self.screen, trajectory_color, False, trajectory_points, 2)

    def _draw_g_force_meter(self, x, y):
        """G-force 미터 그리기"""
        meter_size = 80
        center = (x + meter_size // 2, y + meter_size // 2)
        radius = meter_size // 2 - 5

        # 배경
        bg_surf = pygame.Surface((meter_size, meter_size), pygame.SRCALPHA)
        pygame.draw.circle(bg_surf, (50, 50, 50, 150), (meter_size//2, meter_size//2), radius)

        # 눈금 그리기
        pygame.draw.circle(bg_surf, (100, 100, 100), (meter_size//2, meter_size//2), radius, 1)
        pygame.draw.circle(bg_surf, (100, 100, 100), (meter_size//2, meter_size//2), radius//2, 1)

        # 십자선
        pygame.draw.line(bg_surf, (100, 100, 100), (meter_size//2, 0), (meter_size//2, meter_size), 1)
        pygame.draw.line(bg_surf, (100, 100, 100), (0, meter_size//2), (meter_size, meter_size//2), 1)

        self.screen.blit(bg_surf, (x, y))

        # G-force 포인터 그리기
        g_lateral = self.state.g_forces[1]
        g_longitudinal = self.state.g_forces[0]

        # G-force를 픽셀로 변환 (1G = 반경의 1/3)
        scale_factor = radius / 3
        pointer_x = center[0] + g_lateral * scale_factor
        pointer_y = center[1] - g_longitudinal * scale_factor  # Y축 반전

        # G-force 범위 제한
        max_distance = radius - 2
        dx = pointer_x - center[0]
        dy = pointer_y - center[1]
        distance = (dx**2 + dy**2)**0.5

        if distance > max_distance:
            scale = max_distance / distance
            pointer_x = center[0] + dx * scale
            pointer_y = center[1] + dy * scale

        # 포인터 그리기
        pygame.draw.circle(self.screen, (255, 50, 50), (int(pointer_x), int(pointer_y)), 5)
        pygame.draw.line(self.screen, (200, 200, 200), center, (int(pointer_x), int(pointer_y)), 2)

        # G-force 값 표시
        g_text = self.font.render(f"{(g_longitudinal**2 + g_lateral**2)**0.5:.1f}G", True, (255, 255, 0))
        self.screen.blit(g_text, (x + 28, y + meter_size + 5))

    def _draw_hud(self):
        """차량 상태 HUD 표시"""
        # 아커만 조향각 계산 (좌/우 앞바퀴)
        wheelbase = self.vehicle_config.WHEELBASE
        track = self.vehicle_config.TRACK
        steer = self.state.steer

        left_steer = steer
        right_steer = steer

        if abs(steer) > 0.001:
            R = wheelbase / np.tan(abs(steer))
            if steer > 0:  # 좌회전
                left_steer = np.arctan(wheelbase / (R - track/2))
                right_steer = np.arctan(wheelbase / (R + track/2))
            else:  # 우회전
                right_steer = -np.arctan(wheelbase / (R - track/2))
                left_steer = -np.arctan(wheelbase / (R + track/2))

        # 기본 정보
        hud = [
            f"Velocity: {self.state.vel:.2f} m/s ({self.state.vel * 3.6:.1f} km/h)",
            f"Steering: {degrees(self.state.steer):.1f}°",
            f"Left wheel: {degrees(left_steer):.1f}°, Right wheel: {degrees(right_steer):.1f}°",
            f"Position: ({self.state.x:.1f}, {self.state.y:.1f})",
            f"Heading: {degrees(self.state.yaw):.1f}°",
            f"Accel: {self.state.accel:.2f} m/s²"
        ]

        # 디버그 정보 추가가
        if self.sim_config.ENABLE_DEBUG_INFO:
            hud.extend([
                f"Lateral Vel: {self.state.vel_lateral:.2f} m/s",
                f"Drift Angle: {degrees(self.state.drift_angle):.1f}°",
                f"G-Forces: Long {self.state.g_forces[0]:.2f}g, Lat {self.state.g_forces[1]:.2f}g",
                f"FPS: {self._performance_metrics['fps']:.1f}",
            ])

        # HUD 배경
        hud_width = 280
        hud_height = len(hud) * 25 + 10
        hud_surf = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        hud_surf.fill(self.sim_config.HUD_BG_COLOR)
        self.screen.blit(hud_surf, (10, 10))

        # HUD 텍스트
        y_pos = 15
        for line in hud:
            text = self.font.render(line, True, self.sim_config.HUD_FG_COLOR)
            self.screen.blit(text, (20, y_pos))
            y_pos += 25

        # G-force 미터
        self._draw_g_force_meter(self.sim_config.SIM_WIDTH - 90, 20)

        # 큰 속도계
        speed_kmh = self.state.vel * 3.6
        speed_text = self.large_font.render(f"{speed_kmh:.0f}", True, (255, 255, 255))
        kmh_text = self.font.render("km/h", True, (200, 200, 200))

        # 중앙 상단에 위치
        speed_rect = speed_text.get_rect(center=(self.sim_config.SIM_WIDTH // 2, 40))
        kmh_rect = kmh_text.get_rect(center=(self.sim_config.SIM_WIDTH // 2, 65))

        self.screen.blit(speed_text, speed_rect)
        self.screen.blit(kmh_text, kmh_rect)

    def _handle_keyboard_input(self):
        """키보드 입력 처리 (상태 업데이트)"""
        pressed = pygame.key.get_pressed()

        # 메인 키 상태 업데이트
        self._keys_pressed['up'] = pressed[pygame.K_UP] or pressed[pygame.K_w]
        self._keys_pressed['down'] = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
        self._keys_pressed['left'] = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
        self._keys_pressed['right'] = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
        self._keys_pressed['brake'] = pressed[pygame.K_SPACE]

        # 카메라 컨트롤
        if pressed[pygame.K_PLUS] or pressed[pygame.K_EQUALS]:
            self._camera_zoom = min(2.0, self._camera_zoom * 1.05)
        if pressed[pygame.K_MINUS]:
            self._camera_zoom = max(0.5, self._camera_zoom / 1.05)

        # 카메라 리셋
        if pressed[pygame.K_r]:
            self._camera_zoom = 1.0
            self._camera_offset = (0, 0)

        # 카메라 팬
        pan_speed = 5
        if pressed[pygame.K_i]:  # 위로 이동
            self._camera_offset = (self._camera_offset[0], self._camera_offset[1] + pan_speed)
        if pressed[pygame.K_k]:  # 아래로 이동
            self._camera_offset = (self._camera_offset[0], self._camera_offset[1] - pan_speed)
        if pressed[pygame.K_j]:  # 왼쪽으로 이동
            self._camera_offset = (self._camera_offset[0] + pan_speed, self._camera_offset[1])
        if pressed[pygame.K_l]:  # 오른쪽으로 이동
            self._camera_offset = (self._camera_offset[0] - pan_speed, self._camera_offset[1])

        # 시뮬레이션 설정 토글
        if pressed[pygame.K_t]:  # 타이어 자국 토글
            self.sim_config.ENABLE_TRACK_MARKS = not self.sim_config.ENABLE_TRACK_MARKS

        if pressed[pygame.K_c]:  # 카메라 추적 토글
            self.sim_config.CAMERA_FOLLOW = not self.sim_config.CAMERA_FOLLOW

        if pressed[pygame.K_h]:  # 디버그 정보 토글
            self.sim_config.ENABLE_DEBUG_INFO = not self.sim_config.ENABLE_DEBUG_INFO

        if pressed[pygame.K_f]:  # 모든 자국 지우기
            self._track_marks = []

        # 저장 및 로드
        if pressed[pygame.K_F5]:  # 상태 저장
            self._save_state()

        if pressed[pygame.K_F9]:  # 상태 로드
            self._load_state()

        # 입력에서 액션 결정
        action = np.zeros(2)

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
            action[1] = 1.0  # 좌회전
        if self._keys_pressed['right']:
            action[1] = -1.0  # 우회전

        return action

    def _save_state(self):
        """현재 시뮬레이션 상태 저장"""
        save_data = {
            'state': self.state,
            'track_marks': self._track_marks,
            'vehicle_config': self.vehicle_config
        }

        try:
            os.makedirs('saves', exist_ok=True)
            with open(f'saves/sim_state_{time.strftime("%Y%m%d_%H%M%S")}.pickle', 'wb') as f:
                pickle.dump(save_data, f)
            print("시뮬레이션 상태가 저장되었습니다.")
        except Exception as e:
            print(f"저장 오류: {e}")

    def _load_state(self):
        """시뮬레이션 상태 불러오기"""
        try:
            save_files = sorted([f for f in os.listdir('saves') if f.startswith('sim_state_')])
            if not save_files:
                print("저장된 파일이 없습니다.")
                return

            # 가장 최근 파일 로드
            latest_file = save_files[-1]
            with open(f'saves/{latest_file}', 'rb') as f:
                save_data = pickle.load(f)

            self.state = save_data['state']
            self._track_marks = save_data['track_marks']
            self.vehicle_config = save_data['vehicle_config']

            # 그래픽 리소스 다시 로드
            self._load_assets()

            print(f"시뮬레이션 상태를 불러왔습니다: {latest_file}")
        except Exception as e:
            print(f"로드 오류: {e}")

    def _update_performance_metrics(self):
        """성능 측정 지표 업데이트"""
        current_time = time.time()
        frame_time = current_time - self._last_update_time
        self._frame_times.append(frame_time)
        self._last_update_time = current_time

        # FPS 계산 (이동 평균)
        avg_frame_time = sum(self._frame_times) / len(self._frame_times)
        self._performance_metrics['fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    def step(self, action):
        """
        환경 스텝 실행

        Args:
            action: [가속도, 조향] 명령 [-1, 1]

        Returns:
            obs: 관측 [x, y, cos(yaw), sin(yaw), vel, vel_lateral, drift_angle, g_forces[0], g_forces[1]]
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        # 프레임 시간 계산
        current_time = time.time()
        dt = current_time - self._last_update_time
        dt = min(dt, 0.1)  # 최대 타임스텝 제한 (너무 긴 프레임 방지)

        # 물리 시뮬레이션 시작시간
        physics_start = time.time()

        # 물리 모델 적용
        DynamicModel.apply_forces(self.state, action, dt, self.sim_config, self.vehicle_config)

        # 물리 시뮬레이션 시간 측정
        self._performance_metrics['physics_time'] = time.time() - physics_start

        # 관측 반환
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = False

        # 상태 기록 (리플레이용)
        self._state_history.append(self.state.__dict__.copy())

        info = {

        }

        # 성능 측정 업데이트
        self._update_performance_metrics()

        return obs, reward, done, info

    def reset(self):
        """환경 초기화"""
        self.state = VehicleState()
        self._track_marks = []
        self._state_history = []
        self._camera_offset = (0, 0)
        self._camera_zoom = 1.0
        return self._get_obs()

    def render(self, mode='human'):
        """환경 렌더링"""
        # 렌더링 시작 시간
        render_start = time.time()

        # 카메라 업데이트
        self._update_camera()

        # 배경 지우기
        self.screen.fill(self.sim_config.BACKGROUND_COLOR)

        # 고정 그리드 그리기
        self._draw_grid()

        # 🔴 장애물 시각화 추가 김형선 코드 추가
        self._draw_obstacles()

        # 타이어 자국 그리기
        self._draw_tire_marks()

        # 차량 궤적 그리기
        self._draw_trajectory()

        # ✅ 글로벌 경로 그리기 (초록색 선)
        if self.global_path:
            for i in range(len(self.global_path) - 1):
                start = self._world_to_screen(self.global_path[i][0], self.global_path[i][1])
                end = self._world_to_screen(self.global_path[i+1][0], self.global_path[i+1][1])
                pygame.draw.line(self.screen, (0, 255, 0), start, end, 2)

            # ✅ 출발 지점 표시 (파란색 점)
            start_screen = self._world_to_screen(self.global_path[0][0], self.global_path[0][1])
            pygame.draw.circle(self.screen, (0, 0, 255), start_screen, 8)  # 반지름 8

            # ✅ 목표 지점 표시 (빨간색 점)
            goal_screen = self._world_to_screen(self.global_path[-1][0], self.global_path[-1][1])
            pygame.draw.circle(self.screen, (255, 0, 0), goal_screen, 8)  # 반지름 8

            # ✅ 목표 위치와 실제 도착지 차이 계산
            actual_x, actual_y,_ = self.global_path[-1]  # 실제 마지막 경로 지점
            goal_x, goal_y = (1000,100)  # 사용자가 입력한 목표 지점

            error_distance = ((actual_x - goal_x) ** 2 + (actual_y - goal_y) ** 2) ** 0.5

            # 🔹 HUD에 차이 표시
            error_text = self.font.render(f"Goal Error: {error_distance:.2f}m", True, (255, 255, 0))
            self.screen.blit(error_text, (20, 70))  # 화면 좌측 상단에 표시


        # 차체 렌더링
        rotated_surf = pygame.transform.rotate(self.car_surf, degrees(self.state.yaw))
        rect = rotated_surf.get_rect(center=self._world_to_screen(self.state.x, self.state.y))
        self.screen.blit(rotated_surf, rect.topleft)

        # 타이어 렌더링
        self._draw_tires()

        # HUD 표시
        self._draw_hud()

        # 렌더링 시간 측정
        self._performance_metrics['render_time'] = time.time() - render_start

        # 화면 업데이트
        pygame.display.flip()
        self.clock.tick(self.sim_config.FPS)  # FPS 제한




    def close(self):
        """환경 종료"""
        pygame.quit()

    def _get_obs(self):
        """관측 정보 반환 [x, y, cos(yaw), sin(yaw), vel, vel_lateral, drift_angle, g_forces[0], g_forces[1]]"""
        cos_yaw, sin_yaw = self.state.encoding_angle(self.state.yaw)
        return np.array([
            self.state.x,
            self.state.y,
            cos_yaw,
            sin_yaw,
            self.state.vel,
            self.state.vel_lateral,
            self.state.drift_angle,
            self.state.g_forces[0],
            self.state.g_forces[1]
        ])

    def _calculate_reward(self):
        """보상 함수 (드리프트 및 속도 기반)"""
        # 속도 보상 (최대 속도에 가까울수록 높은 보상)
        speed_reward = self.state.vel / self.vehicle_config.MAX_SPEED

        # 종합 보상
        reward = 0.6 * speed_reward

        return reward
