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
from .config import VehicleConfig, SimConfig
from .vehicle import Vehicle
from .object import ObstacleManager, GoalManager

# ======================
# Simulation Environment
# ======================
class CarSimulatorEnv(gym.Env):
    """2D Top-Down 시점 차량 시뮬레이터(Gym) 환경"""
    metadata = {'render.modes': ['human']}

    def __init__(self, multi_vehicle=False, num_vehicles=1, config_path=None):
        """
        시뮬레이터 환경 초기화

        Args:
            multi_vehicle: 다중 차량 모드 여부
            num_vehicles: 차량 수 (multi_vehicle=True일 때만 유효)
            config_path: 설정 파일 경로 (None이면 기본값 사용)
        """
        # 설정 객체 초기화
        if config_path:
            sim_config_path = os.path.join(config_path, 'sim_config.json')
            vehicle_config_path = os.path.join(config_path, 'vehicle_config.json')
            self.sim_config = SimConfig.from_json(sim_config_path)
            self.vehicle_config = VehicleConfig.from_json(vehicle_config_path)
        else:
            self.sim_config = SimConfig()
            self.vehicle_config = VehicleConfig()

        # 다중 차량 모드 설정
        self.multi_vehicle = multi_vehicle
        self.num_vehicles = max(1, num_vehicles if multi_vehicle else 1)

        # 차량 리스트 생성
        self.vehicles = []
        for i in range(self.num_vehicles):
            vehicle = Vehicle(vehicle_id=i, vehicle_config=self.vehicle_config, sim_config=self.sim_config)
            self.vehicles.append(vehicle)

        # 주 차량 설정
        self.active_vehicle_idx = 0
        self.vehicle = self.vehicles[self.active_vehicle_idx]

        # 장애물 매니저 초기화
        self.obstacle_manager = ObstacleManager()

        # 목적지 매니저 초기화
        self.goal_manager = GoalManager()

        # 카메라 관련 변수
        self._camera_offset = (0, 0)    # 카메라 오프셋 (팬/줌 기능용)
        self._camera_zoom = 1.0         # 카메라 줌 수준

        # 그리드 렌더링 최적화용 캐시
        self._grid_cache = None
        self._grid_cache_params = None

        # 성능 측정 변수
        self._performance_metrics = {   # 성능 측정 데이터
            'fps': 0,
            'physics_time': 0,
            'render_time': 0
        }

        # 차량 상태 기록 (리플레이용)
        self._state_history = deque(maxlen=100)

        # 시간 관리 변수
        self._last_update_time = time.time()
        self._frame_times = deque(maxlen=60)  # 최근 60프레임 시간 (FPS 계산용)

        # 키보드 상태 (연속 입력 처리)
        self._keys_pressed = {
            'up': False,     # 가속
            'down': False,   # 제동
            'left': False,   # 좌회전
            'right': False,  # 우회전
            'brake': False,  # 수동 브레이크 (Space)
        }

        # Pygame 초기화
        self._init_pygame()

        # Gym 인터페이스
        if not multi_vehicle:
            # 단일 차량 모드
            self.action_space = spaces.Box(
                low=np.array([-1, -1]),  # [가속, 조향]
                high=np.array([1, 1]),
                dtype=np.float32
            )

            # 관측 공간: [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, g_forces[0], g_forces[1], distance_to_target, yaw_diff_to_target]
            self.observation_space = spaces.Box(
                low=np.array([-np.inf, -np.inf, -1, -1, -20, -10, -5, -5, 0, -np.pi]),
                high=np.array([np.inf, np.inf, 1, 1, 60, 10, 5, 5, np.inf, np.pi]),
                dtype=np.float32
            )
        else:
            # 다중 차량 모드 - 각 차량마다 별도 액션
            self.action_space = spaces.Tuple([
                spaces.Box(
                    low=np.array([-1, -1]),  # [가속, 조향]
                    high=np.array([1, 1]),
                    dtype=np.float32
                ) for _ in range(self.num_vehicles)
            ])

            # 다중 차량 관측 공간 - 각 차량마다 별도 관측
            self.observation_space = spaces.Tuple([
                spaces.Box(
                    low=np.array([-np.inf, -np.inf, -1, -1, -20, -10, -5, -5, 0, -np.pi]),
                    high=np.array([np.inf, np.inf, 1, 1, 60, 10, 5, 5, np.inf, np.pi]),
                    dtype=np.float32
                ) for _ in range(self.num_vehicles)
            ])

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
        """그래픽 리소스 생성"""
        # G-force 표시기
        self.g_force_surf = pygame.Surface((100, 100), pygame.SRCALPHA)

    def _world_to_screen(self, x, y, cam_x=None, cam_y=None):
        """
        월드 좌표 → 화면 좌표 변환
        월드: 원점 중앙, Y축 위로 증가
        화면: 원점 좌상단, Y축 아래로 증가

        항상 현재 활성 차량이 화면 중앙에 오도록 변환

        Args:
            x, y: 변환할 월드 좌표
            cam_x, cam_y: 카메라 위치 (None이면 활성 차량 위치 사용)
        """
        # 카메라 위치가 주어지지 않은 경우 활성 차량 위치 사용
        if cam_x is None or cam_y is None:
            cam_x = self.vehicle.state.x
            cam_y = self.vehicle.state.y

        # 줌 처리
        scale = self.sim_config.SCALE * self._camera_zoom

        # 월드 좌표를 카메라 기준으로 상대화
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
        scale = self.sim_config.SCALE
        grid_size = 10 * scale   # 그리드 단위 (픽셀 수) * 스케일

        # 카메라 위치 계산 (활성 차량 중심)
        cam_x = self.vehicle.state.x
        cam_y = self.vehicle.state.y

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

            # 화면 좌표로 변환하여 그리기 (카메라 위치 전달)
            start_v = self._world_to_screen(grid_x, world_bottom, cam_x, cam_y)
            end_v = self._world_to_screen(grid_x, world_top, cam_x, cam_y)
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

            # 화면 좌표로 변환하여 그리기 (카메라 위치 전달)
            start_h = self._world_to_screen(world_left, grid_y, cam_x, cam_y)
            end_h = self._world_to_screen(world_right, grid_y, cam_x, cam_y)
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
        g_lateral = self.vehicle.state.g_forces[1]
        g_longitudinal = self.vehicle.state.g_forces[0]

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
        state = self.vehicle.state
        config = self.vehicle.config

        # 아커만 조향각 계산 (좌/우 앞바퀴)
        wheelbase = config.WHEELBASE
        track = config.TRACK
        steer = state.steer

        if abs(steer) > 0.001:
            R = wheelbase / np.tan(abs(steer))
            if steer < 0:  # 좌회전
                left_steer = -np.arctan(wheelbase / (R - track/2))
                right_steer = -np.arctan(wheelbase / (R + track/2))
            else:  # 우회전
                left_steer = np.arctan(wheelbase / (R + track/2))
                right_steer = np.arctan(wheelbase / (R - track/2))
        else:
            left_steer = right_steer = 0.0

        # 기본 정보
        hud = [
            f"Vehicle {self.active_vehicle_idx + 1}/{self.num_vehicles}",
            f"Position: ({state.x:.1f}, {state.y:.1f})",
            f"Heading: {np.degrees(state.yaw):.1f}°",
            f"Velocity: {state.vel_long:.2f} m/s ({state.vel_long * 3.6:.1f} km/h)"
        ]

        # 목적지 정보 추가
        goal = self.goal_manager.get_vehicle_goal(self.vehicle.get_id())
        if goal:
            distance = state.distance_to_target
            yaw_diff = np.degrees(state.yaw_diff_to_target)
            hud.extend([
                f"Target: ({state.target_x:.1f}, {state.target_y:.1f})",
                f"Distance to target: {distance:.2f}m",
                f"Heading diff: {yaw_diff:.1f}°"
            ])

        # 디버그 정보 추가
        if self.sim_config.ENABLE_DEBUG_INFO:
            hud.extend([
                f"Throttling: {state.throttle:.2f} m/s²",
                f"Longitude Accel: {state.acc_long:.2f} m/s²",
                f"Steering: {np.degrees(state.steer):.1f}°",
                f"Left wheel: {np.degrees(left_steer):.1f}°, Right wheel: {np.degrees(right_steer):.1f}°",
                f"Lateral Vel: {state.vel_lat:.2f} m/s",
                f"Lateral Accel: {state.acc_lat:.2f} m/s²",
                f"G-Forces: Long {state.g_forces[0]:.2f}g, Lat {state.g_forces[1]:.2f}g",
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
        speed_kmh = state.vel_long * 3.6
        speed_text = self.large_font.render(f"{speed_kmh:.0f}", True, (255, 255, 255))
        kmh_text = self.font.render("km/h", True, (200, 200, 200))

        # 중앙 상단에 위치
        speed_rect = speed_text.get_rect(center=(self.sim_config.SIM_WIDTH // 2, 40))
        kmh_rect = kmh_text.get_rect(center=(self.sim_config.SIM_WIDTH // 2, 65))

        self.screen.blit(speed_text, speed_rect)
        self.screen.blit(kmh_text, kmh_rect)

    def _draw_target_direction_arrows(self):
        """각 차량의 목표 방향 화살표 그리기"""
        for vehicle in self.vehicles:
            goal = self.goal_manager.get_vehicle_goal(vehicle.id)
            if goal:
                # 차량 위치와 목표 위치를 화면 좌표로 변환
                vehicle_pos = self._world_to_screen(vehicle.state.x, vehicle.state.y)
                target_pos = self._world_to_screen(vehicle.state.target_x, vehicle.state.target_y)

                # 화살표 그리기 (차량에서 목표로)
                # 화살표 색상은 활성 차량이면 밝은 노란색, 아니면 어두운 노란색
                arrow_color = (255, 255, 0) if vehicle.id == self.vehicles[self.active_vehicle_idx].id else (180, 180, 0)
                pygame.draw.line(self.screen, arrow_color, vehicle_pos, target_pos, 2)

                # 화살표 머리 그리기
                # 화살표 방향 계산
                dx = target_pos[0] - vehicle_pos[0]
                dy = target_pos[1] - vehicle_pos[1]
                length = np.sqrt(dx*dx + dy*dy)

                if length > 10:  # 일정 거리 이상일 때만 화살표 머리 그리기
                    # 단위 벡터
                    dx, dy = dx/length, dy/length

                    # 화살표 머리 크기
                    head_length = 10

                    # 화살표 머리 끝점 좌표
                    arrow_head1_x = target_pos[0] - head_length * (dx*0.866 + dy*0.5)
                    arrow_head1_y = target_pos[1] - head_length * (-dx*0.5 + dy*0.866)
                    arrow_head2_x = target_pos[0] - head_length * (dx*0.866 - dy*0.5)
                    arrow_head2_y = target_pos[1] - head_length * (dx*0.5 + dy*0.866)

                    # 화살표 머리 그리기
                    pygame.draw.line(self.screen, arrow_color, target_pos, (int(arrow_head1_x), int(arrow_head1_y)), 2)
                    pygame.draw.line(self.screen, arrow_color, target_pos, (int(arrow_head2_x), int(arrow_head2_y)), 2)

    def handle_keyboard_input(self):
        """키보드 입력 처리 (상태 업데이트)"""
        pressed = pygame.key.get_pressed()

        # 메인 키 상태 업데이트
        self._keys_pressed['up'] = pressed[pygame.K_UP] or pressed[pygame.K_w]
        self._keys_pressed['down'] = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
        self._keys_pressed['left'] = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
        self._keys_pressed['right'] = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
        self._keys_pressed['brake'] = pressed[pygame.K_SPACE]

        # 차량 전환 (다중 차량 모드일 때만)
        if self.multi_vehicle and pressed[pygame.K_TAB]:
            # 다음 차량으로 전환
            self.active_vehicle_idx = (self.active_vehicle_idx + 1) % self.num_vehicles
            self.vehicle = self.vehicles[self.active_vehicle_idx]
            print(f"현재 차량: {self.active_vehicle_idx + 1}/{self.num_vehicles}")

        # 카메라 컨트롤
        if pressed[pygame.K_PLUS] or pressed[pygame.K_EQUALS]:
            self._camera_zoom = min(5.0, self._camera_zoom * 1.05)
        if pressed[pygame.K_MINUS]:
            self._camera_zoom = max(0.25, self._camera_zoom / 1.05)

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
        if pressed[pygame.K_c]:  # 카메라 추적 토글
            self.sim_config.CAMERA_FOLLOW = not self.sim_config.CAMERA_FOLLOW

        if pressed[pygame.K_h]:  # 디버그 정보 토글
            self.sim_config.ENABLE_DEBUG_INFO = not self.sim_config.ENABLE_DEBUG_INFO

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
            action[1] = -1.0  # 좌회전
        if self._keys_pressed['right']:
            action[1] = 1.0  # 우회전

        # 다중 차량 모드에서는 활성 차량만 입력으로 제어
        if self.multi_vehicle:
            actions = [np.zeros(2) for _ in range(self.num_vehicles)]
            actions[self.active_vehicle_idx] = action
            return actions
        else:
            return action

    def get_obstacle_manager(self):
        """장애물 관리자 객체 반환"""
        return self.obstacle_manager

    def get_goal_manager(self):
        """목적지 관리자 객체 반환"""
        return self.goal_manager

    def _save_state(self):
        """현재 시뮬레이션 상태 저장"""
        # 차량 상태 및 설정 저장
        save_data = {
            'states': [v.state for v in self.vehicles],
            'vehicle_configs': [v.config for v in self.vehicles],
            # 목적지 매니저 관련 데이터
            'goals': {gid: {
                'x': goal.x,
                'y': goal.y,
                'yaw': goal.yaw,
                'radius': goal.radius,
                'color': goal.color
            } for gid, goal in self.goal_manager.goals.items()},
            'vehicle_goals': self.goal_manager.vehicle_goals,
            'next_goal_id': self.goal_manager.next_goal_id
        }

        try:
            os.makedirs('saves', exist_ok=True)
            with open(f'saves/sim_state_{time.strftime("%Y%m%d_%H%M%S")}.pickle', 'wb') as f:
                pickle.dump(save_data, f)
            print("시뮬레이션 상태가 저장되었습니다.")
        except Exception as e:
            print(f"저장 오류: {e}")

    def _load_state(self):
        """시뮬레이션 상태 불러오기 (add_goal_with_id 메소드 활용)"""
        try:
            save_files = sorted([f for f in os.listdir('saves') if f.startswith('sim_state_')])
            if not save_files:
                print("저장된 파일이 없습니다.")
                return

            # 가장 최근 파일 로드
            latest_file = save_files[-1]
            with open(f'saves/{latest_file}', 'rb') as f:
                save_data = pickle.load(f)

            # 차량 상태 복원
            states = save_data.get('states', [])
            vehicle_configs = save_data.get('vehicle_configs', [])

            # 불러온 상태 정보에 맞게 차량 수 조정
            self.num_vehicles = len(states)
            self.multi_vehicle = self.num_vehicles > 1

            # 차량 리스트 재생성
            self.vehicles = []
            for i in range(self.num_vehicles):
                vehicle = Vehicle(
                    vehicle_id=i,
                    vehicle_config=vehicle_configs[i] if i < len(vehicle_configs) else self.vehicle_config,
                    sim_config=self.sim_config
                )
                if i < len(states):
                    vehicle.state = states[i]
                self.vehicles.append(vehicle)

            # 현재 활성 차량 설정
            self.active_vehicle_idx = min(self.active_vehicle_idx, self.num_vehicles - 1)
            self.vehicle = self.vehicles[self.active_vehicle_idx]

            # 목적지 정보 복원
            self.goal_manager.clear_goals()  # 기존 목적지 삭제

            # 목적지 객체 복원 (원래 ID 유지)
            saved_goals = save_data.get('goals', {})
            for goal_id_str, goal_data in saved_goals.items():
                # 문자열 키를 정수로 변환 (pickle로 저장할 때 키가 문자열로 변환될 수 있음)
                goal_id = int(goal_id_str) if isinstance(goal_id_str, str) else goal_id_str

                # add_goal_with_id 메소드로 원래 ID 유지하며 목적지 추가
                self.goal_manager.add_goal_with_id(
                    goal_id=goal_id,
                    x=goal_data['x'],
                    y=goal_data['y'],
                    yaw=goal_data['yaw'],
                    radius=goal_data['radius'],
                    color=goal_data['color']
                )

            # 차량-목적지 매핑 복원 (ID 변환 불필요)
            self.goal_manager.vehicle_goals = save_data.get('vehicle_goals', {})

            # 각 차량에 목적지 정보 전달
            for vehicle_id, goal_id in self.goal_manager.vehicle_goals.items():
                # vehicle_id가 문자열로 저장되었을 수 있음
                v_id = int(vehicle_id) if isinstance(vehicle_id, str) else vehicle_id

                if v_id < len(self.vehicles):
                    vehicle = self.vehicles[v_id]
                    vehicle.goal_id = goal_id
                    goal = self.goal_manager.goals.get(goal_id)
                    if goal:
                        vehicle.update_target(goal.x, goal.y, goal.yaw)

            # 그래픽 리소스 다시 로드
            for vehicle in self.vehicles:
                vehicle._load_graphics()

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
            action: 단일 차량 모드: [가속도, 조향] 명령 [-1, 1]
                    다중 차량 모드: 차량별 [가속도, 조향] 명령 리스트

        Returns:
            obs: 관측 [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, g_forces[0], g_forces[1], distance_to_target, yaw_diff_to_target]
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

        # 장애물 업데이트
        self.obstacle_manager.update(dt)

        # 차량 업데이트 및 충돌 감지
        collisions = {}
        reached_targets = {}

        if self.multi_vehicle:
            # 다중 차량 모드 - action은 차량별 액션 리스트
            for i, vehicle in enumerate(self.vehicles):
                vehicle_action = action[i] if i < len(action) else np.zeros(2)
                vehicle.step(vehicle_action, dt)

                # 충돌 감지
                collision = vehicle.check_collision(self.obstacle_manager)
                collisions[vehicle.id] = collision

                # 목적지 도달 확인
                goal = self.goal_manager.get_vehicle_goal(vehicle.id)
                if goal:
                    reached = vehicle.check_target_reached()
                    reached_targets[vehicle.id] = reached
        else:
            # 단일 차량 모드 - action은 단일 차량 액션
            # 현재 차량만 업데이트
            self.vehicle.step(action, dt)

            # 충돌 감지
            collision = self.vehicle.check_collision(self.obstacle_manager)
            collisions[self.vehicle.id] = collision

            # 목적지 도달 확인
            goal = self.goal_manager.get_vehicle_goal(self.vehicle.id)
            if goal:
                reached = self.vehicle.check_target_reached()
                reached_targets[self.vehicle.id] = reached

        # 물리 시뮬레이션 시간 측정
        self._performance_metrics['physics_time'] = time.time() - physics_start

        # 충돌 여부 확인인
        any_collision = any(collisions.values())

        # 관측 반환
        obs = self._get_obs()
        reward = self._calculate_reward(collisions, reached_targets)
        done = False

        info = {
            'collisions': collisions,
            'reached_targets': reached_targets,
            'collision': any_collision
        }

        # 충돌 처리
        if any_collision:
            # 충돌로 인한 종료 상태 설정 (보상은 이미 계산됨)
            done = True

        # 상태 기록 (리플레이용)
        self._state_history.append({v.id: v.state.__dict__.copy() for v in self.vehicles})

        # 성능 측정 업데이트
        self._update_performance_metrics()

        return obs, reward, done, info

    def reset(self):
        """환경 초기화"""
        # 차량 초기화
        for vehicle in self.vehicles:
            vehicle.reset()

        # 상태 기록 초기화
        self._state_history.clear()

        # 카메라 설정 초기화
        self._camera_offset = (0, 0)
        self._camera_zoom = 1.0

        # 장애물 관리자 초기화 (모든 장애물 제거)
        self.obstacle_manager.clear_obstacles()

        # 목적지 관리자 초기화
        self.goal_manager.clear_goals()

        return self._get_obs()

    def render(self, mode='human'):
        """환경 렌더링"""
        # 렌더링 시작 시간
        render_start = time.time()

        # 배경 지우기
        self.screen.fill(self.sim_config.BACKGROUND_COLOR)

        # 고정 그리드 그리기
        self._draw_grid()

        # 목적지 그리기
        self.goal_manager.draw(self.screen, self._world_to_screen, self.sim_config.ENABLE_DEBUG_INFO)

        # 장애물 그리기
        self.obstacle_manager.draw(self.screen, self._world_to_screen, self.sim_config.ENABLE_DEBUG_INFO)

        # 모든 차량 그리기
        for vehicle in self.vehicles:
            vehicle.draw(self.screen, self._world_to_screen, self._camera_zoom)

        # HUD 표시
        self._draw_hud()

        # 차량 목표 방향 화살표 그리기
        self._draw_target_direction_arrows()

        # 렌더링 시간 측정
        self._performance_metrics['render_time'] = time.time() - render_start

        # 화면 업데이트
        pygame.display.flip()
        self.clock.tick(self.sim_config.FPS)  # FPS 제한

    def close(self):
        """환경 종료"""
        pygame.quit()

    def add_goal_for_vehicle(self, vehicle_id, x, y, yaw=0.0, radius=1.0, color=(0, 255, 0)):
        """
        차량에 목적지 추가

        Args:
            vehicle_id: 차량 ID
            x: 목적지 X 좌표
            y: 목적지 Y 좌표
            yaw: 목적지 방향
            radius: 목적지 반경
            color: 목적지 색상 (RGB)

        Returns:
            추가된 목적지 ID
        """
        # 목적지 추가
        goal_id = self.goal_manager.add_goal(x, y, yaw, radius, color)

        # 목적지 관리 정보 추가
        self.goal_manager.assign_goal_to_vehicle(vehicle_id, goal_id)

        # 차량에 목적지 추가
        if 0 <= vehicle_id < len(self.vehicles):
            vehicle = self.vehicles[vehicle_id]
            vehicle.set_goal(goal_id, self.goal_manager)

        return goal_id

    def _get_obs(self):
        """
        관측 정보 반환 [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, g_forces[0], g_forces[1], distance_to_target, yaw_diff_to_target]
        단일 차량: 단일 차량 상태
        다중 차량: 차량별 상태 리스트
        """
        if self.multi_vehicle:
            # 다중 차량 모드 - 차량별 관측 리스트 반환
            return [self._get_vehicle_obs(vehicle) for vehicle in self.vehicles]
        else:
            # 단일 차량 모드 - 현재 차량 관측만 반환
            return self._get_vehicle_obs(self.vehicle)

    def _get_vehicle_obs(self, vehicle):
        """단일 차량에 대한 관측 정보 반환"""
        state = vehicle.state
        cos_yaw, sin_yaw = state.encoding_angle(state.yaw)

        # 기본 차량 상태
        obs = np.array([
            state.x,
            state.y,
            cos_yaw,
            sin_yaw,
            state.vel_long,
            state.vel_lat,
            state.g_forces[0],
            state.g_forces[1],
            state.distance_to_target,
            state.yaw_diff_to_target
        ])

        return obs

    def _calculate_reward(self, collisions, reached_targets):
        """
        보상 함수 계산
        다중 차량 모드: 차량별 보상 리스트 반환
        단일 차량 모드: 단일 값 반환

        Args:
            collisions: 차량별 충돌 여부 {vehicle_id: bool}
            reached_targets: 차량별 목표 도달 여부 {vehicle_id: bool}
        """
        if self.multi_vehicle:
            # 다중 차량 모드 - 차량별 보상 계산
            rewards = []
            for vehicle in self.vehicles:
                reward = self._calculate_vehicle_reward(
                    vehicle,
                    collisions.get(vehicle.id, False),
                    reached_targets.get(vehicle.id, False)
                )
                rewards.append(reward)
            return rewards
        else:
            # 단일 차량 모드 - 현재 차량 보상만 계산
            return self._calculate_vehicle_reward(
                self.vehicle,
                collisions.get(self.vehicle.id, False),
                reached_targets.get(self.vehicle.id, False)
            )

    def _calculate_vehicle_reward(self, vehicle, collision, reached_target):
        """
        단일 차량에 대한 보상 계산

        Args:
            vehicle: 차량 객체
            collision: 충돌 여부
            reached_target: 목표 도달 여부
        """
        state = vehicle.state

        # 속도 보상 (최대 속도에 가까울수록 높은 보상)
        speed_reward = state.vel_long / vehicle.config.MAX_SPEED * 0.2

        # 목표 관련 보상
        goal_reward = 0
        if self.goal_manager.get_vehicle_goal(vehicle.id):
            # 목표 거리에 따른 보상 (가까울수록 높은 보상)
            if state.distance_to_target > 0:
                proximity_reward = 1.0 / (1.0 + state.distance_to_target) * 0.1
                goal_reward += proximity_reward

            # 방향 일치 보상 (차량이 목표를 바라볼수록 높은 보상)
            direction_reward = (1.0 - abs(state.yaw_diff_to_target) / np.pi) * 0.2
            goal_reward += direction_reward

            # 목표 도달 보상
            if reached_target:
                goal_reward += 1.0

        # 종합 보상
        reward = speed_reward + goal_reward

        # 충돌 패널티
        if collision:
            reward -= 2.0

        return reward
