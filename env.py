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
from vehicle import Vehicle
from object import ObstacleManager

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

        # Vehicle 클래스 생성
        self.vehicle = Vehicle(self.vehicle_config, self.sim_config)

        # 장애물 매니저 초기화
        self.obstacle_manager = ObstacleManager()

        # 카메라 관련 변수
        self._camera_offset = (0, 0)    # 카메라 오프셋 (팬/줌 기능용)
        self._camera_zoom = 1.0         # 카메라 줌 수준

        # 성능 측정 변수
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

        # 충돌 관련 변수
        self._collision_state = False

        # 시간 관리 변수
        self._last_update_time = time.time()
        self._frame_times = deque(maxlen=60)  # 최근 60프레임 시간 (FPS 계산용)

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

    def _world_to_screen(self, x, y):
        """
        월드 좌표 → 화면 좌표 변환
        월드: 원점 중앙, Y축 위로 증가
        화면: 원점 좌상단, Y축 아래로 증가

        항상 차량이 화면 중앙에 오도록 변환
        """
        # 차량의 현재 위치를 화면 중앙으로
        cam_x = self.vehicle.state.x
        cam_y = self.vehicle.state.y

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

    def _update_camera(self):
        """카메라 위치 업데이트 (시선 위치 조정)"""
        # 기본적으로 차량은 항상 화면 중심에 위치 (월드-스크린 변환에서 처리)
        if self.sim_config.CAMERA_FOLLOW:
            # 전방 주시 효과 (시선 약간 앞쪽으로)
            look_ahead_distance = min(self.vehicle.state.vel * 0.5, 50)  # 속도에 비례하여 전방 주시

            if look_ahead_distance > 5:  # 일정 속도 이상일 때만 전방 주시
                # 전방 주시 방향으로 약간의 오프셋 적용
                target_offset_x = look_ahead_distance * np.cos(self.vehicle.state.yaw) * self.sim_config.SCALE * self._camera_zoom * 0.3
                target_offset_y = -look_ahead_distance * np.sin(self.vehicle.state.yaw) * self.sim_config.SCALE * self._camera_zoom * 0.3

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

    def _draw_grid(self):
        """고정 그리드 그리기"""
        scale = self.sim_config.SCALE * self._camera_zoom
        grid_size = 10 * self.sim_config.SCALE   # 그리드 단위 (100 픽셀) * 스케일

        # 카메라 위치 계산 (차량 중심)
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

        left_steer = steer
        right_steer = steer

        if abs(steer) > 0.001:
            R = wheelbase / np.tan(abs(steer))
            if steer < 0:  # 좌회전
                left_steer = -np.arctan(wheelbase / (R - track/2))
                right_steer = -np.arctan(wheelbase / (R + track/2))
            else:  # 우회전
                left_steer = np.arctan(wheelbase / (R + track/2))
                right_steer = np.arctan(wheelbase / (R - track/2))

        # 기본 정보
        hud = [
            f"Velocity: {state.vel:.2f} m/s ({state.vel * 3.6:.1f} km/h)",
            f"Steering: {np.degrees(state.steer):.1f}°",
            f"Left wheel: {np.degrees(left_steer):.1f}°, Right wheel: {np.degrees(right_steer):.1f}°",
            f"Position: ({state.x:.1f}, {state.y:.1f})",
            f"Heading: {np.degrees(state.yaw):.1f}°",
            f"Accel: {state.accel:.2f} m/s²"
        ]

        # 디버그 정보 추가
        if self.sim_config.ENABLE_DEBUG_INFO:
            hud.extend([
                f"Lateral Vel: {state.vel_lateral:.2f} m/s",
                f"Drift Angle: {np.degrees(state.drift_angle):.1f}°",
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
        speed_kmh = state.vel * 3.6
        speed_text = self.large_font.render(f"{speed_kmh:.0f}", True, (255, 255, 255))
        kmh_text = self.font.render("km/h", True, (200, 200, 200))

        # 중앙 상단에 위치
        speed_rect = speed_text.get_rect(center=(self.sim_config.SIM_WIDTH // 2, 40))
        kmh_rect = kmh_text.get_rect(center=(self.sim_config.SIM_WIDTH // 2, 65))

        self.screen.blit(speed_text, speed_rect)
        self.screen.blit(kmh_text, kmh_rect)

    def handle_keyboard_input(self):
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
            self.vehicle.clear_track_marks()

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

        return action

    def get_obstacle_manager(self):
        """장애물 관리자 객체 반환"""
        return self.obstacle_manager

    def _save_state(self):
        """현재 시뮬레이션 상태 저장"""
        save_data = {
            'state': self.vehicle.state,
            'track_marks': self.vehicle.get_track_marks(),
            'vehicle_config': self.vehicle.config
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

            self.vehicle.state = save_data['state']
            self.vehicle.set_track_marks(save_data['track_marks'])
            self.vehicle.config = save_data['vehicle_config']

            # 그래픽 리소스 다시 로드
            self.vehicle._load_graphics()

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

        # 장애물 업데이트
        self.obstacle_manager.update(dt)

        # 차량 업데이트
        self.vehicle.step(action, dt)

        # 충돌 감지
        collision = self.vehicle.check_collision(self.obstacle_manager)

        # 물리 시뮬레이션 시간 측정
        self._performance_metrics['physics_time'] = time.time() - physics_start

        # 관측 반환
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = False

        info = {}

        # 충돌 처리
        if collision:
            # 충돌 상태 설정
            self._collision_state = True

            # 충돌로 인한 보상 감소 및 종료 상태 설정
            reward -= 50.0
            done = True
            info['collision'] = True

        # 상태 기록 (리플레이용)
        self._state_history.append(self.vehicle.state.__dict__.copy())

        # 성능 측정 업데이트
        self._update_performance_metrics()

        return obs, reward, done, info

    def reset(self):
        """환경 초기화"""
        self.vehicle.reset()
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

        # 장애물 그리기
        self.obstacle_manager.draw(self.screen, self._world_to_screen, self.sim_config.ENABLE_DEBUG_INFO)

        # 차량 및 타이어 자국 그리기
        self.vehicle.draw(self.screen, self._world_to_screen)

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
        state = self.vehicle.state
        cos_yaw, sin_yaw = state.encoding_angle(state.yaw)
        return np.array([
            state.x,
            state.y,
            cos_yaw,
            sin_yaw,
            state.vel,
            state.vel_lateral,
            state.drift_angle,
            state.g_forces[0],
            state.g_forces[1]
        ])

    def _calculate_reward(self):
        """보상 함수 (드리프트 및 속도 기반)"""
        # 속도 보상 (최대 속도에 가까울수록 높은 보상)
        speed_reward = self.vehicle.state.vel / self.vehicle.config.MAX_SPEED

        # 종합 보상
        reward = 0.6 * speed_reward

        return reward
