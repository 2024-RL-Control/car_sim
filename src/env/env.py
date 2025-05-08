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
from ..model.vehicle import VehicleManager
from ..model.object import ObstacleManager
from ..model.road import RoadNetworkManager
from ..utils.config_utils import load_config
from ..ui.camera import Camera
from ..ui.keyboard import KeyboardHandler
from ..ui.hud import HUD
from ..ui.renderer import Renderer

# ======================
# Simulation Environment
# ======================
class CarSimulatorEnv(gym.Env):
    """2D Top-Down 시점 차량 시뮬레이터(Gym) 환경"""
    metadata = {'render.modes': ['human', 'rgb_array'],}

    def __init__(self, config_path: str = None):
        """
        시뮬레이터 환경 초기화

        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        super(CarSimulatorEnv, self).__init__()

        # config.yaml에서 설정 로드
        self.config = load_config(config_path)

        # 다중 차량 모드 설정
        self.num_vehicles = self.config['simulation']['num_vehicles']
        self.multi_vehicle = (self.num_vehicles > 1)

        # 차량 관리자 초기화
        self.vehicle_manager = VehicleManager(
            vehicle_config=self.config['vehicle'],
            physics_config=self.config['physics'],
            visual_config=self.config['visualization']
        )

        # 초기 차량들 생성
        for i in range(self.num_vehicles):
            self.vehicle_manager.create_vehicle(x=i*100, y=0, vehicle_id=i)

        # 장애물 매니저 초기화
        self.obstacle_manager = ObstacleManager(bounding_circle_colors=self.config['visualization']['bounding_circle_color'])

        # 도로 시스템 초기화
        self.road_manager = RoadNetworkManager(self.config['simulation']['path_planning'])

        # UI 모듈 초기화
        self.camera = Camera(self.config)
        self.keyboard_handler = KeyboardHandler(self.config)
        self.renderer = None
        self.hud = None

        # 차량 상태 기록 (리플레이용)
        self._state_history = deque(maxlen=100)

        # 시간 관리 변수
        self._last_update_time = time.time()
        self._time_elapsed = 0.0  # 경과 시간 (초 단위)

        # Pygame 초기화
        self._init_pygame()

        # Gym 인터페이스
        if not self.multi_vehicle:
            # 단일 차량 모드
            self.action_space = spaces.Box(
                low=np.array([ 0.0, 0.0, -1.0]),   # 엔진[0,1], 브레이크[0,1], 조향[-1,1]
                high=np.array([1.0, 1.0,  1.0]),
                dtype=np.float32
            )

            # 관측 공간: [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, distance_to_target, yaw_diff_to_target]
            self.observation_space = spaces.Box(
                low=np.array([-np.inf, -np.inf, -1, -1, -20, -25, 0, -np.pi]),
                high=np.array([np.inf, np.inf, 1, 1, 65, 25, np.inf, np.pi]),
                dtype=np.float32
            )
        else:
            # 다중 차량 모드 - 각 차량마다 별도 액션
            self.action_space = spaces.Tuple([
                spaces.Box(
                    low=np.array([ 0.0, 0.0, -1.0]),   # 엔진[0,1], 브레이크[0,1], 조향[-1,1]
                    high=np.array([1.0, 1.0,  1.0]),
                    dtype=np.float32
                ) for _ in range(self.num_vehicles)
            ])

            # 다중 차량 관측 공간 - 각 차량마다 별도 관측
            self.observation_space = spaces.Tuple([
                spaces.Box(
                    low=np.array([-np.inf, -np.inf, -1, -1, -20, -25, 0, -np.pi]),
                    high=np.array([np.inf, np.inf, 1, 1, 65, 25, np.inf, np.pi]),
                    dtype=np.float32
                ) for _ in range(self.num_vehicles)
            ])

    def _init_pygame(self):
        """Pygame 초기화 및 그래픽 리소스 로드"""
        # 렌더러와 HUD 초기화
        self.renderer = Renderer(self.config)
        self.screen, self.clock = self.renderer.init_pygame()
        self.hud = HUD(self.config)

    def _world_to_screen(self, x, y, cam_x=None, cam_y=None):
        """카메라의 world_to_screen 메서드를 호출하는 래퍼 함수"""
        active_vehicle = self.vehicle_manager.get_active_vehicle()
        return self.camera.world_to_screen(x, y, cam_x, cam_y, active_vehicle)

    def handle_keyboard_input(self):
        """키보드 입력 처리 (상태 업데이트)"""
        return self.keyboard_handler.handle_keyboard_input(self, self.camera)

    def get_obstacle_manager(self):
        """장애물 관리자 객체 반환"""
        return self.obstacle_manager

    def _save_state(self):
        """현재 시뮬레이션 상태 저장"""
        # 차량 상태 및 설정 저장
        save_data = {
            'vehicles': self.vehicle_manager.get_serializable_state(),

            # 장애물 매니저 관련 데이터
            'obstacles': self.obstacle_manager.get_serializable_obstacles(),

            # 카메라 설정
            'camera': self.camera.get_serializable_camera(),

            # 시간 관련 정보
            'time_elapsed': self._time_elapsed
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

            # 차량 관리자 상태 복원
            if 'vehicles' in save_data:
                self.vehicle_manager.load_from_serialized(save_data['vehicles'])

                # 다중 차량 모드 설정 업데이트
                self.num_vehicles = self.vehicle_manager.get_vehicle_count()
                self.multi_vehicle = (self.num_vehicles > 1)

            # 장애물 정보 복원
            self.obstacle_manager.clear_obstacles()  # 기존 장애물 제거
            if 'obstacles' in save_data:
                self.obstacle_manager.load_obstacles_from_serialized(save_data['obstacles'])

            # 카메라 설정 복원
            if 'camera' in save_data:
                self.camera.load_from_serialized(save_data['camera'])

            # 시간 관련 정보 복원
            if 'time_elapsed' in save_data:
                self._time_elapsed = save_data['time_elapsed']

            print(f"시뮬레이션 상태를 불러왔습니다: {latest_file}")
        except Exception as e:
            print(f"로드 오류: {e}")
            import traceback
            traceback.print_exc()

    def step(self, action):
        """
        환경 스텝 실행

        Args:
            action: 단일 차량 모드: [가속도, 조향] 명령 [-1, 1]
                    다중 차량 모드: 차량별 [가속도, 조향] 명령 리스트

        Returns:
            obs: 관측 [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, distance_to_target, yaw_diff_to_target]
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        # 프레임 시간 계산
        current_time = time.time()
        dt = current_time - self._last_update_time
        dt = max(min(dt, 0.1), 1e-4)  # 최소 0.0001초, 최대 0.1초

        # 물리 시뮬레이션 시작시간
        physics_start = time.time()

        # 장애물 업데이트 후 외접원 반환
        obstacles = self.obstacle_manager.update(dt)

        # 시간 계산
        self._time_elapsed += dt

        # 차량 업데이트 및 충돌 감지
        collisions, reached_targets = self.vehicle_manager.step(
            action, dt, self._time_elapsed, obstacles
        )

        # 관측값 및 보상 계산
        obs = self._get_obs()
        reward = self._calculate_rewards(collisions, reached_targets)

        # 물리 시뮬레이션 시간 측정
        physics_time = time.time() - physics_start
        if self.renderer:
            self.renderer.set_physics_time(physics_time)

        # 충돌 여부 확인
        any_collision = any(collisions.values())

        # 종료 여부 결정
        done = any_collision

        info = {
            'collisions': collisions,
            'reached_targets': reached_targets,
            'collision': any_collision
        }

        # 상태 기록 (리플레이용)
        vehicle_states = {}
        for vehicle in self.vehicle_manager.get_all_vehicles():
            vehicle_states[vehicle.id] = vehicle.state.__dict__.copy()
        self._state_history.append(vehicle_states)

        # 시간 업데이트
        self._last_update_time = current_time

        return obs, reward, done, info

    def reset(self):
        """환경 초기화"""
        # 차량 초기화
        self.vehicle_manager.reset_vehicle()

        # 각 차량 위치 설정 (X축으로 간격 두고 배치)
        for i, vehicle in enumerate(self.vehicle_manager.get_all_vehicles()):
            vehicle.set_position(i*100, 0)

        # 상태 기록 초기화
        self._state_history.clear()

        # 카메라 설정 초기화
        self.camera.reset()

        # 키보드 핸들러 초기화
        self.keyboard_handler.reset()

        # 장애물 관리자 초기화 (모든 장애물 제거)
        self.obstacle_manager.clear_obstacles()

        # 도로 시스템 초기화
        self.road_manager.reset()

        return self._get_obs()

    def render(self, mode='human'):
        """환경 렌더링"""
        if self.renderer:
            self.renderer.render(self, self.camera, self.hud)

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
        bool = self.vehicle_manager.add_goal_for_vehicle(vehicle_id, x, y, yaw, radius, color)
        if bool:
            vehicle = self.vehicle_manager.get_vehicle_by_id(vehicle_id)
            objects = []
            obstacles = self.obstacle_manager.get_all_outer_circles()
            if obstacles:
                objects.extend(obstacles)
            if self.vehicle_manager.get_vehicle_count() > 0:
                vehicles = self.vehicle_manager.get_all_vehicles()
                for v in vehicles:
                    if v.id != vehicle_id:
                        objects.extend(v.get_outer_circle())
            self.road_manager.connect(vehicle.get_position(), (x, y, yaw), objects)

        return bool

    def _get_obs(self):
        """
        모든 차량의 관측값 반환

        Returns:
            관측값 리스트 (단일 차량이면 단일 관측값)
        """
        if not self.multi_vehicle:
            # 단일 차량 모드
            if len(self.vehicle_manager.get_all_vehicles()) > 0:
                return self._get_vehicle_observation(self.vehicle_manager.get_all_vehicles()[0])
            return np.zeros(self.observation_space.shape)
        else:
            # 다중 차량 모드
            return [self._get_vehicle_observation(vehicle) for vehicle in self.vehicle_manager.get_all_vehicles()]

    def _get_vehicle_observation(self, vehicle):
        """
        단일 차량 관측값 계산

        Args:
            vehicle: 차량 객체

        Returns:
            관측값 (numpy 배열)
        """
        state = vehicle.get_state()
        cos_yaw, sin_yaw = state.encoding_angle(state.yaw)

        # 기본 차량 상태
        obs = np.array([
            state.x,
            state.y,
            cos_yaw,
            sin_yaw,
            state.vel_long,
            state.vel_lat,
            state.distance_to_target,
            state.yaw_diff_to_target
        ], dtype=np.float32)

        return obs

    def _calculate_rewards(self, collisions, reached_targets):
        """
        모든 차량의 보상 계산

        Args:
            collisions: 차량별 충돌 여부 {vehicle_id: bool}
            reached_targets: 차량별 목표 도달 여부 {vehicle_id: bool}

        Returns:
            보상 리스트 (단일 차량이면 단일 보상)
        """
        if not self.multi_vehicle:
            # 단일 차량 모드
            if len(self.vehicle_manager.get_all_vehicles()) > 0:
                vehicle = self.vehicle_manager.get_all_vehicles()[0]
                return self._calculate_vehicle_reward(
                    vehicle,
                    collisions.get(vehicle.id, False),
                    reached_targets.get(vehicle.id, False)
                )
            return 0.0
        else:
            # 다중 차량 모드
            return [
                self._calculate_vehicle_reward(
                    vehicle,
                    collisions.get(vehicle.id, False),
                    reached_targets.get(vehicle.id, False)
                )
                for vehicle in self.vehicle_manager.get_all_vehicles()
            ]

    def _calculate_vehicle_reward(self, vehicle, collision, reached_target):
        """
        단일 차량 보상 계산

        Args:
            vehicle: 차량 객체
            collision: 충돌 여부
            reached_target: 목표 도달 여부

        Returns:
            계산된 보상값
        """
        state = vehicle.get_state()

        # 속도 보상 (최대 속도에 가까울수록 높은 보상)
        speed_reward = state.vel_long / self.config['vehicle']['max_speed'] * 0.2

        # 목표 관련 보상
        goal_reward = 0
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
