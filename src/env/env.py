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
from ..model.vehicle import Vehicle
from ..model.object import ObstacleManager, GoalManager
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

        # 차량 리스트 생성
        self.vehicles = []
        for i in range(self.num_vehicles):
            vehicle = Vehicle(vehicle_id=i, vehicle_config=self.config['vehicle'], physics_config=self.config['physics'], visual_config=self.config['visualization'])
            self.vehicles.append(vehicle)

        # 주 차량 설정
        self.active_vehicle_idx = 0
        self.vehicle = self.vehicles[self.active_vehicle_idx]

        # 장애물 매니저 초기화
        self.obstacle_manager = ObstacleManager(bounding_circle_colors=self.config['visualization']['bounding_circle_color'])

        # 목적지 매니저 초기화
        self.goal_manager = GoalManager(bounding_circle_colors=self.config['visualization']['bounding_circle_color'])

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
        # 렌더러와 HUD 초기화
        self.renderer = Renderer(self.config)
        self.screen, self.clock = self.renderer.init_pygame()
        self.hud = HUD(self.config)

    def _world_to_screen(self, x, y, cam_x=None, cam_y=None):
        """카메라의 world_to_screen 메서드를 호출하는 래퍼 함수"""
        return self.camera.world_to_screen(x, y, cam_x, cam_y, self.vehicle)

    def handle_keyboard_input(self):
        """키보드 입력 처리 (상태 업데이트)"""
        return self.keyboard_handler.handle_keyboard_input(self, self.camera)

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
            # 차량 설정 (config 속성 대신 개별 설정 저장)
            'vehicle_settings': [{
                'vehicle_config': v.vehicle_config,
                'physics_config': v.physics_config,
                'visual_config': v.visual_config,
                'id': v.id
            } for v in self.vehicles],
            'active_vehicle_idx': self.active_vehicle_idx,

            # 목적지 매니저 관련 데이터
            'goals': self.goal_manager.get_serializable_goals(),

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

            # 차량 상태 복원
            states = save_data.get('states', [])
            vehicle_settings = save_data.get('vehicle_settings', [])

            # 불러온 상태 정보에 맞게 차량 수 조정
            self.num_vehicles = len(states)
            self.multi_vehicle = (self.num_vehicles > 1)

            # 차량 리스트 재생성
            self.vehicles = []
            for i in range(self.num_vehicles):
                # 저장된 설정이 있으면 사용, 없으면 기본 설정 사용
                vehicle_config = self.config['vehicle']
                physics_config = self.config['physics']
                visual_config = self.config['visualization']

                if i < len(vehicle_settings):
                    vs = vehicle_settings[i]
                    vehicle_config = vs.get('vehicle_config', vehicle_config)
                    physics_config = vs.get('physics_config', physics_config)
                    visual_config = vs.get('visual_config', visual_config)
                    vehicle_id = vs.get('id', i)
                else:
                    vehicle_id = i

                vehicle = Vehicle(
                    vehicle_id=vehicle_id,
                    vehicle_config=vehicle_config,
                    physics_config=physics_config,
                    visual_config=visual_config
                )

                if i < len(states):
                    vehicle.state = states[i]

                self.vehicles.append(vehicle)

            # 현재 활성 차량 설정
            self.active_vehicle_idx = save_data.get('active_vehicle_idx', 0)
            self.active_vehicle_idx = min(self.active_vehicle_idx, self.num_vehicles - 1)
            self.vehicle = self.vehicles[self.active_vehicle_idx]

            # 장애물 정보 복원
            self.obstacle_manager.clear_obstacles()  # 기존 장애물 제거
            if 'obstacles' in save_data:
                self.obstacle_manager.load_obstacles_from_serialized(save_data['obstacles'])

            # 목적지 정보 복원
            if 'goals' in save_data:
                self.goal_manager.load_from_serialized(save_data['goals'])

                # 각 차량에 목적지 정보 전달
                for vehicle_id_str, goal_id in self.goal_manager.vehicle_goals.items():
                    # vehicle_id가 문자열로 저장되었을 수 있음
                    vehicle_id = int(vehicle_id_str) if isinstance(vehicle_id_str, str) else vehicle_id_str

                    # 해당하는 차량 찾기
                    vehicle = None
                    for v in self.vehicles:
                        if v.id == vehicle_id:
                            vehicle = v
                            break

                    if vehicle:
                        vehicle.goal_id = goal_id
                        goal = self.goal_manager.goals.get(goal_id)
                        if goal:
                            vehicle.update_target(goal.x, goal.y, goal.yaw)

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
            obs: 관측 [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, g_forces[0], g_forces[1], distance_to_target, yaw_diff_to_target]
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

        # 장애물 업데이트
        self.obstacle_manager.update(dt)

        # 차량 업데이트 및 충돌 감지
        collisions = {}
        reached_targets = {}

        self._time_elapsed += dt

        if self.multi_vehicle:
            # 다중 차량 모드 - action은 차량별 액션 리스트
            for i, vehicle in enumerate(self.vehicles):
                vehicle_action = action[i] if i < len(action) else np.zeros(2)
                _, collision, reached = vehicle.step(vehicle_action, dt, self._time_elapsed, self.obstacle_manager, self.goal_manager, self.vehicles)

                collisions[vehicle.id] = collision
                reached_targets[vehicle.id] = reached
        else:
            # 단일 차량 모드 - action은 단일 차량 액션
            # 현재 차량만 업데이트
            _, collision, reached = self.vehicle.step(action, dt, self._time_elapsed, self.obstacle_manager, self.goal_manager, self.vehicles)

            # 충돌 감지
            collisions[self.vehicle.id] = collision
            reached_targets[self.vehicle.id] = reached

        # 물리 시뮬레이션 시간 측정
        physics_time = time.time() - physics_start
        if self.renderer:
            self.renderer.set_physics_time(physics_time)

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

        # 시간 업데이트
        self._last_update_time = current_time

        return obs, reward, done, info

    def reset(self):
        """환경 초기화"""
        # 차량 초기화
        for i, vehicle in enumerate(self.vehicles):
            vehicle.reset()
            vehicle.set_position(i*100, 0)  # 차량 초기 위치 설정 (X축으로 간격 두고 배치)

        # 상태 기록 초기화
        self._state_history.clear()

        # 카메라 설정 초기화
        self.camera.reset()

        # 키보드 핸들러 초기화
        self.keyboard_handler.reset()

        # 장애물 관리자 초기화 (모든 장애물 제거)
        self.obstacle_manager.clear_obstacles()

        # 목적지 관리자 초기화
        self.goal_manager.clear_goals()

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
        state = vehicle.get_state()
        sensor_manager = vehicle.get_sensor_manager()
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
        speed_reward = state.vel_long / vehicle.vehicle_config['max_speed'] * 0.2

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
