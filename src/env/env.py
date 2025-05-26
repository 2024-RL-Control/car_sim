# -*- coding: utf-8 -*-
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pickle
import os
from collections import deque
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

    def __init__(self, config_path: str = None, setup: bool = True):
        """
        시뮬레이터 환경 초기화

        Args:
            config_path: Path to the configuration file. If None, uses default.
        """
        super(CarSimulatorEnv, self).__init__()

        # config.yaml에서 설정 로드
        self.config = load_config(config_path)

        self.max_vel_long = self.config['vehicle']['max_speed']
        self.min_vel_long = self.config['vehicle']['min_speed']
        self.max_acc_long = self.config['vehicle']['max_accel']
        self.min_acc_long = self.config['vehicle']['max_brake']
        self.max_vel_lat = self.config['vehicle']['max_vel_lat']
        self.max_acc_lat = self.config['vehicle']['max_acc_lat']

        # 다중 차량 모드 설정
        self.num_vehicles = self.config['simulation']['num_vehicles']

        # 장애물 매니저 초기화
        self.obstacle_manager = ObstacleManager(bounding_circle_colors=self.config['visualization']['bounding_circle_color'])

        # 도로 시스템 초기화
        self.road_manager = RoadNetworkManager(self.config['simulation']['path_planning'])

        # 차량 관리자 초기화
        self.vehicle_manager = VehicleManager(
            road_manager=self.road_manager,
            vehicle_config=self.config['vehicle'],
            physics_config=self.config['physics'],
            visual_config=self.config['visualization']
        )

        # 초기 차량들 생성
        self.setup = setup
        if self.setup:
            for i in range(self.num_vehicles):
                self.vehicle_manager.create_vehicle(x=i*100, y=0, vehicle_id=i)

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

        # 각 차량의 조작 공간
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),   # 엔진[0,1], 브레이크[0,1], 조향[-1,1]
            high=np.array([1.0, 1.0,  1.0]),
            dtype=np.float64
        )

        # 각 차량의 관측 공간, 모든 관측 값이 [-1, 1] 또는 [0, 1] 범위로 정규화됨
        obs_dim = 91  # 13(기본상태) + 36(LIDAR) + 42(궤적)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float64
        )

    def _init_pygame(self):
        """Pygame 초기화 및 그래픽 리소스 로드"""
        # 렌더러와 HUD 초기화
        self.renderer = Renderer(self.config)
        self.screen, self.clock = self.renderer.init_pygame()
        self.hud = HUD(self.config)

    def _world_to_screen(self, x, y, vehicle=None):
        """카메라의 world_to_screen 메서드를 호출하는 래퍼 함수"""
        return self.camera.world_to_screen(x, y, vehicle)

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

            # 도로 네트워크 관련 데이터
            'road_network': self.road_manager.get_serializable_state(),

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

                # 차량 수 업데이트
                self.num_vehicles = self.vehicle_manager.get_vehicle_count()

            # 장애물 정보 복원
            self.obstacle_manager.clear_obstacles()  # 기존 장애물 제거
            if 'obstacles' in save_data:
                self.obstacle_manager.load_obstacles_from_serialized(save_data['obstacles'])

            # 장애물 정보 복원
            obstacles = self.obstacle_manager.get_all_outer_circles()

            # 도로 네트워크 정보 복원
            if 'road_network' in save_data:
                self.road_manager.load_from_serialized(save_data['road_network'], obstacles)

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

    def step(self, actions):
        """
        환경 스텝 실행

        Args:
            actions: 차량별 [엔진, 브레이크, 조향] 명령 (액션 리스트, 2차원 배열)

        Returns:
            observations: 차량별 관측 (num_vehicles, obs_dim) [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, distance_to_target, yaw_diff_to_target]
            rewards: 차량별 보상 (num_vehicles,)
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
        collisions, outside_roads, reached_targets, dones = self.vehicle_manager.step(
            actions, dt, self._time_elapsed, obstacles
        )

        # 관측값 및 보상 계산
        observations = self._get_obs()
        rewards = self._calculate_rewards(collisions, outside_roads, reached_targets)

        # 물리 시뮬레이션 시간 측정
        physics_time = time.time() - physics_start
        if self.renderer:
            self.renderer.set_physics_time(physics_time)

        # 종료 여부 결정, 모든 차량이 종료 상태라면 종료
        done = all(dones.values())

        info = {
            'collisions': collisions,
            'outside_roads': outside_roads,
            'reached_targets': reached_targets,
            'dones': dones,
            'done': done
        }

        # 상태 기록 (리플레이용)
        vehicle_states = {}
        for vehicle in self.vehicle_manager.get_all_vehicles():
            vehicle_states[vehicle.id] = vehicle.state.__dict__.copy()
        self._state_history.append(vehicle_states)

        # 시간 업데이트
        self._last_update_time = current_time

        return observations, rewards, done, False, info

    def reset(self, *, seed=None, options=None):
        """환경 초기화

        Args:
            seed: 환경의 난수 생성기 시드
            options: 추가 옵션 딕셔너리

        Returns:
            observation: 초기 관측값
            info: 추가 정보 딕셔너리
        """
        # 시드가 제공되면 설정
        if seed is not None:
            np.random.seed(seed)

        # 차량 초기화
        self.vehicle_manager.reset_vehicle()

        # 각 차량 위치 설정 (X축으로 간격 두고 배치)
        if self.setup:
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
        goal_id = self.vehicle_manager.add_goal_for_vehicle(vehicle_id, x, y, yaw, radius, color)
        if goal_id is not None:
            vehicle = self.vehicle_manager.get_vehicle_by_id(vehicle_id)
            road = self.add_road(vehicle, x, y, yaw)
            return road
        else:
            return goal_id

    def add_road(self, vehicle, x, y, yaw):
        """
        차량에 도로 추가

        Args:
            vehicle: 차량 객체
            x: 도로 X 좌표
            y: 도로 Y 좌표
            yaw: 도로 방향

        Returns:
            도로 추가 여부(Boolean)
        """
        vehicle_id = vehicle.get_id()
        objects = []
        obstacles = self.obstacle_manager.get_all_outer_circles()
        if obstacles:
            objects.extend(obstacles)
        if self.vehicle_manager.get_vehicle_count() > 0:
            vehicles = self.vehicle_manager.get_all_vehicles()
            body = []
            for v in vehicles:
                if v.id != vehicle_id:
                    body.extend(v.get_outer_circles_world())
            if vehicle._check_collision(body):
                raise ValueError("위치가 다른 차량과 충돌합니다")
            objects.extend(body)

        return self.road_manager.connect(vehicle.get_position(), (x, y, yaw), objects)

    def _get_obs(self):
        """
        모든 차량의 관측값 반환

        Returns:
            관측값 (numpy 배열), (num_vehicles, obs_dim)
        """
        observations = np.array([self._get_vehicle_observation(vehicle) for vehicle in self.vehicle_manager.get_all_vehicles()])
        return observations

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
        scale_vel_long, scale_acc_long = state.scale_long(state.vel_long, state.acc_long, self.max_vel_long, self.min_vel_long, self.max_acc_long, self.min_acc_long)
        scale_vel_lat, scale_acc_lat = state.scale_lat(state.vel_lat, state.acc_lat, self.max_vel_lat, self.max_acc_lat)
        goal_distance = state.scale_distance(state.distance_to_target)
        cos_goal_yaw_diff, sin_goal_yaw_diff = state.encoding_angle(state.yaw_diff_to_target)
        frenet_d = state.scale_frenet_d(state.frenet_d)

        # 기본 차량 상태 (13, )
        obs = np.array([
            state.steer,            # -1 ~ 1
            state.throttle_engine,  # 0 ~ 1
            state.throttle_brake,   # 0 ~ 1
            cos_yaw,                # -1 ~ 1
            sin_yaw,                # -1 ~ 1
            scale_vel_long,         # -1 ~ 1
            scale_acc_long,         # -1 ~ 1
            scale_vel_lat,          # -1 ~ 1
            scale_acc_lat,          # -1 ~ 1
            goal_distance,          # 0 ~ 1
            cos_goal_yaw_diff,      # -1 ~ 1
            sin_goal_yaw_diff,      # -1 ~ 1
            frenet_d,               # -1 ~ 1
        ], dtype=np.float32)

        # (36, ), 0 ~ 1, 정규화된 데이터
        lidar_data = state.get_lidar_data()
        # (6, 7), (norm_distance, cos_diff, sin_diff, scale_vel_long, scale_acc_long, scale_vel_lat, scale_acc_lat)
        trajectory_data = state.get_trajectory_data(self.max_vel_long, self.min_vel_long, self.max_acc_long, self.min_acc_long, self.max_vel_lat, self.max_acc_lat)

        obs = np.concatenate((obs, lidar_data, trajectory_data.flatten())) # (91, )
        return obs

    def _calculate_rewards(self, collisions, outside_roads, reached_targets):
        """
        모든 차량의 보상 계산

        Args:
            collisions: 차량별 충돌 여부 {vehicle_id: bool}
            outside_roads: 차량별 도로 이탈 여부 {vehicle_id: bool}
            reached_targets: 차량별 목표 도달 여부 {vehicle_id: bool}

        Returns:
            보상 배열 (numpy 배열), (num_vehicles, )
        """
        reward_array = np.array([
            self._calculate_vehicle_reward(
                vehicle,
                collisions.get(vehicle.id, False),
                outside_roads.get(vehicle.id, False),
                reached_targets.get(vehicle.id, False)
            )
            for vehicle in self.vehicle_manager.get_all_vehicles()
        ])
        return reward_array

    def _calculate_vehicle_reward(self, vehicle, collision, outside_road, reached_target):
        """
        단일 차량 보상 계산

        Args:
            vehicle: 차량 객체
            collision: 충돌 여부
            outside_road: 도로 이탈 여부
            reached_target: 목표 도달 여부

        Returns:
            계산된 보상값
        """
        state = vehicle.get_state()
        rewards = self.config['simulation']['rewards']

        reward = 0

        # 속도 보상 (최대 속도에 가까울수록 높은 보상)
        speed_reward = (state.vel_long / self.config['vehicle']['max_speed']) * rewards['speed_reward_factor']
        reward += speed_reward

        # 목표 거리에 따른 보상 (가까울수록 높은 보상)
        if state.distance_to_target > 0:
            proximity_reward = 1.0 / (1.0 + state.distance_to_target) * rewards['distance_reward_factor']
            reward += proximity_reward

        # 방향 일치 보상 (차량이 목표를 바라볼수록 높은 보상)
        direction_reward = (1.0 - abs(state.yaw_diff_to_target) / np.pi) * rewards['orientation_reward_factor']
        reward += direction_reward

        # 충돌 패널티
        if collision:
            reward -= rewards['collision_penalty']

        # 도로 이탈 패널티
        if outside_road:
            reward -= rewards['outside_road_penalty']

        # 목표 도달 보상
        if reached_target:
            reward += rewards['goal_reached_reward']

        return reward
