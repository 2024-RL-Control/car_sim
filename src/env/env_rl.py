# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import pygame
import random
from math import pi
import numpy as np
import gymnasium as gym
from datetime import datetime
from collections import deque, defaultdict
from typing import Tuple
import re
from src.env.env import CarSimulatorEnv
from ..model.sb3 import SACVehicleAlgorithm, PPOVehicleAlgorithm, CustomFeatureExtractor, CustomFeatureExtractor2
from ..model.sb3 import create_optimized_callbacks, get_callback_summary
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

class ActionController:
    """
    강화학습 환경에서 행동 선택 빈도를 제어하는 클래스
    시뮬레이션 시간 기준으로 지정된 Hz에 맞춰 새로운 행동을 선택하고,
    그 사이에는 이전 행동을 유지합니다.
    """

    def __init__(self, action_hz: float, num_vehicles: int, action_dim: int, debug: bool = False):
        """
        ActionController 초기화

        Args:
            action_hz: 새로운 행동 선택 주파수 [Hz] (시뮬레이션 시간 기준)
            num_vehicles: 차량 수
            action_dim: 각 차량의 행동 차원
            debug: 디버그 출력 여부
        """
        self.action_hz = action_hz
        self.action_interval = 1.0 / action_hz if action_hz > 0 else 0.0
        self.num_vehicles = num_vehicles
        self.action_dim = action_dim
        self.debug = debug

        # 이전 행동 저장
        self.last_actions = np.zeros((num_vehicles, action_dim), dtype=np.float64)

        # 타이밍 관리
        self.last_action_selection_time = 0.0
        self.action_selection_count = 0
        self.step_count = 0

        if self.debug:
            print(f"ActionController 초기화: {action_hz}Hz, 간격: {self.action_interval:.4f}초")

    def should_select_new_action(self, current_simulation_time: float) -> bool:
        """
        새로운 행동을 선택해야 하는지 판단

        Args:
            current_simulation_time: 현재 시뮬레이션 시간 [초]

        Returns:
            새로운 행동 선택이 필요한지 여부
        """
        if self.action_interval <= 0:
            return True  # Hz가 0이면 매번 새로운 행동 선택

        time_since_last_selection = current_simulation_time - self.last_action_selection_time

        # 첫 번째 호출이거나 충분한 시간이 경과했으면 새로운 행동 선택
        should_select = (self.action_selection_count == 0) or (time_since_last_selection >= self.action_interval)

        if self.debug and should_select:
            actual_hz = 1.0 / time_since_last_selection if time_since_last_selection > 0 else float('inf')
            print(f"    디버깅: 새로운 행동 선택: 시간 {current_simulation_time:.4f}초, "
                  f"간격 {time_since_last_selection:.4f}초, 실제 Hz: {actual_hz:.2f}")

        return should_select

    def update_action(self, new_actions: np.ndarray, current_simulation_time: float):
        """
        새로운 행동으로 업데이트

        Args:
            new_actions: 새로운 행동 배열 (num_vehicles, action_dim)
            current_simulation_time: 현재 시뮬레이션 시간 [초]
        """
        self.last_actions = new_actions.copy()
        self.last_action_selection_time = current_simulation_time
        self.action_selection_count += 1

        if self.debug:
            print(f"    디버깅: 행동 업데이트: {self.action_selection_count}번째, "
                  f"행동값: {new_actions[0] if len(new_actions) > 0 else 'None'}")

    def get_current_action(self, current_simulation_time: float, new_actions: np.ndarray = None) -> np.ndarray:
        """
        현재 사용할 행동 반환 (새로운 행동 또는 이전 행동 유지)

        Args:
            current_simulation_time: 현재 시뮬레이션 시간 [초]
            new_actions: 새로운 행동 후보 (필요시에만 사용)

        Returns:
            사용할 행동 배열 (num_vehicles, action_dim)
        """
        self.step_count += 1

        if self.should_select_new_action(current_simulation_time):
            if new_actions is not None:
                self.update_action(new_actions, current_simulation_time)
                return self.last_actions.copy()
            else:
                # new_actions가 제공되지 않았지만 새로운 행동이 필요한 경우
                # 이전 행동을 그대로 사용
                if self.debug:
                    print("    디버깅: 새로운 행동이 필요하지만 제공되지 않음. 이전 행동 유지.")
                return self.last_actions.copy()
        else:
            # 이전 행동 유지
            if self.debug and self.step_count % 100 == 0:  # 너무 자주 출력하지 않도록
                print(f"    디버깅: 이전 행동 유지 (스텝 {self.step_count})")
            return self.last_actions.copy()

    def reset(self):
        """ActionController 상태 초기화"""
        self.last_actions.fill(0.0)
        self.last_action_selection_time = 0.0
        self.action_selection_count = 0
        self.step_count = 0

        if self.debug:
            print("    초기화: ActionController 리셋")

    def get_statistics(self) -> dict:
        """통계 정보 반환"""
        return {
            'action_hz': self.action_hz,
            'action_interval': self.action_interval,
            'action_selection_count': self.action_selection_count,
            'step_count': self.step_count,
            'actual_selection_rate': self.action_selection_count / max(self.step_count, 1)
        }

class DummyEnv(gym.Env):
    def __init__(self, rl_env):
        self.rl_env = rl_env
        # single_stacked_shape = (self.rl_env.observation_space.shape[1],)
        single_stacked_shape = self.rl_env.stacked_obs_dim # (frame_stack_size, single_obs_dim)
        self.observation_space = gym.spaces.Box(
            # low=np.tile(self.rl_env.env.observation_space.low, self.rl_env.obs_stack_size),
            # high=np.tile(self.rl_env.env.observation_space.high, self.rl_env.obs_stack_size),
            low=-1.0,  # 정규화된 최소값
            high=1.0,  # 정규화된 최대값
            shape=single_stacked_shape,
            dtype=np.float64
        )
        self.action_space = self.rl_env.env.action_space

    def reset(self, *, seed=None, options=None):
        observations, info = self.rl_env.reset(seed=seed, options=options)
        return observations[0], info

    def step(self, actions):
        multi_actions = np.tile(actions, (self.rl_env.num_vehicles, 1))
        observations, reward, done, truncated, info = self.rl_env.step(multi_actions)
        return observations[0], reward, done, truncated, info

class BasicRLDrivingEnv(gym.Env):
    """
    강화학습 기반 기초 자율주행 에이전트를 위한 환경 클래스
    목표:
    1. 목적지 도착
    2. 장애물 회피
    3. 차선 유지
    """
    def __init__(self, config_path=None, config: dict = None, verbose=1):
        """
        RL 환경 초기화

        Args:
            config_path: 설정 파일 경로 (기본값: None)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.verbose = verbose
        if self.verbose > 0:
            print(f"    정보: 사용 중인 디바이스: {self.device}")

        # 기본 CarSimulator 환경 초기화
        self.env = CarSimulatorEnv(config_path, config, setup=False)
        self.vehicle_manager = self.env.get_vehicle_manager()

        self.num_vehicles = self.env.num_vehicles
        self.active_agents = [True] * self.num_vehicles

        self.rl_config = self.env.config['simulation']['rl']
        self.rl_callback_config = self.rl_config['callbacks']

        self.base_seed = self.rl_config['seed']
        self.test_base_seed = None
        self.test_seed_offset = self.rl_config['eval']['test_seed_offset']
        if self.verbose > 0:
            print(f"    설정: 환경 시드 생성기 초기화 (Base Seed: {self.base_seed})")
        self.env_seed_rng = np.random.RandomState(self.base_seed)

        # 관측 설정
        self.obs_stack_size = self.rl_config['frame_stack_size']
        if self.verbose > 0:
            print(f"    설정: Frame Stacking 활성화: {self.obs_stack_size} 프레임")
        self.observation_buffers = [
            deque(maxlen=self.obs_stack_size) for _ in range(self.num_vehicles)
        ]
        self.single_obs_dim = self.env.observation_space.shape[0]
        # self.stacked_obs_dim = self.single_obs_dim * self.obs_stack_size
        self.stacked_obs_dim = (self.obs_stack_size, self.single_obs_dim)
        self.observation_space = gym.spaces.Box(
            low=-1.0,  # 정규화된 최소값
            high=1.0,  # 정규화된 최대값
            shape=(self.num_vehicles, *self.stacked_obs_dim), # (num_vehicles, frame_stack_size, obs_dim)
            dtype=np.float64
        )

        # 행동 설정 및 ActionController 초기화
        action_hold_behavior = self.rl_config.get('action_hold_behavior', True)
        if action_hold_behavior:
            action_hz = self.rl_config.get('action_selection_hz', 10)
            self.action_controller = ActionController(
                action_hz=action_hz,
                num_vehicles=self.num_vehicles,
                action_dim=self.env.action_space.shape[0],
                debug=self.rl_config.get('debug_action_timing', False)
            )
            if self.verbose > 0:
                print(f"    설정: ActionController 활성화: {action_hz}Hz로 행동 선택")
        else:
            self.action_controller = None
            if self.verbose > 0:
                print("    설정: ActionController 비활성화: 매 스텝마다 새로운 행동 선택")
        self.action_space = gym.spaces.Box(
            low=np.tile(self.env.action_space.low, (self.num_vehicles, 1)),
            high=np.tile(self.env.action_space.high, (self.num_vehicles, 1)),
            shape=(self.num_vehicles, self.env.action_space.shape[0]),
            dtype=np.float64
        )

        # 장애물 및 환경 설정
        self.num_static_obstacles = self.env.config['simulation']['obstacle']['num_static_obstacles']
        self.num_dynamic_obstacles = self.env.config['simulation']['obstacle']['num_dynamic_obstacles']
        # 환경 바운더리 설정
        self.boundary = {
            'x_min': self.env.config['simulation']['boundary']['x_min'],
            'x_max': self.env.config['simulation']['boundary']['x_max'],
            'y_min': self.env.config['simulation']['boundary']['y_min'],
            'y_max': self.env.config['simulation']['boundary']['y_max']
        }
        # 장애물 배치 가능 공간
        self.obstacle_area = {
            'x_min': self.env.config['simulation']['obstacle']['boundary']['x_min'],
            'x_max': self.env.config['simulation']['obstacle']['boundary']['x_max'],
            'y_min': self.env.config['simulation']['obstacle']['boundary']['y_min'],
            'y_max': self.env.config['simulation']['obstacle']['boundary']['y_max']
        }
        # 버퍼 거리 (차량과 장애물 공간 사이 여유)
        self.buffer_distance = self.env.config['simulation']['obstacle']['buffer_distance']
        # 차량 배치 가능 공간 계산
        self.vehicle_area = self._calculate_vehicle_areas()

        # 초기화 시 장애물 및 목적지 설정
        self.setup_environment()

        # 스텝 카운터
        self.steps = 0
        self.max_step = self.rl_config['train_max_steps']
        self.max_episode_steps = self.rl_config['train_max_episode_steps']

        # 조기 종료 설정
        self.termination_check_step = self.rl_config['termination_check_step']
        self.progress_change_threshold = self.rl_config['progress_change_threshold']
        self.early_termination_penalty = self.rl_config['rewards']['termination']
        self.accumulated_delta_progress = defaultdict(float)
        self.next_progress_check_step = self.termination_check_step

        # 에피소드 정보 저장용
        self.episode_count = -1

    def _calculate_vehicle_areas(self):
        """차량 배치 가능 영역 계산"""
        areas = {}

        # 좌측 영역
        areas['left'] = {
            'x_min': self.boundary['x_min'],
            'x_max': self.obstacle_area['x_min'] - self.buffer_distance,
            'y_min': self.boundary['y_min'],
            'y_max': self.boundary['y_max']
        }

        # 우측 영역
        areas['right'] = {
            'x_min': self.obstacle_area['x_max'] + self.buffer_distance,
            'x_max': self.boundary['x_max'],
            'y_min': self.boundary['y_min'],
            'y_max': self.boundary['y_max']
        }

        # 상단 영역
        areas['top'] = {
            'x_min': self.boundary['x_min'],
            'x_max': self.boundary['x_max'],
            'y_min': self.obstacle_area['y_max'] + self.buffer_distance,
            'y_max': self.boundary['y_max']
        }

        # 하단 영역
        areas['bottom'] = {
            'x_min': self.boundary['x_min'],
            'x_max': self.boundary['x_max'],
            'y_min': self.boundary['y_min'],
            'y_max': self.obstacle_area['y_min'] - self.buffer_distance
        }

        return areas

    def setup_environment(self):
        """
        환경 구성: 장애물 배치, 차량 배치, 목적지 설정
        """
        success = False

        while not success:
            try:
                # 장애물 배치
                self._setup_obstacles()

                # 차량 배치
                self.vehicle_start_position = []
                self._setup_vehicle()

                # 목적지 설정
                self._setup_goal()

                success = True
            except Exception as e:
                if self.verbose > 0:
                    print(f"Setup failed: {e}")
                continue

    def _setup_obstacles(self):
        """
        장애물 배치: 정적 및 동적 장애물
        """
        obstacle_manager = self.env.get_obstacle_manager()
        obstacle_manager.clear_obstacles()

        # 정적 장애물 배치
        n_static = self.num_static_obstacles
        xs = np.random.uniform(self.obstacle_area['x_min'], self.obstacle_area['x_max'], size=n_static)
        ys = np.random.uniform(self.obstacle_area['y_min'], self.obstacle_area['y_max'], size=n_static)
        yaws = np.random.uniform(-pi/2, pi/2, size=n_static)
        sizes = np.random.uniform(1.0, 5.0, size=n_static)
        types = np.random.choice(['circle', 'square', 'rectangle'], size=n_static)
        static_color = (100, 100, 200)  # 정적 장애물 색상

        for x, y, yaw, sz, t in zip(xs, ys, yaws, sizes, types):
            if t == 'circle':
                obstacle_manager.add_circle_obstacle(None, x, y, 0, 0, 0, sz, static_color)
            elif t == 'square':
                obstacle_manager.add_square_obstacle(None, x, y, yaw, 0, 0, sz, static_color)
            else:  # rectangle
                width = random.uniform(1.5, 5.0)
                height = random.uniform(1.0, 3.5)
                obstacle_manager.add_rectangle_obstacle(None, x, y, yaw, 0, 0, width, height, static_color)

        # 동적 장애물 배치
        n_dynamic = self.num_dynamic_obstacles
        xs = np.random.uniform(self.obstacle_area['x_min'], self.obstacle_area['x_max'], size=n_dynamic)
        ys = np.random.uniform(self.obstacle_area['y_min'], self.obstacle_area['y_max'], size=n_dynamic)
        yaws = np.random.uniform(-pi/2, pi/2, size=n_dynamic)
        yaw_rates = np.random.uniform(-0.2, 0.2, size=n_dynamic)  # 회전 속도
        speeds = np.random.uniform(1.0, 3.0, size=n_dynamic)
        sizes = np.random.uniform(1.0, 3.0, size=n_dynamic)
        types = np.random.choice(['circle', 'square'], size=n_dynamic)
        dynamic_color = (200, 100, 100)  # 동적 장애물 색상

        for x, y, yaw, yaw_rate, speed, sz, t in zip(xs, ys, yaws, yaw_rates, speeds, sizes, types):
            if t == 'circle':
                obstacle_manager.add_circle_obstacle(None, x, y, yaw, yaw_rate, speed, sz, dynamic_color)
            else:  # square
                obstacle_manager.add_square_obstacle(None, x, y, yaw, yaw_rate, speed, sz, dynamic_color)

    def _setup_vehicle(self):
        """
        차량 배치: 외곽 영역에 차량 배치
        """
        # 기존 차량 모두 제거
        self.vehicle_manager.clear_all_vehicles()

        # 차량 배치 가능 공간은 외곽 영역
        # 상하좌우 네 영역 중 랜덤 선택
        for i in range(self.num_vehicles):
            placement_area = random.choice(['top', 'bottom', 'left', 'right'])
            # placement_area = 'left'
            boundary = self.vehicle_area[placement_area]
            x = random.uniform(boundary['x_min'], boundary['x_max'])
            y = random.uniform(boundary['y_min'], boundary['y_max'])
            yaw_volatility = random.uniform(-pi/7, pi/7)

            if placement_area == 'top':
                yaw = -pi/2 + yaw_volatility  # 아래쪽 방향
            elif placement_area == 'bottom':
                yaw = pi/2 + yaw_volatility  # 위쪽 방향
            elif placement_area == 'left':
                yaw = 0 + yaw_volatility  # 오른쪽 방향
            else:  # right
                yaw = pi + yaw_volatility  # 왼쪽 방향

            # 새 차량 생성
            self.vehicle_manager.create_vehicle(x=x, y=y, yaw=yaw, vehicle_id=i)

            # 차량 시작 위치 저장 (목적지 설정에 사용)
            self.vehicle_start_position.append({
                'x': x,
                'y': y,
                'placement': placement_area
            })

    def _setup_goal(self):
        """
        목적지 설정: 차량 시작 위치의 반대편에 목적지 설정
        """
        # 차량 시작 위치의 반대편 결정
        for i in range(self.num_vehicles):
            vehicle_placement = self.vehicle_start_position[i]['placement']
            yaw_volatility = random.uniform(-pi/7, pi/7)
            if vehicle_placement == 'top':
                boundary = self.vehicle_area['bottom']
                yaw = -pi/2 + yaw_volatility
            elif vehicle_placement == 'bottom':
                boundary = self.vehicle_area['top']
                yaw = pi/2 + yaw_volatility
            elif vehicle_placement == 'left':
                boundary = self.vehicle_area['right']
                yaw = 0 + yaw_volatility
            else:  # right
                boundary = self.vehicle_area['left']
                yaw = pi + yaw_volatility

            x = random.uniform(boundary['x_min'], boundary['x_max'])
            y = random.uniform(boundary['y_min'], boundary['y_max'])

            # 차량에 목적지 추가
            self.env.add_goal_for_vehicle(i, x, y, yaw, radius=4.0)

    def _stack_observation(self, new_observations, pad_with_current=False):
        """
        새로운 관측값을 버퍼에 추가하고 스택된 관측값을 반환합니다.

        Args:
            new_observations: (num_vehicles, single_obs_dim)
            pad_with_current: True이면 reset 시 현재 obs로 버퍼를 채웁니다.

        Returns:
            stacked_observations: (num_vehicles, stacked_obs_dim)
        """
        stacked_obs_list = []

        for i in range(self.num_vehicles):
            buffer = self.observation_buffers[i]
            current_obs = new_observations[i]

            if pad_with_current:
                # reset() 호출 시: 현재 관측값으로 버퍼를 채움
                buffer.clear()
                for _ in range(self.obs_stack_size):
                    buffer.append(current_obs.copy())
            else:
                # step() 호출 시: 새 관측값 추가
                buffer.append(current_obs.copy())

            # # 버퍼의 모든 프레임을 하나의 벡터로 결합 (axis=None으로 1D로 flatten)
            # stacked_obs_list.append(np.concatenate(list(buffer), axis=None))
            stacked_obs_list.append(np.array(list(buffer), dtype=np.float64)) # (frame_stack_size, obs_dim)

        return np.array(stacked_obs_list, dtype=np.float64)

    def deactivate_action_controller(self):
        """ActionController 비활성화"""
        self.action_controller = None

    def reset(self, *, seed=None, options=None):
        """
        환경 초기화 및 초기 관측값 반환

        Args:
            seed: 환경의 난수 생성기 시드
            options: 추가 옵션 딕셔너리

        Returns:
            observations: 차량별 초기 관측 값 (num_vehicles, obs_dim)
        """
        env_seed = None
        if seed is not None:
            # 1. 명시적 시드 사용
            env_seed = seed
        else:
            # 2. 내부 시퀀셜 RNG 사용
            env_seed = self.env_seed_rng.randint(0, 2**31 - 1)

        # 환경 초기화
        _ = self.env.reset(seed=env_seed, options=options)

        # 스텝 카운터 초기화
        self.steps = 0

        # 에피소드 카운터 증가
        self.episode_count += 1

        # 에이전트 활성화 상태 초기화
        self.active_agents = [True] * self.num_vehicles

        # ActionController 리셋
        if self.action_controller is not None:
            self.action_controller.reset()

        # 조기 종료 관련 변수 초기화
        self.accumulated_delta_progress = defaultdict(float)
        self.next_progress_check_step = self.termination_check_step

        # 장애물, 차량, 목적지 다시 설정
        self.setup_environment()

        # 초기 스텝으로 환경 안정화
        dummy_actions = np.zeros((self.num_vehicles, self.env.action_space.shape[-1]), dtype=np.float64)
        self.env.step(dummy_actions)

        # 초기 관측값 반환
        single_observations = self.env._get_obs()

        # Frame Stacking: 버퍼를 초기 관측값으로 채우고 스택된 obs 반환
        stacked_observations = self._stack_observation(single_observations, pad_with_current=True)
        return stacked_observations, self.active_agents

    def step(self, actions):
        """
        환경에서 한 스텝 진행

        Args:
            actions: 모든 차량의 행동 (num_vehicles, action_dim)

        Returns:
            observations: 모든 차량의 관측값 (num_vehicles, obs_dim)
            reward: 모든 차량의 평균 보상
            done: 종료 여부 bool
            info: 추가 정보
        """
        # ActionController를 사용하여 행동 제어
        if self.action_controller is not None:
            # 현재 시뮬레이션 시간 가져오기
            current_simulation_time = self.env._time_elapsed

            # ActionController를 통해 실제 사용할 행동 결정
            actual_actions = self.action_controller.get_current_action(
                current_simulation_time, actions
            )
        else:
            # ActionController가 비활성화된 경우 원래 행동 그대로 사용
            actual_actions = actions

        # 환경에서 스텝 진행 (실제 사용할 행동으로)
        single_observations, rewards, done, _, info = self.env.step(actual_actions)

        # 스텝 카운터 증가
        self.steps += 1

        # 차량 비활성화
        for vehicle_id in range(self.num_vehicles):
            truncated = info['truncated'].get(vehicle_id, False)
            terminated = info['terminated'].get(vehicle_id, False)
            vehicle_done = truncated or terminated

            if vehicle_done and self.active_agents[vehicle_id]:
                self.active_agents[vehicle_id] = False

            if self.active_agents[vehicle_id]:
                vehicle = self.vehicle_manager.get_vehicle_by_id(vehicle_id)
                if vehicle:
                    delta = vehicle.state.get_delta_progress()
                    self.accumulated_delta_progress[vehicle_id] -= delta

        if self.steps >= self.max_episode_steps:
            done = True

        # 조기 종료 체크
        is_early_termination = False
        if not done and self.steps == self.next_progress_check_step:
            active_accumulators = [self.accumulated_delta_progress[vehicle_id] for vehicle_id, is_active in enumerate(self.active_agents) if is_active]
            if active_accumulators:
                mean_progress = np.mean(active_accumulators)
                if mean_progress < self.progress_change_threshold:
                    is_early_termination = True
                    for vehicle_id, is_active in enumerate(self.active_agents):
                        if is_active:
                            rewards[vehicle_id] += self.early_termination_penalty

            self.next_progress_check_step += self.termination_check_step
            self.accumulated_delta_progress.clear()

        # 최종 done 플래그 및 info 설정
        done = done or is_early_termination
        info['early_termination'] = is_early_termination
        info['active_agents'] = self.active_agents.copy()
        average_reward = np.mean(rewards)

        # ActionController 통계 정보 추가
        if self.action_controller is not None:
            action_stats = self.action_controller.get_statistics()
            info.update({
                'episode_count': self.episode_count,
                'rewards': rewards,  # 개별 차량별 보상
                'episode_length': self.steps,  # 현재 에피소드 길이
                'elapsed_time': 0,  # 필요시 실제 시간 계산 가능
                'action_controller_stats': action_stats,  # ActionController 통계
                'action_selection_rate': action_stats['actual_selection_rate']  # 실제 행동 선택 비율
            })
        else:
            info.update({
                'episode_count': self.episode_count,
                'rewards': rewards,  # 개별 차량별 보상
                'episode_length': self.steps,  # 현재 에피소드 길이
                'elapsed_time': 0,  # 필요시 실제 시간 계산 가능
            })

        # Frame Stacking: 새 관측값을 버퍼에 추가하고 스택된 obs 반환
        stacked_observations = self._stack_observation(single_observations, pad_with_current=False)
        return stacked_observations, average_reward, done, False, info

    def _draw_boundary(self):
        # 경계(boundary)와 장애물 영역(obstacle_area) 시각화
        if self.env.config['visualization']['debug_mode']:
            world_to_screen = self.env.camera.world_to_screen

            # 경계(Boundary) 시각화 - 노란색 선
            boundary_color = (255, 255, 0)  # RGB: 노란색
            pygame.draw.line(self.env.renderer.screen, boundary_color,
                             world_to_screen((self.boundary['x_min'], self.boundary['y_min'])),
                             world_to_screen((self.boundary['x_max'], self.boundary['y_min'])), 2)
            pygame.draw.line(self.env.renderer.screen, boundary_color,
                             world_to_screen((self.boundary['x_max'], self.boundary['y_min'])),
                             world_to_screen((self.boundary['x_max'], self.boundary['y_max'])), 2)
            pygame.draw.line(self.env.renderer.screen, boundary_color,
                             world_to_screen((self.boundary['x_max'], self.boundary['y_max'])),
                             world_to_screen((self.boundary['x_min'], self.boundary['y_max'])), 2)
            pygame.draw.line(self.env.renderer.screen, boundary_color,
                             world_to_screen((self.boundary['x_min'], self.boundary['y_max'])),
                             world_to_screen((self.boundary['x_min'], self.boundary['y_min'])), 2)

            # 장애물 영역(Obstacle Area) 시각화 - 빨간색 선
            obstacle_area_color = (255, 0, 0)  # RGB: 빨간색
            pygame.draw.line(self.env.renderer.screen, obstacle_area_color,
                             world_to_screen((self.obstacle_area['x_min'], self.obstacle_area['y_min'])),
                             world_to_screen((self.obstacle_area['x_max'], self.obstacle_area['y_min'])), 2)
            pygame.draw.line(self.env.renderer.screen, obstacle_area_color,
                             world_to_screen((self.obstacle_area['x_max'], self.obstacle_area['y_min'])),
                             world_to_screen((self.obstacle_area['x_max'], self.obstacle_area['y_max'])), 2)
            pygame.draw.line(self.env.renderer.screen, obstacle_area_color,
                             world_to_screen((self.obstacle_area['x_max'], self.obstacle_area['y_max'])),
                             world_to_screen((self.obstacle_area['x_min'], self.obstacle_area['y_max'])), 2)
            pygame.draw.line(self.env.renderer.screen, obstacle_area_color,
                             world_to_screen((self.obstacle_area['x_min'], self.obstacle_area['y_max'])),
                             world_to_screen((self.obstacle_area['x_min'], self.obstacle_area['y_min'])), 2)

            # 화면 업데이트
            pygame.display.flip()

    def render(self, mode='human'):
        """
        환경 렌더링
        """
        # 기본 환경 렌더링 수행
        self.env.render(mode)
        # 영역 시각화
        self._draw_boundary()

    def handle_keyboard_input(self):
        """
        키보드 입력 처리
        """
        self.env.handle_keyboard_input()

    def close(self):
        """
        환경 종료
        """
        self.env.close()

    def print_basic_controls(self):
        print("\n=== RL Basic Driving Env ===")
        print("  C: Toggle camera follow")
        print("  R: Reset camera view")
        print("  +/-: Zoom in/out")
        print("  I/J/K/L: Pan camera")
        print("  F1: Toggle Training Mode")
        print("  F2: Toggle Visualization")
        print("  F3: Toggle HUD")
        print("  F4: Toggle debug mode")
        print("  F5: Save state")
        print("  F9: Load state")
        print("  Tab: Switch between vehicles")
        print("  ESC: Quit")

    def _find_latest_model(self, models_dir: str, algorithm: str) -> Tuple[str, str]:
        """
        지정된 디렉토리에서 로드할 최신/최고의 모델 반환
        우선순위: final > highest_step > best

        Returns:
            (model_path, replay_buffer_path)
        """
        if not os.path.exists(models_dir):
            return None, None

        files = os.listdir(models_dir)

        # 1. 우선순위 1: final
        final_model = os.path.join(models_dir, f"{algorithm}_final.zip")
        if final_model in [os.path.join(models_dir, f) for f in files]:
            buffer = os.path.join(models_dir, f"{algorithm}_final_replay_buffer.pkl")
            return final_model, buffer if os.path.exists(buffer) else None

        # 2. 우선순위 2: 가장 높은 step
        step_models = []
        for f in files:
            # 정규식: {algorithm이름}_(숫자)_steps.zip
            match = re.match(rf"^{re.escape(algorithm)}_(\d+)_steps\.zip$", f)
            if match:
                steps = int(match.group(1))
                step_models.append((steps, os.path.join(models_dir, f)))

        if step_models:
            # 스텝 수가 가장 많은 모델 (내림차순 정렬 후 첫 번째)
            step_models.sort(key=lambda x: x[0], reverse=True)
            latest_step_model_path = step_models[0][1]
            # 스텝 체크포인트는 .zip 내부에 리플레이 버퍼를 저장하므로
            # 별도의 버퍼 경로는 None을 반환 (SAC.load가 알아서 처리)
            return latest_step_model_path, None

        # 3. 우선순위 3: best
        best_model = os.path.join(models_dir, f"{algorithm}_best.zip")
        if best_model in [os.path.join(models_dir, f) for f in files]:
            buffer = os.path.join(models_dir, f"{algorithm}_best_replay_buffer.pkl")
            return best_model, buffer if os.path.exists(buffer) else None

        # 4. 모델을 찾을 수 없음
        return None, None

    def learn(self, algorithm='sac'):
        """
        자율주행 에이전트 학습 함수 (learn() 메소드 사용)
        """
        # 모델 불러오기 설정
        resume_config = self.rl_config['resume_training']
        resume = resume_config['enabled']
        resume_buffer = resume_config['buffer_load']
        resume_dir = f"./logs/checkpoints/{resume_config['model_path']}"
        if resume:
            model_path, replay_buffer_path = self._find_latest_model(resume_dir, algorithm)
            if model_path is None:
                print(f"    경고: '{resume_dir}'에서 '{algorithm}' 알고리즘 모델을 찾을 수 없습니다. 새 학습을 시작합니다.")
                resume = False

        # 실행 이름 생성 (날짜-시간 포함)
        current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_name = f"{algorithm}_{current_time}"

        # 로그 및 모델 저장 경로 설정
        models_dir = f"./logs/checkpoints/{run_name}"
        log_dir = f"./logs/log/{run_name}"

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Monitor와 DummyVecEnv로 환경 래핑
        dummy_env = DummyEnv(self)
        env = Monitor(dummy_env, log_dir)
        env = DummyVecEnv([lambda: env])

        # 신경망 아키텍처 설정 (강화학습 개선)
        policy_kwargs = {
            # 정책 및 가치 네트워크 아키텍처
            "net_arch": [1024, 1024],
            # 활성화 함수
            "activation_fn": torch.nn.GELU,

            # 커스텀 추출기
            "features_extractor_class": CustomFeatureExtractor2,
            # 추출기 아키텍처
            "features_extractor_kwargs": {
                "net_arch": [128]
            },
            "share_features_extractor": True,
        }

        if algorithm == 'sac':
            # SAC 하이퍼파라미터 설정
            hyperparameters ={
                "buffer_size": self.max_step // 5,
                "learning_rate": 3e-4,
                "batch_size": 512,
                "learning_starts": 5000,
                "n_envs": 1,
                "tau": 0.005,                   # 타겟 네트워크 업데이트 속도
                "gamma": 0.995,                 # 할인 인수
                "ent_coef": "auto",             # 엔트로피 계수
                "train_freq": 8,                # 훈련 빈도
                "gradient_steps": 1,            # 그라디언트 스텝
                "target_update_interval": 1     # 타겟 업데이트 간격
            }

            if resume:
                model = SACVehicleAlgorithm.load(
                    model_path,
                    env=env,
                    device=self.device
                )
                print(f"SAC 모델 로드 완료: {os.path.basename(model_path)}")

                # 리플레이 버퍼 로드
                if resume_buffer:
                    if replay_buffer_path and os.path.exists(replay_buffer_path):
                        # 별도 저장된 버퍼 로드 (final, best)
                        model.load_replay_buffer(replay_buffer_path)
                        print(f"    설정: 별도 리플레이 버퍼 로드 완료: {os.path.basename(replay_buffer_path)}")
                    elif model.replay_buffer is not None:
                        # .zip 파일 내부에 저장된 버퍼 사용 (step 체크포인트)
                        print("    설정: 모델(.zip)에 포함된 리플레이 버퍼를 사용합니다.")
                    else:
                        print("    경고: 리플레이 버퍼를 찾을 수 없습니다. 새 버퍼를 생성합니다.")
                        shared_buffer = ReplayBuffer(
                            buffer_size=hyperparameters['buffer_size'],
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                            device=self.device,
                            n_envs=hyperparameters['n_envs'],
                            handle_timeout_termination=False
                        )
                        model.replay_buffer = shared_buffer
                else:
                    print("    설정: 새 버퍼를 생성합니다.")
                    shared_buffer = ReplayBuffer(
                        buffer_size=hyperparameters['buffer_size'],
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        device=self.device,
                        n_envs=hyperparameters['n_envs'],
                        handle_timeout_termination=False
                    )
                    model.replay_buffer = shared_buffer
            else:
                # 공유 리플레이 버퍼 생성
                shared_buffer = ReplayBuffer(
                    buffer_size=hyperparameters['buffer_size'],
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    device=self.device,
                    n_envs=hyperparameters['n_envs'],
                    handle_timeout_termination=False
                )

                # 커스텀 SAC 모델 생성
                model = SACVehicleAlgorithm(
                    policy="MlpPolicy",
                    env=env,
                    seed=self.base_seed,
                    logging_freq=self.rl_callback_config['logging_freq'],
                    learning_rate=hyperparameters['learning_rate'],
                    policy_kwargs=policy_kwargs,
                    buffer_size=0,
                    learning_starts=hyperparameters['learning_starts'],
                    batch_size=hyperparameters['batch_size'],
                    tau=hyperparameters['tau'],
                    gamma=hyperparameters['gamma'],
                    ent_coef=hyperparameters['ent_coef'],
                    train_freq=hyperparameters['train_freq'],
                    gradient_steps=hyperparameters['gradient_steps'],
                    target_update_interval=hyperparameters['target_update_interval'],
                    verbose=1,
                    tensorboard_log=log_dir,
                    device=self.device
                )

                # 커스텀 리플레이 버퍼를 사용하도록 모델 설정
                model.replay_buffer = shared_buffer
        elif algorithm == 'ppo':
            # PPO 하이퍼파라미터 설정
            hyperparameters = {
                "learning_rate": 3e-4,
                "n_steps": 1024,
                "batch_size": 256,
                "n_epochs": 10,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,      # 클리핑 범위
                "ent_coef": 0.01,       # 엔트로피 보너스, 정책 탐험 장려
                "vf_coef": 0.5,         # 가치 함수 손실 계수
                "max_grad_norm": 0.5    # 그라디언트 클리핑
            }

            if resume:
                # 학습 이어하기: 모델 로드 (PPO는 리플레이 버퍼 없음)
                model = PPOVehicleAlgorithm.load(
                    model_path,
                    env=env,
                    device=self.device
                )
                print(f"    설정: PPO 모델 로드 완료: {os.path.basename(model_path)}")
            else:
                # 커스텀 PPO 모델 생성
                model = PPOVehicleAlgorithm(
                    policy="MlpPolicy",
                    env=env,
                    seed=self.base_seed,
                    logging_freq=self.rl_callback_config['logging_freq'],
                    learning_rate=hyperparameters['learning_rate'],
                    n_steps=hyperparameters['n_steps'],
                    batch_size=hyperparameters['batch_size'],
                    n_epochs=hyperparameters['n_epochs'],
                    gamma=hyperparameters['gamma'],
                    gae_lambda=hyperparameters['gae_lambda'],
                    clip_range=hyperparameters['clip_range'],
                    ent_coef=hyperparameters['ent_coef'],
                    vf_coef=hyperparameters['vf_coef'],
                    max_grad_norm=hyperparameters['max_grad_norm'],
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=log_dir,
                    device=self.device,
                )

        # 로거 설정
        model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))

        # 모델 출력
        print(model.policy)

        # 최적화된 콜백 시스템 설정
        param_config = {
            "algorithm": model.__class__.__name__,
            "dir": run_name,
            "seed": self.base_seed,
            "net_arch": str(policy_kwargs['net_arch']),
            "activation_fn": policy_kwargs['activation_fn'].__name__,
            "action_space": str(self.action_space),
            "observation_space": str(self.observation_space),
            "observation_detail": str(self.rl_config['observation']),
            "features_extractor_class": policy_kwargs['features_extractor_class'].__name__,
            "features_extractor_net_arch": str(policy_kwargs['features_extractor_kwargs']['net_arch']),
            "share_features_extractor": policy_kwargs['share_features_extractor'],
            "frame_stack_size": self.obs_stack_size,
            "resume_training": resume,
            "resume_model_path": resume_config['model_path'] if resume else "N/A",
            "resume_buffer_load": resume_buffer if resume else "N/A"
        }
        param_config.update(hyperparameters)

        callback_config = {
            'algorithm': algorithm,
            'hyperparameters': param_config,
            'num_vehicles': self.num_vehicles,
            'num_static_obstacles': self.num_static_obstacles,
            'num_dynamic_obstacles': self.num_dynamic_obstacles,
            'max_episode_steps': self.max_episode_steps,
            'max_episodes_history': self.rl_callback_config['max_episodes_history'],
            'termination_check_step': self.termination_check_step,
            'progress_change_threshold': self.progress_change_threshold,
            'reward_factor': self.rl_config['rewards'],
            'max_steps_history': self.rl_callback_config['max_steps_history'],
            'gpu_memory_limit': self.rl_callback_config['gpu_memory_limit'],
            'monitoring_freq': self.rl_callback_config['monitoring_freq'],
            'logging_freq': self.rl_callback_config['logging_freq'],
            'checkpoint_freq': self.rl_callback_config['checkpoint_freq'],
            'max_checkpoints': self.rl_callback_config['max_checkpoints'],
            'save_best_model': self.rl_callback_config['save_best_model'],
            'verbose': self.rl_callback_config['verbose']
        }

        # 최적화된 콜백 시스템 생성
        callbacks, metrics_store = create_optimized_callbacks(callback_config, log_dir, models_dir)
        callback = CallbackList(callbacks)

        # 콜백 요약 정보 출력
        summary = get_callback_summary(callbacks)
        if callback_config['verbose'] >= 1:
            print(f"\n=== Optimized Callback System ===")
            print(f"Total callbacks: {summary['total_callbacks']}")
            print(f"Centralized metrics: {summary['centralized_metrics']}")
            print(f"Types: {', '.join(summary['callback_types'])}")
            print("=================================\n")

        # learn() 메소드로 학습 시작
        try:
            # 기본 컨트롤 정보 출력
            self.print_basic_controls()

            # 모델에 환경 정보 및 메트릭 스토어 설정
            model.set_env_info(self)
            model.set_metrics_store(metrics_store)

            model.learn(
                total_timesteps=self.max_step,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=not resume
            )

            # 학습된 모델 저장
            if algorithm == 'sac':
                model.save(os.path.join(models_dir, "sac_final"))
                model.save_replay_buffer(os.path.join(models_dir, "sac_final_replay_buffer"))
            elif algorithm == 'ppo':
                model.save(os.path.join(models_dir, "ppo_final"))

            # 학습 완료 후 추가 로그 저장
            print("\n===== 학습 완료 =====")

            if metrics_store:
                total_time = abs((metrics_store.training_start_time - metrics_store.last_step_time) / 3600)
                print(f"총 학습 시간: {total_time:.2f} 시간")

            if torch.cuda.is_available() and metrics_store:
                if len(metrics_store.gpu_memory_usage) > 0:
                    max_gpu_memory = max(metrics_store.gpu_memory_usage)
                    print(f"최대 GPU 메모리 사용량: {max_gpu_memory:.2f} GB")
                # 메모리 정리
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            error_msg = str(e) if str(e).strip() else repr(e)
            if not error_msg.strip():
                error_msg = f"{exc_type.__name__}: No error message available"

            print(f"\n\n학습 중단됨: {error_msg}")
            print(f'file name: {fname}')
            print(f'error type: {exc_type.__name__}')
            print(f'error msg: {error_msg}')
            print(f'line number: {exc_tb.tb_lineno}')
            print(f'Full traceback:')
            traceback.print_exc()
            print("")

            # 오류 발생시에도 현재까지 학습된 모델 저장 시도
            try:
                if algorithm == 'sac':
                    model.save(os.path.join(models_dir, "sac_interrupted"))
                    model.save_replay_buffer(os.path.join(models_dir, "sac_interrupted_replay_buffer"))
                elif algorithm == 'ppo':
                    model.save(os.path.join(models_dir, "ppo_interrupted"))
                print("중단된 모델 상태가 저장되었습니다.")
            except Exception as save_error:
                print(f"중단된 모델 저장 실패: {save_error}")
        finally:
            # 환경 종료
            self.close()

    def set_test_mode(self, test_seed_offset=None):
        """
        환경 시드 생성기(RNG)를 테스트용으로 재설정합니다.
        훈련 시퀀스와 겹치지 않는 별도의 시드 시퀀스를 생성합니다.
        (offset을 더해 훈련 시드와 완전히 다른 시작점을 보장)
        """
        if test_seed_offset is None:
            test_base_seed = self.base_seed + self.test_seed_offset
        else:
            test_base_seed = self.base_seed + test_seed_offset

        self.test_base_seed = test_base_seed
        self.env_seed_rng = np.random.RandomState(test_base_seed)
        if self.verbose > 0:
            print(f"    설정: 테스트 모드 활성화. 환경 시드 RNG 재설정 (Base: {test_base_seed})")

    def get_test_seed(self):
        """
        현재 테스트 시드를 반환합니다.
        """
        return self.test_base_seed

    def test(self, algorithm='sac'):
        """
        학습된 모델을 테스트하는 함수
        """
        # 모델 경로 설정
        model_path = self.rl_config['eval']['model_path']
        algorithm = model_path.split('_')[0].lower()
        checkpoints_dir = f"./logs/checkpoints/{model_path}"

        if not os.path.exists(checkpoints_dir):
            print(f"    에러: {model_path} 체크포인트가 존재하지 않습니다")
            return

        files = os.listdir(checkpoints_dir)
        final_model = os.path.join(checkpoints_dir, f"{algorithm}_final.zip")
        if final_model in [os.path.join(checkpoints_dir, f) for f in files]:
            print(f"    설정: 최종 모델을 찾았습니다: {final_model}")
            model_path = final_model
        else:
            print(f"    에러: 최종 모델을 찾을 수 없습니다: {final_model}")
            return

        self.set_test_mode()
        max_episode = self.rl_config['eval']['episodes']

        dummy_env = DummyEnv(self)
        env = Monitor(dummy_env)
        env = DummyVecEnv([lambda: env])

        # 모델 로드
        if algorithm == 'sac':
            model = SACVehicleAlgorithm.load(
                model_path,
                env=env,
                device=self.device
            )
        elif algorithm == 'ppo':
            model = PPOVehicleAlgorithm.load(
                model_path,
                env=env,
                device=self.device
            )
        model.policy.eval()

        # 테스트 루프
        success = []
        for ep in range(1, max_episode + 1):
            observations, _ = self.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                # 키보드 입력 수집 (ESC 등)
                self.handle_keyboard_input()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return False

                # 모델 예측
                actions, _ = model.predict(observations, deterministic=True)

                # 한 스텝 진행
                observations, reward, done, truncated, info = self.step(actions)
                total_reward += reward
                steps += 1

                # 화면 갱신
                self.render()

            success.append(info['terminated'].get(0, False))
            # 한 에피소드 결과 출력
            print(f"[Test] Episode {ep}/{max_episode} — Steps: {steps}, Total Reward: {total_reward:.2f}")

        # 시뮬레이터 종료
        print(f"\n[Test] Success Rate: {np.mean(success) * 100:.2f}% ({sum(success)}/{max_episode})")
        self.close()
