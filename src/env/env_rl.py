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
from src.env.env import CarSimulatorEnv
from ..model.sb3 import SACVehicleAlgorithm, PPOVehicleAlgorithm, CustomFeatureExtractor
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
            print(f"새로운 행동 선택: 시간 {current_simulation_time:.4f}초, "
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
            print(f"행동 업데이트: {self.action_selection_count}번째, "
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
                    print("새로운 행동이 필요하지만 제공되지 않음. 이전 행동 유지.")
                return self.last_actions.copy()
        else:
            # 이전 행동 유지
            if self.debug and self.step_count % 100 == 0:  # 너무 자주 출력하지 않도록
                print(f"이전 행동 유지 (스텝 {self.step_count})")
            return self.last_actions.copy()

    def reset(self):
        """ActionController 상태 초기화"""
        self.last_actions.fill(0.0)
        self.last_action_selection_time = 0.0
        self.action_selection_count = 0
        self.step_count = 0

        if self.debug:
            print("ActionController 리셋")

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
        self.observation_space = self.rl_env.env.observation_space
        self.action_space = self.rl_env.env.action_space

    def reset(self, *, seed=None, options=None):
        observations, info = self.rl_env.reset(seed=seed, options=options)
        return observations[0], info

    def step(self, actions):
        multi_actions = np.tile(actions, (self.rl_env.num_vehicles, 1))
        observations, reward, done, truncated, info = self.rl_env.step(multi_actions)

        # Monitor를 위한 info 딕셔너리 수정 - episode 종료 시 올바른 정보 전달
        if done:
            # Monitor가 기대하는 형태로 info 구성
            episode_info = {
                'r': reward,  # 에피소드 총 보상
                'l': info.get('episode_length', 1),  # 에피소드 길이 (최소 1)
                't': info.get('elapsed_time', 0.0)  # 경과 시간
            }
            info['episode'] = episode_info

            # 디버깅용 출력 (verbose 모드에서만)
            if hasattr(self.rl_env, 'episode_count'):
                episode_msg = f"Monitor 기록: Episode {self.rl_env.episode_count}, Reward: {reward:.2f}, Length: {episode_info['l']}"

                # ActionController 통계 추가
                if hasattr(self.rl_env, 'action_controller') and self.rl_env.action_controller is not None:
                    action_stats = info.get('action_controller_stats', {})
                    selection_rate = action_stats.get('actual_selection_rate', 0)
                    selection_count = action_stats.get('action_selection_count', 0)
                    episode_msg += f", 행동선택: {selection_count}회 ({selection_rate:.2%})"

                print(episode_msg)

        return observations[0], reward, done, truncated, info

class BasicRLDrivingEnv(gym.Env):
    """
    강화학습 기반 기초 자율주행 에이전트를 위한 환경 클래스
    목표:
    1. 목적지 도착
    2. 장애물 회피
    3. 차선 유지
    """
    def __init__(self, config_path=None):
        """
        RL 환경 초기화

        Args:
            config_path: 설정 파일 경로 (기본값: None)
        """
        # 기본 CarSimulator 환경 초기화
        self.env = CarSimulatorEnv(config_path, setup=False)

        self.num_vehicles = self.env.num_vehicles
        self.active_agents = [True] * self.num_vehicles

        self.observation_space = gym.spaces.Box(
            low=np.tile(self.env.observation_space.low, (self.num_vehicles, 1)),
            high=np.tile(self.env.observation_space.high, (self.num_vehicles, 1)),
            shape=(self.num_vehicles, self.env.observation_space.shape[0]),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Box(
            low=np.tile(self.env.action_space.low, (self.num_vehicles, 1)),
            high=np.tile(self.env.action_space.high, (self.num_vehicles, 1)),
            shape=(self.num_vehicles, self.env.action_space.shape[0]),
            dtype=np.float64
        )

        self.num_static_obstacles = self.env.config['simulation']['obstacle']['num_static_obstacles']
        self.num_dynamic_obstacles = self.env.config['simulation']['obstacle']['num_dynamic_obstacles']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        print(f"사용 중인 디바이스: {self.device}")

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
        self.max_step = self.env.config['simulation']['train_max_steps']
        self.max_episode_steps = self.env.config['simulation']['train_max_episode_steps']

        # 에피소드 정보 저장용
        self.episode_count = -1

        # ActionController 초기화
        self.rl_config = self.env.config['simulation']['reinforcement_learning']
        self.rl_callback_config = self.rl_config['callbacks']

        action_hold_behavior = self.rl_config.get('action_hold_behavior', True)
        if action_hold_behavior:
            action_hz = self.rl_config.get('action_selection_hz', 10)
            self.action_controller = ActionController(
                action_hz=action_hz,
                num_vehicles=self.num_vehicles,
                action_dim=self.env.action_space.shape[0],
                debug=self.rl_config.get('debug_action_timing', False)
            )
            print(f"ActionController 활성화: {action_hz}Hz로 행동 선택")
        else:
            self.action_controller = None
            print("ActionController 비활성화: 매 스텝마다 새로운 행동 선택")

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
        # 장애물 배치
        self._setup_obstacles()

        # 차량 배치
        self.vehicle_start_position = []
        self._setup_vehicle()

        # 목적지 설정
        self._setup_goal()

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
        self.env.vehicle_manager.clear_all_vehicles()

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
            self.env.vehicle_manager.create_vehicle(x=x, y=y, yaw=yaw, vehicle_id=i)

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

    def reset(self, *, seed=None, options=None):
        """
        환경 초기화 및 초기 관측값 반환

        Args:
            seed: 환경의 난수 생성기 시드
            options: 추가 옵션 딕셔너리

        Returns:
            observations: 차량별 초기 관측 값 (num_vehicles, obs_dim) [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, distance_to_target, yaw_diff_to_target]
        """
        success = False

        while not success:
            try:
                # 환경 초기화
                observations = self.env.reset(seed=seed, options=options)

                # 장애물, 차량, 목적지 다시 설정
                self.setup_environment()

                # 에이전트 활성화 상태 초기화
                self.active_agents = [True] * self.num_vehicles

                success = True
            except Exception as e:
                print(f"Reset failed: {e}")
                continue

        # 스텝 카운터 초기화
        self.steps = 0

        # ActionController 리셋
        if self.action_controller is not None:
            self.action_controller.reset()

        # 에피소드 카운터 증가
        self.episode_count += 1

        # 초기 관측값 반환
        return observations, self.active_agents

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
        observations, rewards, done, _, info = self.env.step(actual_actions)
        average_reward = np.mean(rewards)

        # 스텝 카운터 증가
        self.steps += 1

        # 차량 비활성화
        for vehicle_id in range(self.num_vehicles):
            truncated = info['truncated'].get(vehicle_id, False)
            terminated = info['terminated'].get(vehicle_id, False)
            vehicle_done = truncated or terminated

            if vehicle_done and self.active_agents[vehicle_id]:
                self.active_agents[vehicle_id] = False

        if self.steps >= self.max_episode_steps:
            done = True

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

        return observations, average_reward, done, False, info

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
        print("=== RL Basic Driving Env ===")
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

    def learn(self, algorithm='sac'):
        """
        자율주행 에이전트 학습 함수 (learn() 메소드 사용)
        """
        # 로그 및 모델 저장 경로 설정
        models_dir = "./logs/checkpoints"
        log_dir = "./logs/log"

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Monitor와 DummyVecEnv로 환경 래핑
        dummy_env = DummyEnv(self)
        env = Monitor(dummy_env, log_dir)
        env = DummyVecEnv([lambda: env])

        # 신경망 아키텍처 설정 (강화학습 개선)
        policy_kwargs = {
            # 정책 및 가치 네트워크 아키텍처
            "net_arch": [256, 256],
            # 활성화 함수
            "activation_fn": torch.nn.GELU,

            # # 커스텀 추출기
            # "features_extractor_class": CustomFeatureExtractor,
            # # 추출기 아키텍처
            # "features_extractor_kwargs": {
            #     "net_arch": [32, 32, 32]
            # },
            # "share_features_extractor": True,
        }

        if algorithm == 'sac':
            # SAC 하이퍼파라미터 설정
            buffer_size             = self.max_step // 5
            learning_rate           = 3e-4
            batch_size              = 256
            learning_starts         = 5000
            n_envs                  = 1
            tau                     = 0.005     # 타겟 네트워크 업데이트 속도
            gamma                   = 0.995     # 할인 인수
            ent_coef                = "auto"    # 엔트로피 계수
            train_freq              = 1         # 훈련 빈도
            gradient_steps          = 1         # 그라디언트 스텝
            target_update_interval  = 1         # 타겟 업데이트 간격

            # 공유 리플레이 버퍼 생성
            shared_buffer = ReplayBuffer(
                buffer_size=buffer_size,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                device=self.device,
                n_envs=n_envs,
                handle_timeout_termination=False
            )

            # 커스텀 SAC 모델 생성
            model = SACVehicleAlgorithm(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                buffer_size=0,
                learning_starts=learning_starts,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                ent_coef=ent_coef,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                target_update_interval=target_update_interval,
                verbose=1,
                tensorboard_log=log_dir,
                device=self.device
            )

            # 커스텀 리플레이 버퍼를 사용하도록 모델 설정
            model.replay_buffer = shared_buffer
        elif algorithm == 'ppo':
            # PPO 하이퍼파라미터 설정
            learning_rate = 3e-4
            n_steps       = 1024
            batch_size    = 256
            n_epochs      = 10
            gamma         = 0.995
            gae_lambda    = 0.95
            clip_range    = 0.2     # 클리핑 범위
            ent_coef      = 0.01    # 엔트로피 보너스, 정책 탐험 장려
            vf_coef       = 0.5     # 가치 함수 손실 계수
            max_grad_norm = 0.5     # 그라디언트 클리핑

            # 커스텀 PPO 모델 생성
            model = PPOVehicleAlgorithm(
                policy="MlpPolicy",
                env=env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir,
                device=self.device,
            )

        # 환경 정보는 learn() 내부에서 설정됨

        # 로거 설정
        model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))

        # 모델 출력
        print(model.policy)

        # 최적화된 콜백 시스템 설정
        callback_config = {
            'algorithm': algorithm,
            'num_vehicles': self.num_vehicles,
            'num_static_obstacles': self.num_static_obstacles,
            'num_dynamic_obstacles': self.num_dynamic_obstacles,
            'max_episode_steps': self.max_episode_steps,
            'max_episodes_history': self.rl_callback_config['max_episodes_history'],
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
                progress_bar=True
            )

            # 학습된 모델 저장
            if algorithm == 'sac':
                model.save(os.path.join(models_dir, "sac_final"))
                model.save_replay_buffer(os.path.join(models_dir, "sac_final_replay_buffer"))
            elif algorithm == 'ppo':
                model.save(os.path.join(models_dir, "ppo_final"))

            # 학습 완료 후 추가 로그 저장
            print("\n===== 학습 완료 =====")

            # 성능 콜백에서 학습 시간 정보 가져오기
            performance_callback = None
            for cb in callbacks:
                if hasattr(cb, 'training_start_time'):
                    performance_callback = cb
                    break

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

    def test(self, algorithm='sac'):
        """
        학습된 모델을 테스트하는 함수
        """
        # 모델 경로 설정
        model_path = f"./logs/checkpoints/{algorithm}_final.zip"
        if not os.path.exists(model_path):
            print(f"모델 파일이 존재하지 않습니다: {model_path}")
            return

        max_episode = self.env.config['simulation']['eval_episode']

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

            # 한 에피소드 결과 출력
            print(f"[Test] Episode {ep}/{max_episode} — Steps: {steps}, Total Reward: {total_reward:.2f}")

        # 시뮬레이터 종료
        self.close()
