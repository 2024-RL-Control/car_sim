# -*- coding: utf-8 -*-
import pygame
import gymnasium as gym
import numpy as np
import random
from math import pi
from src.env.env import CarSimulatorEnv
import os
import sys
import torch
import time
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

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
        return observations[0], reward, done, truncated, info


class CleanCheckpointCallback(CheckpointCallback):
    """
    커스텀 체크포인트 콜백: 지정된 주기로 모델을 저장하고, 오래된 체크포인트를 삭제합니다.
    """
    def __init__(self, save_freq, save_path, name_prefix='model', max_checkpoints=5, **kwargs):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, **kwargs)
        self.max_checkpoints = max_checkpoints

    def _on_step(self):
        super_result = super()._on_step()

        # 현재 체크포인트 저장 후 초과된 것들 삭제
        if self.num_timesteps % self.save_freq == 0:
            files = sorted([f for f in os.listdir(self.save_path)
                            if f.startswith(self.name_prefix) and f.endswith('.zip')])
            # 오래된 체크포인트 삭제
            for old in files[:-self.max_checkpoints]:
                try: os.remove(os.path.join(self.save_path, old))
                except: pass

        return super_result

class CustomLoggingCallback(BaseCallback):
    """
    커스텀 로깅 콜백: 학습률, GPU 메모리 사용량, 에피소드 보상 등을 기록합니다.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_start = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.learning_rates = []
        self.gpu_memory_usage = []

    def _on_step(self):
        # GPU 메모리 사용량 추적 (가능한 경우)
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB 단위
            self.gpu_memory_usage.append(mem_allocated)

        # 현재 학습률 추적
        if hasattr(self.model, 'learning_rate'):
            current_lr = self.model.learning_rate
            self.learning_rates.append(current_lr)

        # 추가 로그 기록 (1000스텝마다)
        if self.n_calls % 1000 == 0:
            elapsed_time = time.time() - self.training_start
            steps_per_second = self.n_calls / elapsed_time
            est_remaining_time = (self.locals.get('total_timesteps', 0) - self.n_calls) / steps_per_second if steps_per_second > 0 else 0

            # 학습 진행상황 상세 로깅
            self.logger.record("time/steps_per_second", steps_per_second)
            self.logger.record("time/est_remaining_hours", est_remaining_time / 3600)

            if torch.cuda.is_available():
                self.logger.record("resources/gpu_memory_gb", mem_allocated)

            # 메모리 상태 확인 및 청소 (가능한 경우)
            if torch.cuda.is_available() and mem_allocated > 3.0:  # 3GB 이상 사용 시 메모리 청소
                torch.cuda.empty_cache()
                print("\nGPU 메모리 캐시 정리 완료")

        return True

class MultiVehicleAlgorithm(SAC):
    """
    다중 차량을 위한 커스텀 강화학습 알고리즘 클래스
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 추가 속성 설정
        self.rl_env = None
        self.num_vehicles = None
        self.prev_observations = None
        self.total_reward = 0
        self.episode_steps = 0

    def set_env_info(self, rl_env):
        """
        BasicRLDrivingEnv 정보 설정
        """
        self.rl_env = rl_env
        self.num_vehicles = rl_env.num_vehicles

    def _excluded_save_params(self) -> list[str]:
        # 제외된 저장 파라미터 목록
        return super()._excluded_save_params() + ["rl_env"]

    def _reset(self):
        # 환경 리셋 (성공할 때까지 무한 시도)
        reset_success = False
        reset_attempts = 0

        while not reset_success:
            try:
                prev_observations, _ = self.rl_env.reset()
                reset_success = True
                self.prev_observations = prev_observations
                self.total_reward = 0
                self.episode_steps = 0
            except Exception as e:
                reset_attempts += 1
                if "Failed to find rrt-dubins path" in str(e):
                    continue
                else:
                    # 다른 예외는 다시 발생시키기
                    raise

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        다중 차량을 위한 학습 메소드 오버라이드
        """
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        # 환경 초기화
        self._reset()

        while self.num_timesteps < total_timesteps:
            continue_training = self._custom_rollout_step(callback)
            if not continue_training:
                break

        callback.on_training_end()

        return self

    def _custom_rollout_step(self, callback) -> bool:
        """
        한 스텝 진행하고 경험 수집
        """
        # 1) 키보드 입력 및 이벤트 처리
        self.rl_env.handle_keyboard_input()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

        # 2) 렌더링
        self.rl_env.render()

        # 3) 행동 결정
        observations = self.prev_observations
        active_mask = np.array(self.rl_env.active_agents)
        if self._n_updates < self.learning_starts:
            # 랜덤 행동 (벡터화)
            actions = np.zeros((self.num_vehicles, 3))
            if active_mask.any():
                random_actions = np.array([self.rl_env.env.action_space.sample() for _ in range(active_mask.sum())])
                # random_actions[:, 1] = 0.0  # 브레이크 제거
                actions[active_mask] = random_actions
        else:
            with torch.no_grad():
                # 활성화된 에이전트만 행동 예측 (벡터화)
                actions = np.zeros((self.num_vehicles, 3))
                if active_mask.any():
                    # 활성화된 에이전트의 관측값만 선택
                    active_obs = observations[active_mask]
                    # 배치로 한번에 예측
                    active_obs_tensor = torch.as_tensor(active_obs, device=self.device)
                    actions_tensor, _ = self.policy.actor.action_log_prob(active_obs_tensor)
                    # NumPy 배열로 변환
                    actions[active_mask] = actions_tensor.cpu().numpy()

        # 4) 환경 스텝 & 보상 누적
        next_observations, reward, done, _, info = self.rl_env.step(actions)
        self.total_reward += reward
        self.episode_steps += 1
        self.num_timesteps += 1

        # 5) 활성화된 에이전트만 경험 저장
        active_idx = np.where(active_mask)[0]
        for idx in active_idx:
            self.replay_buffer.add(
                obs=self.prev_observations[idx],
                action=actions[idx],
                reward=info['rewards'][idx],
                next_obs=next_observations[idx],
                done=done,
                infos={"terminal_observation": next_observations[idx] if done else None}
            )

        # 6) 콜백 업데이트 및 종료 체크
        callback.update_locals(locals())
        if not callback.on_step():
            return False

        # 7) 학습 트리거
        if self.num_timesteps > self.learning_starts:
            self.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)

        # 8) 에피소드 종료 시 처리
        if done:
            # 로깅
            print(f"Episode {self.rl_env.episode_count}, "
                  f"Steps: {self.episode_steps}, "
                  f"Total Reward: {self.total_reward:.2f}")

            # 텐서보드 로깅
            self.logger.record("train/episode_reward", self.total_reward)
            self.logger.record("train/episode_length", self.episode_steps)
            self.logger.record("train/active_vehicles", sum(self.rl_env.active_agents))
            self.logger.dump(self.num_timesteps)

            # 환경 초기화
            self._reset()
        else:
            # 관측값 업데이트
            self.prev_observations = next_observations
        return True

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

        # 차량 배치 가능 공간
        self.vehicle_area = {
            'left': {
                'x_min': self.boundary['x_min'],
                'x_max': self.obstacle_area['x_min'] - self.buffer_distance,
                'y_min': self.boundary['y_min'],
                'y_max': self.boundary['y_max']
            },
            'right': {
                'x_min': self.obstacle_area['x_max'] + self.buffer_distance,
                'x_max': self.boundary['x_max'],
                'y_min': self.boundary['y_min'],
                'y_max': self.boundary['y_max']
            },
            'top': {
                'x_min': self.boundary['x_min'],
                'x_max': self.boundary['x_max'],
                'y_min': self.obstacle_area['y_max'] + self.buffer_distance,
                'y_max': self.boundary['y_max']
            },
            'bottom': {
                'x_min': self.boundary['x_min'],
                'x_max': self.boundary['x_max'],
                'y_min': self.boundary['y_min'],
                'y_max': self.obstacle_area['y_min'] - self.buffer_distance
            }
        }

        # 초기화 시 장애물 및 목적지 설정
        self.setup_environment()

        # 스텝 카운터
        self.steps = 0
        self.max_step = self.env.config['simulation']['train_max_steps']
        self.max_episode_steps = self.env.config['simulation']['train_max_episode_steps']

        # 에피소드 정보 저장용
        self.episode_count = 0

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
        sizes = np.random.uniform(1.0, 10.0, size=n_static)
        types = np.random.choice(['circle', 'square', 'rectangle'], size=n_static)
        static_color = (100, 100, 200)  # 정적 장애물 색상

        for x, y, yaw, sz, t in zip(xs, ys, yaws, sizes, types):
            if t == 'circle':
                obstacle_manager.add_circle_obstacle(None, x, y, 0, 0, 0, sz, static_color)
            elif t == 'square':
                obstacle_manager.add_square_obstacle(None, x, y, yaw, 0, 0, sz, static_color)
            else:  # rectangle
                width = random.uniform(1.5, 10.0)
                height = random.uniform(1.0, 3.5)
                obstacle_manager.add_rectangle_obstacle(None, x, y, yaw, 0, 0, width, height, static_color)

        # 동적 장애물 배치
        n_dynamic = self.num_dynamic_obstacles
        xs = np.random.uniform(self.obstacle_area['x_min'], self.obstacle_area['x_max'], size=n_dynamic)
        ys = np.random.uniform(self.obstacle_area['y_min'], self.obstacle_area['y_max'], size=n_dynamic)
        yaws = np.random.uniform(-pi/2, pi/2, size=n_dynamic)
        yaw_rates = np.random.uniform(-0.4, 0.4, size=n_dynamic)  # 회전 속도
        speeds = np.random.uniform(1.0, 10.0, size=n_dynamic)
        sizes = np.random.uniform(1.0, 10.0, size=n_dynamic)
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
            boundary = self.vehicle_area[placement_area]
            x = random.uniform(boundary['x_min'], boundary['x_max'])
            y = random.uniform(boundary['y_min'], boundary['y_max'])
            yaw_volatility = random.uniform(-pi/6, pi/6)

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
            yaw_volatility = random.uniform(-pi/6, pi/6)
            if vehicle_placement == 'top':
                boundary = self.vehicle_area['bottom']
                yaw = pi/2 + yaw_volatility
            elif vehicle_placement == 'bottom':
                boundary = self.vehicle_area['top']
                yaw = -pi/2 + yaw_volatility
            elif vehicle_placement == 'left':
                boundary = self.vehicle_area['right']
                yaw = pi + yaw_volatility
            else:  # right
                boundary = self.vehicle_area['left']
                yaw = 0 + yaw_volatility

            x = random.uniform(boundary['x_min'], boundary['x_max'])
            y = random.uniform(boundary['y_min'], boundary['y_max'])

            # 차량에 목적지 추가
            self.env.add_goal_for_vehicle(i, x, y, yaw, radius=2.0)

    def reset(self, *, seed=None, options=None):
        """
        환경 초기화 및 초기 관측값 반환

        Args:
            seed: 환경의 난수 생성기 시드
            options: 추가 옵션 딕셔너리

        Returns:
            observations: 차량별 초기 관측 값 (num_vehicles, obs_dim) [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, distance_to_target, yaw_diff_to_target]
        """
        # 환경 초기화
        _ = self.env.reset(seed=seed, options=options)

        # 장애물, 차량, 목적지 다시 설정
        self.setup_environment()

        # 에이전트 활성화 상태 초기화
        self.active_agents = [True] * self.num_vehicles

        # 스텝 카운터 초기화
        self.steps = 0

        # 에피소드 카운터 증가
        self.episode_count += 1

        # 초기 행동으로 0을 사용하여 첫 번째 스텝 실행
        actions = np.zeros((self.num_vehicles, 3))
        observations, _, _, _, _ = self.step(actions)

        # 초기 관측값 반환
        return observations, {}

    def step(self, actions):
        """
        환경에서 한 스텝 진행

        Args:
            actions: 모든 차량의 행동 (num_vehicles, 3)

        Returns:
            observations: 모든 차량의 관측값 (num_vehicles, obs_dim)
            reward: 모든 차량의 보상값의 합
            done: 종료 여부 bool
            info: 추가 정보
        """
        # 환경에서 스텝 진행
        observations, rewards, done, _, info = self.env.step(actions)

        # 스텝 카운터 증가
        self.steps += 1

        # 차량 비활성화
        for vehicle_id in range(self.num_vehicles):
            vehicle_done = info['dones'].get(vehicle_id, False)

            if vehicle_done and self.active_agents[vehicle_id]:
                self.active_agents[vehicle_id] = False

        if self.steps >= self.max_episode_steps:
            done = True

        info.update({
            'episode_count': self.episode_count,
            'rewards': rewards,  # 개별 차량별 보상
        })

        return observations, np.sum(rewards), done, False, info

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

    def learn(self):
        """
        자율주행 에이전트 학습 함수 (learn() 메소드 사용)
        """
        # 로그 및 모델 저장 경로 설정
        models_dir = "./logs/model"
        log_dir = "./logs/train"

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Monitor와 DummyVecEnv로 환경 래핑
        dummy_env = DummyEnv(self)
        env = Monitor(dummy_env, log_dir)
        env = DummyVecEnv([lambda: env])

        # SAC 하이퍼파라미터 설정
        buffer_size = 500000
        learning_rate = 3e-3
        batch_size = 256
        learning_starts = 30000
        n_envs = 1

        # 학습률 스케줄링 함수 정의
        def lr_schedule(progress_remaining):
            return learning_rate * progress_remaining  # 학습 진행에 따라 학습률 감소

        # 공유 리플레이 버퍼 생성
        shared_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            n_envs=n_envs,
            handle_timeout_termination=False
        )

        # 신경망 아키텍처 설정
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 256, 256, 256, 64], qf=[128, 256, 256, 256, 64, 32]),
            activation_fn=torch.nn.LeakyReLU
        )

        # 커스텀 SAC 모델 생성
        model = MultiVehicleAlgorithm(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            buffer_size=0,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=5,
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=10,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=self.device
        )

        # 커스텀 리플레이 버퍼를 사용하도록 모델 설정
        model.replay_buffer = shared_buffer

        # 환경 정보 설정
        model.set_env_info(self)

        # 로거 설정
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        # 체크포인트 콜백 설정
        clean_checkpoint_callback = CleanCheckpointCallback(
            save_freq=10000,  # 10000스텝마다 저장
            save_path=models_dir,
            name_prefix="sac",
            save_replay_buffer=False,  # 체크포인트에서는 리플레이 버퍼를 저장하지 않음
            save_vecnormalize=True,
            max_checkpoints=10  # 최대 10개의 체크포인트만 유지
        )

        # 고급 로깅 콜백
        logging_callback = CustomLoggingCallback()

        # 콜백 목록 생성
        callback = CallbackList([clean_checkpoint_callback, logging_callback])

        # learn() 메소드로 학습 시작
        try:
            # 기본 컨트롤 정보 출력
            self.print_basic_controls()

            model.learn(
                total_timesteps=self.max_step,
                callback=callback,
                log_interval=100,
                progress_bar=True
            )

            # 학습된 모델 저장
            model.save(os.path.join(models_dir, "sac_final"))
            model.save_replay_buffer(os.path.join(models_dir, "sac_final_replay_buffer"))

            # 학습 완료 후 추가 로그 저장
            print("\n===== 학습 완료 =====")
            print(f"총 학습 시간: {(time.time() - logging_callback.training_start)/3600:.2f} 시간")

            if torch.cuda.is_available():
                print(f"최대 GPU 메모리 사용량: {max(logging_callback.gpu_memory_usage):.2f} GB")
                # 메모리 정리
                torch.cuda.empty_cache()

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(f"\n\n학습 중단됨: {e}")
            print(f'file name: {str(fname)}')
            print(f'error type: {str(exc_type)}')
            print(f'error msg: {str(e)}')
            print(f'line number: {str(exc_tb.tb_lineno)}')
            # 오류 발생시에도 현재까지 학습된 모델 저장 시도
            try:
                model.save(os.path.join(models_dir, "sac_interrupted"))
                model.save_replay_buffer(os.path.join(models_dir, "sac_interrupted_replay_buffer"))
                print("중단된 모델 상태가 저장되었습니다.")
            except Exception as save_error:
                print(f"중단된 모델 저장 실패: {save_error}")
        finally:
            # 환경 종료
            self.close()
