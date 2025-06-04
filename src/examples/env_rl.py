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
from ..model.sb3 import SACVehicleAlgorithm, PPOVehicleAlgorithm, CleanCheckpointCallback, CustomLoggingCallback, CustomFeatureExtractor, EnhancedFeatureExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CallbackList
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
        yaw_rates = np.random.uniform(-0.4, 0.4, size=n_dynamic)  # 회전 속도
        speeds = np.random.uniform(1.0, 10.0, size=n_dynamic)
        sizes = np.random.uniform(1.0, 5.0, size=n_dynamic)
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
            reward: 모든 차량의 평균 보상
            done: 종료 여부 bool
            info: 추가 정보
        """
        # 환경에서 스텝 진행
        observations, rewards, done, _, info = self.env.step(actions)
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

        info.update({
            'episode_count': self.episode_count,
            'rewards': rewards,  # 개별 차량별 보상
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

        # 신경망 아키텍처 설정
        policy_kwargs = {
            # 1) 기본 MLP-extractor 완전 비활성화
            "net_arch": [],
            "activation_fn": torch.nn.GELU,  # mu/q 헤드 내부 활성화

            # 2) extractor 클래스 지정
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": {
                "net_arch": [512, 512, 512, 256, 128, 64]
            },
            "share_features_extractor": True,
        }

        if algorithm == 'sac':
            # SAC 하이퍼파라미터 설정
            buffer_size = self.max_step // 5
            learning_rate = 3e-4
            batch_size = 256
            learning_starts = 100000
            n_envs = 1

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
                tau=0.003,
                gamma=0.995,
                ent_coef=0.1,
                train_freq=10,
                gradient_steps=5,
                target_update_interval=12,
                verbose=1,
                tensorboard_log=log_dir,
                device=self.device
            )

            # 커스텀 리플레이 버퍼를 사용하도록 모델 설정
            model.replay_buffer = shared_buffer
        elif algorithm == 'ppo':
            # PPO 하이퍼파라미터 설정
            n_steps       = 1024
            batch_size    = 64
            n_epochs      = 5
            gamma         = 0.99
            gae_lambda    = 0.95
            clip_range    = 0.2
            learning_rate = 3e-4

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
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir,
                device=self.device,
            )

        # 환경 정보 설정
        model.set_env_info(self)

        # 로거 설정
        model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

        # 모델 출력
        print(model.policy)

        # 체크포인트 콜백 설정
        clean_checkpoint_callback = CleanCheckpointCallback(
            save_freq=10000,  # 10000스텝마다 저장
            save_path=models_dir,
            name_prefix=algorithm,
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
