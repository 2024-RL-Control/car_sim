# -*- coding: utf-8 -*-
import pygame
import numpy as np
import random
from math import pi
from src.env.env import CarSimulatorEnv
import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

class BasicRLDrivingEnv:
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

        self.num_episodes = self.env.config['simulation']['num_episodes']
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
        for _ in range(self.num_static_obstacles):
            # 장애물 배치 가능 공간 내에 랜덤 위치 선택
            x = random.uniform(self.obstacle_area['x_min'], self.obstacle_area['x_max'])
            y = random.uniform(self.obstacle_area['y_min'], self.obstacle_area['y_max'])

            # 랜덤 크기의 장애물
            size = random.uniform(1.0, 10.0)

            # 장애물 유형 랜덤 선택 (원형, 정사각형, 직사각형)
            obstacle_type = random.choice(['circle', 'square', 'rectangle'])
            color = (100, 100, 200)

            if obstacle_type == 'circle':
                obstacle_manager.add_circle_obstacle(None, x, y, 0, 0, 0, size, color)
            elif obstacle_type == 'square':
                obstacle_manager.add_square_obstacle(None, x, y, random.uniform(-pi/2, pi/2), 0, 0, size, color)
            else:  # rectangle
                width = random.uniform(1.5, 10.0)
                height = random.uniform(1.0, 3.5)
                obstacle_manager.add_rectangle_obstacle(None, x, y, random.uniform(-pi/2, pi/2), 0, 0, width, height, color)

        # 동적 장애물 배치
        for _ in range(self.num_dynamic_obstacles):
            x = random.uniform(self.obstacle_area['x_min'], self.obstacle_area['x_max'])
            y = random.uniform(self.obstacle_area['y_min'], self.obstacle_area['y_max'])

            # 랜덤 속도, 방향 및 크기
            speed = random.uniform(1.0, 10.0)
            direction = random.uniform(-pi/2, pi/2)
            size = random.uniform(1.0, 10.0)
            yaw_rate = random.uniform(-0.4, 0.4)  # 회전 속도

            # 장애물 유형 랜덤 선택
            obstacle_type = random.choice(['circle', 'square'])
            color = (200, 100, 100)

            if obstacle_type == 'circle':
                obstacle_manager.add_circle_obstacle(None, x, y, direction, yaw_rate, speed, size, color)
            else:  # square
                obstacle_manager.add_square_obstacle(None, x, y, direction, yaw_rate, speed, size, color)

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

    def reset(self):
        """
        환경 초기화 및 초기 관측값 반환

        Returns:
            observations: 차량별 초기 관측 값 (num_vehicles, obs_dim) [x, y, cos(yaw), sin(yaw), vel_long, vel_lat, distance_to_target, yaw_diff_to_target]
        """
        # 환경 초기화
        observations = self.env.reset()

        # 장애물, 차량, 목적지 다시 설정
        self.setup_environment()

        # 에이전트 활성화 상태 초기화
        self.active_agents = [True] * self.num_vehicles

        # 스텝 카운터 초기화
        self.steps = 0

        # 에피소드 카운터 증가
        self.episode_count += 1

        # 초기 관측값 반환
        return observations

    def step(self, actions):
        """
        환경에서 한 스텝 진행

        Args:
            actions: 모든 차량의 행동 (num_vehicles, 3)

        Returns:
            observations: 모든 차량의 관측값 (num_vehicles, obs_dim)
            rewards: 모든 차량의 보상값 (num_vehicles,)
            done: 종료 여부 bool
            info: 추가 정보
        """
        # 환경에서 스텝 진행
        observations, rewards, done, info = self.env.step(actions)

        # 스텝 카운터 증가
        self.steps += 1

        # 차량 비활성화
        for vehicle_id in range(self.num_vehicles):
            vehicle_done = info['dones'].get(vehicle_id, False)

            if vehicle_done and self.active_agents[vehicle_id]:
                self.active_agents[vehicle_id] = False

        return observations, rewards, done, info

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

    def train(self):
        """
        자율주행 에이전트 학습 함수
        """
        self.print_basic_controls()

        # 로그 및 모델 저장 경로 설정
        models_dir = "./logs/model"
        log_dir = "./logs/train"

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        env = Monitor(self.env, log_dir)
        env = DummyVecEnv([lambda: env])

        # SAC 하이퍼파라미터 설정
        buffer_size = 100000
        learning_rate = 3e-4
        batch_size = 256
        learning_starts = 10000

        # 공유 리플레이 버퍼 생성
        shared_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            n_envs=1,
            handle_timeout_termination=False
        )

        # 모든 차량을 위한 단일 SAC 모델 생성
        model = SAC(
            "MlpPolicy",                # 다층 퍼셉트론 정책 사용
            env,                        # 환경
            learning_rate=learning_rate, # 학습률
            buffer_size=buffer_size,    # 리플레이 버퍼 크기
            learning_starts=learning_starts, # 학습 시작 전 환경 탐색 스텝 수
            batch_size=batch_size,      # 미니배치 크기
            tau=0.005,                  # 타겟 네트워크 소프트 업데이트 계수
            gamma=0.99,                 # 할인 계수
            train_freq=1,               # 업데이트 빈도
            gradient_steps=1,           # 그래디언트 업데이트 스텝 수
            ent_coef="auto",            # 엔트로피 계수 자동 조정
            target_update_interval=1,   # 타겟 네트워크 업데이트 주기
            verbose=1,                  # 로그 출력
            tensorboard_log=log_dir,    # 텐서보드 로그 저장 경로
            device=self.device          # 사용할 디바이스
        )

        # 커스텀 리플레이 버퍼를 사용하도록 모델 설정
        model.replay_buffer = shared_buffer
        model._logger = configure(log_dir, ['stdout', 'csv', 'tensorboard'])

        # 학습 콜백 설정
        checkpoint_callback = CheckpointCallback(
            save_freq=100,              # 100 타임스텝마다 저장
            save_path=models_dir,
            name_prefix="basic_sac",
            verbose=1
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=models_dir,
            eval_freq=100,
            verbose=1
        )
        callback = CallbackList([checkpoint_callback, eval_callback])

        # 에피소드 관리 변수
        episode_rewards = []
        terminated = False
        total_timesteps = 0

        for _ in range(self.num_episodes):
            if terminated:
                break

            # 환경 초기화
            try:
                observations = self.reset()
                actions = np.array([np.zeros(3) for _ in range(self.num_vehicles)])
            except Exception as e:
                print(f"Error resetting environment: {e}")
                continue

            # 에피소드 정보
            total_reward = 0
            done = False

            prev_observations = observations.copy()

            while not done and not terminated:
                # 키보드 입력 처리
                self.handle_keyboard_input()

                if self.steps == 0:
                    observations, _, _, _ = self.step(actions)
                    prev_observations = observations.copy()

                # 각 차량별 행동 결정
                for i in range(self.num_vehicles):
                    if self.active_agents[i]:
                        # 충분한 경험이 쌓이기 전에는 랜덤 행동
                        if total_timesteps < learning_starts:
                            actions[i] = self.env.action_space.sample()
                        else:
                            action, _ = model.predict(observations[i], deterministic=False)
                            actions[i] = action
                    else:
                        actions[i] = np.array([0.0, 0.0, 0.0])

                # 환경에서 한 스텝 진행
                next_observations, rewards, done, info = self.step(actions)
                total_timesteps += 1

                # 보상 누적
                step_reward = sum(rewards)
                total_reward += step_reward

                # 각 차량의 경험을 공유 리플레이 버퍼에 추가
                for i in range(self.num_vehicles):
                    if self.active_agents[i]:
                        # 단일 차량의 경험 생성 (obs, action, reward, next_obs, done)
                        shared_buffer.add(
                            obs=prev_observations[i],
                            action=actions[i],
                            reward=rewards[i],
                            next_obs=next_observations[i],
                            done=done,
                            infos=[{"terminal_observation": next_observations[i] if done else None}]
                        )

                # 충분한 데이터가 쌓이면 SAC 모델 학습
                if total_timesteps > learning_starts and total_timesteps % model.train_freq[0] == 0:
                    model.train(batch_size=batch_size, gradient_steps=1)

                # 다음 스텝을 위해 관측값 업데이트
                prev_observations = next_observations.copy()
                observations = next_observations

                # 렌더링
                self.render()

                # 이벤트 처리
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        terminated = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            terminated = True
                            break

                # 에피소드 종료 확인
                if done:
                    print(f"Episode {self.episode_count}/{self.num_episodes}, Steps: {self.steps}, Total Reward: {total_reward:.2f}")
                    episode_rewards.append(total_reward)
                    break

        # 학습된 모델 저장
        model.save(os.path.join(models_dir, "final_sac_model"))

        # 환경 종료
        self.close()

        # 학습 결과 출력
        if not terminated:
            print(f"\nTraining completed for {self.num_episodes} episodes.")
            print(f"Average reward: {np.mean(episode_rewards):.2f}")
            print(f"Total timesteps: {total_timesteps}")
        else:
            print("Training terminated by user.")
            print(f"Completed episodes: {self.episode_count}/{self.num_episodes}")
            print(f"Total timesteps: {total_timesteps}")

        return episode_rewards
