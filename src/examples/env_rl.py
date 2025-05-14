# -*- coding: utf-8 -*-
import pygame
import numpy as np
import random
import math
from math import pi, cos, sin
from src.env.env import CarSimulatorEnv

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

        self.num_episodes = self.env.config['simulation']['num_episodes']
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
                'y_min': self.boundary['y_min'],
                'y_max': self.obstacle_area['y_max'] - self.buffer_distance
            },
            'bottom': {
                'x_min': self.boundary['x_min'],
                'x_max': self.boundary['x_max'],
                'y_min': self.obstacle_area['y_min'] + self.buffer_distance,
                'y_max': self.boundary['y_max']
            }
        }

        # 초기화 시 장애물 및 목적지 설정
        self.setup_environment()

        # 스텝 카운터
        self.steps = 0
        self.max_steps = self.env.config['simulation']['max_steps']

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
            size = random.uniform(1.0, 3.0)

            # 장애물 유형 랜덤 선택 (원형, 정사각형, 직사각형)
            obstacle_type = random.choice(['circle', 'square', 'rectangle'])

            if obstacle_type == 'circle':
                obstacle_manager.add_circle_obstacle(None, x, y, 0, 0, 0, size, (200, 0, 0))
            elif obstacle_type == 'square':
                obstacle_manager.add_square_obstacle(None, x, y, 0, 0, 0, size, (0, 200, 0))
            else:  # rectangle
                width = random.uniform(1.5, 4.0)
                height = random.uniform(1.0, 2.5)
                obstacle_manager.add_rectangle_obstacle(None, x, y, random.uniform(0, 2*pi), 0, 0, width, height, (0, 0, 200))

        # 동적 장애물 배치
        for _ in range(self.num_dynamic_obstacles):
            x = random.uniform(self.obstacle_area['x_min'], self.obstacle_area['x_max'])
            y = random.uniform(self.obstacle_area['y_min'], self.obstacle_area['y_max'])

            # 랜덤 속도, 방향 및 크기
            speed = random.uniform(1.0, 3.0)
            direction = random.uniform(0, 2*pi)
            size = random.uniform(1.0, 2.0)
            yaw_rate = random.uniform(-0.3, 0.3)  # 회전 속도

            # 랜덤 색상
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

            # 장애물 유형 랜덤 선택
            obstacle_type = random.choice(['circle', 'square'])

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
        for i in range(self.env.num_vehicles):
            placement_area = random.choice(['top', 'bottom', 'left', 'right'])
            boundary = self.vehicle_area[placement_area]
            x = random.uniform(boundary['x_min'], boundary['x_max'])
            y = random.uniform(boundary['y_min'], boundary['y_max'])
            yaw_volatility = random.uniform(-pi/6, pi/6)

            if placement_area == 'top':
                yaw = pi/2 + yaw_volatility  # 아래쪽 방향
            elif placement_area == 'bottom':
                yaw = -pi/2 + yaw_volatility  # 위쪽 방향
            elif placement_area == 'left':
                yaw = 0 + yaw_volatility  # 오른쪽 방향
            else:  # right
                yaw = pi + yaw_volatility  # 왼쪽 방향

            # 새 차량 생성
            self.env.vehicle_manager.create_vehicle(x=x, y=y, yaw=yaw, vehicle_id=i)
            print(placement_area)

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
        for i in range(self.env.num_vehicles):
            vehicle_placement = self.vehicle_start_position[i]['placement']
            yaw_volatility = random.uniform(-pi/6, pi/6)
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
            self.env.add_goal_for_vehicle(i, x, y, yaw, radius=2.0, color=(0, 255, 0))

    def reset(self):
        """
        환경 초기화 및 초기 관측값 반환

        Returns:
            초기 관측값
        """
        # 환경 초기화
        obs = self.env.reset()

        # 장애물, 차량, 목적지 다시 설정
        self.setup_environment()

        # 스텝 카운터 초기화
        self.steps = 0

        # 에피소드 카운터 증가
        self.episode_count += 1

        # 초기 관측값 반환
        return obs

    def step(self, action):
        """
        환경에서 한 스텝 진행

        Args:
            action: 에이전트의 행동 [가속도, 조향]

        Returns:
            obs: 관측값
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        # 환경에서 스텝 진행
        obs, reward, done, info = self.env.step(action)

        # 스텝 카운터 증가
        self.steps += 1

        # 최대 스텝 수 초과시 종료
        if self.steps >= self.max_steps:
            done = True
            info['timeout'] = True

        return obs, reward, done, info

    def render(self, mode='human'):
        """
        환경 렌더링
        """
        return self.env.render(mode)

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
        # 에피소드 스텝을 관리하는 변수
        episode_rewards = []
        terminated = False

        for _ in range(self.num_episodes):
            if terminated:
                break

            # 환경 초기화
            obs_array = self.reset()

            # 에피소드 정보
            total_reward = 0
            done = False

            print(f"Episode {self.episode_count}/{self.num_episodes}")

            while not done and not terminated:
                # 키보드 입력 처리
                self.handle_keyboard_input()

                # 랜덤 액션 테스트 (실제 학습에서는 에이전트가 결정, 다중 차량 고려)
                actions = []
                for _ in range(self.env.num_vehicles):
                    # throttle_engine, throttle_brake, steering
                    actions.append(np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(-1, 1)]))

                # 환경에서 한 스텝 진행
                obs_array, reward_array, done, info = self.step(np.array(actions))

                # 보상 누적
                for reward in reward_array:
                    total_reward += reward

                # info 속 collisions, outside_roads, reached_targets(id, value) 활용, 충돌 or 도착 or 도로 벗어날 시 에이전트 조작 및 학습 사용 X

                # 렌더링
                self.render()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        terminated = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            terminated = True
                            break

                # 종료 확인
                if done:
                    print(f"  Step: {self.steps}, Total Reward: {total_reward:.2f}")
                    episode_rewards.append(total_reward)
                    break

        # 환경 종료
        self.close()

        # 학습 결과 출력
        if not terminated:
            print(f"\nTraining completed for {self.num_episodes} episodes.")
            print(f"Average reward: {np.mean(episode_rewards):.2f}")
        else:
            print("Training terminated by user.")

        return episode_rewards
