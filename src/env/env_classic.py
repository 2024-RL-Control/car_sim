# -*- coding: utf-8 -*-
import os
import sys
import pygame
import numpy as np

from ..env.env_rl import BasicRLDrivingEnv
from src.model.classic import ClassicController
from src.utils.config_utils import load_config

class ClassicDrivingEnv():
    def __init__(self, config_path=None, config: dict = None):
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)
        self.rl_env = BasicRLDrivingEnv(config_path=config_path, config=self.config, verbose=0)
        self.rl_env.deactivate_action_controller()
        self.num_vehicles = self.rl_env.num_vehicles
        self.dt = self.rl_env.env.fixed_dt

        self.controllers = []
        for _ in range(self.num_vehicles):
            controller = ClassicController(
                road_api=self.rl_env.env.road_manager,
                vehicle_config=self.config['vehicle'],
                control_config=self.config['classic']['control'],
                planning_config=self.config['classic']['planning'],
                dt=self.dt
            )
            self.controllers.append(controller)

        self.best_trajectory_color = (0, 255, 100)  # 녹색
        self.candidate_trajectory_color = (100, 100, 100)  # 회색

    def reset(self):
        observations, active_agents = self.rl_env.reset()
        for controller in self.controllers:
            controller.reset()
        return observations, active_agents

    def step(self, actions: np.ndarray):
        """
        시뮬레이션 스텝 실행 (RL 래퍼의 step 사용)

        Args:
            actions: (num_vehicles, action_dim) 형태의 numpy 배열

        Returns:
            (observations, average_reward, env_done, truncated, info)
        """
        return self.rl_env.step(actions)

    def render(self):
        self.rl_env.render()
        if self.rl_env.env.config['visualization']['debug_mode']:
            screen = self.rl_env.env.renderer.screen
            world_to_screen_func = self.rl_env.env.camera.world_to_screen
            vehicle_manager = self.rl_env.env.get_vehicle_manager()
            active_idx = vehicle_manager.get_active_vehicle_index()
            if active_idx < len(self.controllers):
                controller = self.controllers[active_idx]

                # 최적 궤적 및 후보 궤적 가져오기
                best_traj, candidate_trajs = controller.get_trajectory()
                if best_traj and candidate_trajs:
                    # 후보 궤적을 화면에 렌더링
                    for traj in candidate_trajs:
                        candidate_world_points = [(wp.x, wp.y) for wp in traj.global_path]
                        if len(candidate_world_points) >= 2:
                            screen_points = [world_to_screen_func(p) for p in candidate_world_points]
                            pygame.draw.lines(screen, self.candidate_trajectory_color, False, screen_points, 2)
                    # 최적 궤적을 화면에 렌더링
                    best_world_points = [(wp.x, wp.y) for wp in best_traj.global_path]
                    if len(best_world_points) >= 2:
                        best_screen_points = [world_to_screen_func(p) for p in best_world_points]
                        pygame.draw.lines(screen, self.best_trajectory_color, False, best_screen_points, 2)

            pygame.display.flip()

    def handle_keyboard_input(self):
        """
        키보드 입력 처리
        """
        self.rl_env.handle_keyboard_input()

    def close(self):
        self.rl_env.close()

    def run(self):
        num_episodes = self.config['simulation']['rl']['eval_episode']
        vehicle_manager = self.rl_env.env.get_vehicle_manager()
        obstacle_manager = self.rl_env.env.get_obstacle_manager()

        for ep in range(1, num_episodes + 1):
            observations, active_agents = self.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                # 키보드 입력 수집 (ESC 등)
                self.rl_env.handle_keyboard_input()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return False

                vehicles = vehicle_manager.get_all_vehicles()
                active_vehicle_idx = vehicle_manager.get_active_vehicle_index()

                objects = []
                for vehicle in vehicles:
                    if vehicle.id != active_vehicle_idx:
                        objects.extend(vehicle.get_outer_circles_world())
                obstacles_colliders = obstacle_manager.get_all_outer_circles()
                if obstacles_colliders:
                    objects.extend(obstacles_colliders)
                if objects:
                    objects = np.array(objects)
                else:
                    objects = np.empty((0, 3))

                current_actions = np.zeros((self.num_vehicles, 2))
                for i, vehicle in enumerate(vehicles):
                    # BasicRLDrivingEnv가 관리하는 생존 상태(active_agents)인지 확인
                    if not active_agents[i]:
                        continue # 충돌/완료된 차량은 건너뜀

                    # 현재 차량이 UI에서 선택된 활성 차량(active_vehicle_idx)인지 확인
                    if i == active_vehicle_idx:
                        # 활성 차량만 고전 제어 로직을 실행
                        vehicle_state = vehicle.get_state()
                        road_boundary_colliders = vehicle_state.closest_segment.get_boundary()
                        if len(road_boundary_colliders) > 0:
                            objects = np.vstack((objects, road_boundary_colliders))
                        controller = self.controllers[i]
                        action = controller.plan_and_control(vehicle_state, objects)
                        current_actions[i] = action

                observations, avg_reward, done, truncated, info = self.step(current_actions)
                total_reward += avg_reward
                steps += 1

                self.render()

        self.close()
