# -*- coding: utf-8 -*-
import os
import sys
import pygame
import numpy as np
import re
import time
from datetime import datetime
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.env_rl import BasicRLDrivingEnv, DummyEnv
from src.model.classic import ClassicController
from src.model.sb3 import SACVehicleAlgorithm, PPOVehicleAlgorithm
from src.utils.config_utils import load_config

class TestComparisonEnv:
    """
    RL 에이전트와 Classic 컨트롤러를 비교 테스트하기 위한 환경
    """

    def __init__(self, config_path: str, rl_algorithm: str = 'sac'):
        """
        비교 테스트 환경 초기화

        Args:
            config_path: 설정 파일 경로
            rl_algorithm: 로드할 RL 알고리즘 ('sac' 또는 'ppo')
        """
        print("비교 테스트 환경을 초기화합니다...")
        self.config = load_config(config_path)
        self.rl_algorithm_name = rl_algorithm.lower()

        # --- 1. 차량 대수를 2로 강제 설정 ---
        original_num_vehicles = self.config['simulation']['num_vehicles']
        if original_num_vehicles != 2:
            print(f"경고: 설정 파일의 차량 대수({original_num_vehicles})를 2로 강제 변경합니다.")
            self.config['simulation']['num_vehicles'] = 2

        # --- 2. 기본 RL 환경 래퍼 초기화 ---
        self.rl_env = BasicRLDrivingEnv(config_path=config_path, verbose=0)
        # 공정한 비교를 위해 ActionController 비활성화, 두 알고리즘 모두 매 시뮬레이션 스텝마다 행동을 결정
        self.rl_env.deactivate_action_controller()

        # 시뮬레이션 핵심 객체 참조
        self.vehicle_manager = self.rl_env.env.get_vehicle_manager()
        self.obstacle_manager = self.rl_env.env.get_obstacle_manager()
        self.dt = self.rl_env.env.fixed_dt

        # --- 3. 차량 0: 강화학습(RL) 모델 로드 ---
        print(f"차량 0 (RL)을 위해 '{self.rl_algorithm_name}' 모델을 로드합니다...")
        self.rl_model = self._load_rl_model(self.rl_algorithm_name)
        if self.rl_model is None:
            raise FileNotFoundError(f"로드할 '{self.rl_algorithm_name}' 모델을 찾을 수 없습니다.")

        # --- 4. 차량 1: 고전(Classic) 컨트롤러 초기화 ---
        print("차량 1 (Classic)을 위해 'ClassicController'를 초기화합니다...")
        self.classic_controller = ClassicController(
            road_api=self.rl_env.env.road_manager,
            vehicle_config=self.config['vehicle'],
            control_config=self.config['classic']['control'],
            planning_config=self.config['classic']['planning'],
            dt=self.dt
        )
        # 고전 제어기 궤적 시각화 색상 설정 (env_classic.py 참조)
        self.best_trajectory_color = (0, 255, 100)
        self.candidate_trajectory_color = (100, 100, 100)

        # --- 5. TensorBoard 로거 초기화 ---
        current_time = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        log_dir = f"./logs/test/{self.rl_algorithm_name}_vs_classic_{current_time}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def _find_latest_model(self, models_dir: str) -> Tuple[str, str]:
        """
        지정된 디렉토리에서 가장 최근에 학습된 모델 경로를 찾습니다.
        (env_rl.py의 _find_latest_model 로직과 유사하게 작동)

        수정: 'final' 모델 대신 'best' 또는 가장 스텝이 높은 모델을 우선 탐색
        """
        if not os.path.exists(models_dir):
            return None

        files = os.listdir(models_dir)
        algorithm = self.rl_algorithm_name

        # 1. 우선순위 1: best.zip (가장 성능 좋은 모델)
        best_model = os.path.join(models_dir, f"{algorithm}_best.zip")
        if best_model in [os.path.join(models_dir, f) for f in files]:
            print(f"발견된 모델 (Best): {best_model}")
            return best_model

        # 2. 우선순위 2: 가장 높은 step
        step_models = []
        for f in files:
            match = re.match(rf"^{re.escape(algorithm)}_(\d+)_steps\.zip$", f)
            if match:
                steps = int(match.group(1))
                step_models.append((steps, os.path.join(models_dir, f)))

        if step_models:
            step_models.sort(key=lambda x: x[0], reverse=True)
            latest_step_model_path = step_models[0][1]
            print(f"발견된 모델 (Latest Step): {latest_step_model_path}")
            return latest_step_model_path

        # 3. 우선순위 3: final.zip
        final_model = os.path.join(models_dir, f"{algorithm}_final.zip")
        if final_model in [os.path.join(models_dir, f) for f in files]:
            print(f"발견된 모델 (Final): {final_model}")
            return final_model

        return None

    def _load_rl_model(self, algorithm: str):
        """
        가장 최근에 학습된 RL 모델을 로드합니다.
        (env_rl.py의 test() 메소드 로직 기반)
        """
        checkpoints_dir = "./logs/checkpoints/"
        model_path = None

        if not os.path.exists(checkpoints_dir):
            print(f"체크포인트 디렉토리가 존재하지 않습니다: {checkpoints_dir}")
            return None

        try:
            # algorithm 이름으로 시작하는 모든 디렉토리 찾기
            algo_dirs = [d for d in os.listdir(checkpoints_dir)
                         if d.startswith(f"{algorithm}_") and
                         os.path.isdir(os.path.join(checkpoints_dir, d))]

            if not algo_dirs:
                raise FileNotFoundError(f"'{checkpoints_dir}'에서 '{algorithm}' 알고리즘 기록을 찾을 수 없습니다.")

            # 시간순으로 정렬하여 가장 최근 디렉토리 선택
            latest_run_dir_name = max(algo_dirs)
            latest_run_dir_path = os.path.join(checkpoints_dir, latest_run_dir_name)

            print(f"가장 최근 학습 디렉토리에서 모델을 찾습니다: {latest_run_dir_path}")
            model_path = self._find_latest_model(latest_run_dir_path)

        except Exception as e:
            print(f"최근 모델을 찾는 중 오류 발생: {e}")
            return None

        if not model_path or not os.path.exists(model_path):
            print(f"모델 파일이 존재하지 않습니다: {model_path}")
            return None

        # 모델 로드를 위해 더미 환경 래핑
        dummy_env = DummyEnv(self.rl_env)
        env = DummyVecEnv([lambda: dummy_env])

        # 모델 로드
        ModelClass = SACVehicleAlgorithm if algorithm == 'sac' else PPOVehicleAlgorithm

        try:
            model = ModelClass.load(
                model_path,
                env=env,
                device=self.rl_env.device
            )
            model.policy.eval() # 평가 모드
            print(f"성공: {os.path.basename(model_path)} 모델 로드 완료")
            return model
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_classic_obstacles(self, active_vehicle_id: int) -> np.ndarray:
        """
        고전 컨트롤러(차량 1)를 위한 장애물 목록을 생성합니다.
        여기에는 다른 모든 차량(차량 0)과 환경 내 장애물이 포함됩니다.
        """
        objects = []

        # 1. 다른 차량(Vehicle 0)을 장애물로 추가
        for vehicle in self.vehicle_manager.get_all_vehicles():
            if vehicle.id != active_vehicle_id:
                # BasicRLDrivingEnv가 관리하는 생존 상태(active_agents) 확인
                if self.rl_env.active_agents[vehicle.id]:
                    objects.extend(vehicle.get_outer_circles_world())

        # 2. 환경 내 정적/동적 장애물 추가
        obstacles_colliders = self.obstacle_manager.get_all_outer_circles()
        if obstacles_colliders:
            objects.extend(obstacles_colliders)

        return np.array(objects) if objects else np.empty((0, 3))

    def reset(self):
        """환경 리셋"""
        print("환경 리셋...")
        observations, active_agents = self.rl_env.reset()
        self.classic_controller.reset()
        return observations, active_agents

    def handle_keyboard_input(self):
        """키보드 입력 처리"""
        self.rl_env.handle_keyboard_input()

    def render(self):
        """환경 렌더링 (고전 제어기 궤적 포함)"""
        # 1. 기본 RL 환경 렌더링
        self.rl_env.render()

        # 2. 고전 제어기(차량 1) 궤적 렌더링 (디버그 모드 활성화 시)
        if self.rl_env.env.config['visualization']['debug_mode']:
            # 차량 1이 활성 상태일 때만 궤적을 그림
            if self.rl_env.active_agents[1]:
                screen = self.rl_env.env.renderer.screen
                world_to_screen_func = self.rl_env.env.camera.world_to_screen

                # 최적 궤적 및 후보 궤적 가져오기
                best_traj, candidate_trajs = self.classic_controller.get_trajectory()

                if best_traj and candidate_trajs:
                    # 후보 궤적 렌더링
                    for traj in candidate_trajs:
                        world_points = [(wp.x, wp.y) for wp in traj.global_path]
                        if len(world_points) >= 2:
                            screen_points = [world_to_screen_func(p) for p in world_points]
                            pygame.draw.lines(screen, self.candidate_trajectory_color, False, screen_points, 2)

                    # 최적 궤적 렌더링
                    best_world_points = [(wp.x, wp.y) for wp in best_traj.global_path]
                    if len(best_world_points) >= 2:
                        best_screen_points = [world_to_screen_func(p) for p in best_world_points]
                        pygame.draw.lines(screen, self.best_trajectory_color, False, best_screen_points, 2)

            # Pygame 디스플레이 업데이트 (render()가 이미 호출했으므로 중복 호출 피함)
            # pygame.display.flip() # rl_env.render() 내부에서 이미 호출됨

    def close(self):
        """환경 및 로거 종료"""
        print("테스트 환경을 종료합니다.")
        self.rl_env.close()
        if self.writer:
            self.writer.close()

    def run(self):
        """
        메인 비교 테스트 실행 루프
        """
        try:
            num_episodes = self.config['simulation']['rl']['eval_episode']
            log_interval = 50
            print(f"총 {num_episodes} 에피소드 비교 테스트 시작... (스칼라 로그 간격: {log_interval} 스텝)")

            # 전체 실행 시간 측정을 위한 리스트
            all_rl_times_ms = []
            all_classic_times_ms = []
            for ep in range(1, num_episodes + 1):
                # --- 1. 환경 리셋 ---
                # observations: (num_vehicles, frame_stack, obs_dim)
                observations, active_agents = self.reset()
                done = False
                episode_reward = 0.0
                episode_steps = 0

                # 에피소드별 시간 측정을 위한 리스트
                ep_rl_times_ms = []
                ep_classic_times_ms = []

                # 에피소드별 행동 저장을 위한 리스트
                rl_throttles, rl_steers = [], []
                cl_throttles, cl_steers = [], []

                while not done:
                    # --- 2. Pygame 이벤트 처리 ---
                    self.handle_keyboard_input()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("사용자에 의해 종료됨")
                            return
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                print("사용자에 의해 종료됨")
                                return

                    # --- 3. 행동(Action) 결정 ---
                    current_actions = np.zeros((self.rl_env.num_vehicles, 2)) # (2, 2)
                    vehicles = self.vehicle_manager.get_all_vehicles()

                    rl_time_ms = 0.0 # 스텝별 시간 (기본값 0)
                    cl_time_ms = 0.0 # 스텝별 시간 (기본값 0)

                    # 차량 0 (RL) 행동 결정
                    if active_agents[0]:
                        obs_v0 = observations[0] # (frame_stack, obs_dim)
                        obs_v0_batch = np.expand_dims(obs_v0, axis=0) # (1, frame_stack, obs_dim)
                        start_rl = time.perf_counter()
                        action_v0, _ = self.rl_model.predict(obs_v0_batch, deterministic=True)
                        rl_time_ms = (time.perf_counter() - start_rl) * 1000.0
                        current_actions[0] = action_v0[0] # (2,)

                        ep_rl_times_ms.append(rl_time_ms)
                        rl_throttles.append(current_actions[0, 0])
                        rl_steers.append(current_actions[0, 1])

                    # 차량 1 (Classic) 행동 결정
                    if active_agents[1]:
                        vehicle_state_v1 = vehicles[1].get_state()
                        # 장애물 목록 생성 (차량 0 포함)
                        objects = self._get_classic_obstacles(active_vehicle_id=1)
                        # 도로 경계 추가
                        road_boundary = vehicle_state_v1.closest_segment.get_boundary()
                        if len(road_boundary) > 0:
                            if objects.size == 0:
                                objects = road_boundary
                            else:
                                objects = np.vstack((objects, road_boundary))

                        start_cl = time.perf_counter()
                        action_v1 = self.classic_controller.plan_and_control(vehicle_state_v1, objects)
                        cl_time_ms = (time.perf_counter() - start_cl) * 1000.0
                        current_actions[1] = action_v1

                        ep_classic_times_ms.append(cl_time_ms)
                        cl_throttles.append(current_actions[1, 0])
                        cl_steers.append(current_actions[1, 1])


                    # --- 4. 시뮬레이션 스텝 실행 ---
                    # next_observations: (2, frame_stack, obs_dim)
                    # info['rewards']: [rew_v0, rew_v1]
                    next_observations, avg_reward, done, truncated, info = self.rl_env.step(current_actions)

                    episode_reward += avg_reward
                    episode_steps += 1

                    # --- 5. TensorBoard 로깅 ---
                    if self.writer and self.global_step % log_interval == 0:
                        # 행동(Action) 출력 로깅
                        self.writer.add_scalar('Action/RL/Throttle', current_actions[0, 0], self.global_step)
                        self.writer.add_scalar('Action/RL/Steering', current_actions[0, 1], self.global_step)
                        self.writer.add_scalar('Action/Classic/Throttle', current_actions[1, 0], self.global_step)
                        self.writer.add_scalar('Action/Classic/Steering', current_actions[1, 1], self.global_step)

                        # 개별 보상 로깅
                        self.writer.add_scalar('Reward/RL_Individual', info['rewards'][0], self.global_step)
                        self.writer.add_scalar('Reward/Classic_Individual', info['rewards'][1], self.global_step)

                        # 스텝별 수행 속도 로깅 (0 이상일 때만)
                        if rl_time_ms > 0:
                            self.writer.add_scalar('Perf_Step/RL_Predict_ms', rl_time_ms, self.global_step)
                        if cl_time_ms > 0:
                            self.writer.add_scalar('Perf_Step/Classic_Plan_ms', cl_time_ms, self.global_step)

                        # 차량 상태 로깅 (활성 상태일 때만)
                        if active_agents[0]:
                            rl_state = vehicles[0].get_state()
                            self.writer.add_scalar('Vehicle/RL/Speed_kmh', rl_state.vel_long * 3.6, self.global_step)
                            self.writer.add_scalar('Vehicle/RL/Progress', rl_state.get_progress(), self.global_step)
                            self.writer.add_scalar('Vehicle/RL/LaneOffset_m', rl_state.frenet_d, self.global_step)

                        if active_agents[1]:
                            classic_state = vehicles[1].get_state()
                            self.writer.add_scalar('Vehicle/Classic/Speed_kmh', classic_state.vel_long * 3.6, self.global_step)
                            self.writer.add_scalar('Vehicle/Classic/Progress', classic_state.get_progress(), self.global_step)
                            self.writer.add_scalar('Vehicle/Classic/LaneOffset_m', classic_state.frenet_d, self.global_step)

                    # 에피소드 종료 시 평균 보상 기록
                    if self.writer and done:
                        self.writer.add_scalar('Episode/AverageReward', avg_reward, ep)
                        self.writer.add_scalar('Episode/TotalReward', episode_reward, ep)
                        self.writer.add_scalar('Episode/Length', episode_steps, ep)

                        # 에피소드별 수행 속도 통계 로깅
                        if ep_rl_times_ms:
                            rl_times_np = np.array(ep_rl_times_ms)
                            self.writer.add_scalar('Perf_Episode/RL/Mean_ms', np.mean(rl_times_np), ep)
                            self.writer.add_scalar('Perf_Episode/RL/Max_ms', np.max(rl_times_np), ep)
                            self.writer.add_scalar('Perf_Episode/RL/Min_ms', np.min(rl_times_np), ep)
                            self.writer.add_scalar('Perf_Episode/RL/Std_ms', np.std(rl_times_np), ep)
                            all_rl_times_ms.extend(ep_rl_times_ms) # 전체 통계 리스트에 추가

                        if ep_classic_times_ms:
                            cl_times_np = np.array(ep_classic_times_ms)
                            self.writer.add_scalar('Perf_Episode/Classic/Mean_ms', np.mean(cl_times_np), ep)
                            self.writer.add_scalar('Perf_Episode/Classic/Max_ms', np.max(cl_times_np), ep)
                            self.writer.add_scalar('Perf_Episode/Classic/Min_ms', np.min(cl_times_np), ep)
                            self.writer.add_scalar('Perf_Episode/Classic/Std_ms', np.std(cl_times_np), ep)
                            all_classic_times_ms.extend(ep_classic_times_ms) # 전체 통계 리스트에 추가

                        if rl_throttles:
                            self.writer.add_histogram('Action/RL/Throttle/Dist', np.array(rl_throttles), ep)
                            self.writer.add_histogram('Action/RL/Steering/Dist', np.array(rl_steers), ep)
                        if cl_throttles:
                            self.writer.add_histogram('Action/Classic/Throttle/Dist', np.array(cl_throttles), ep)
                            self.writer.add_histogram('Action/Classic/Steering/Dist', np.array(cl_steers), ep)

                    self.global_step += 1

                    # --- 6. 상태 업데이트 및 렌더링 ---
                    observations = next_observations
                    active_agents = info['active_agents']
                    self.render()

                # --- 에피소드 종료 ---
                print(f"[Test] Episode {ep}/{num_episodes} 완료 — Steps: {episode_steps}, Avg Reward: {episode_reward / max(1, episode_steps):.2f}")
                print(f"  - RL (V0) 충돌: {info['collisions'].get(0, False)}, 성공: {info['reached_targets'].get(0, False)}")
                print(f"  - Classic (V1) 충돌: {info['collisions'].get(1, False)}, 성공: {info['reached_targets'].get(1, False)}")

        except KeyboardInterrupt:
            print("\n사용자에 의해 테스트가 중단되었습니다.")
        except Exception as e:
            print(f"\n오류 발생으로 테스트 중단: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # --- 7. 전체 통계 출력 및 로깅 ---
            print("\n" + "="*30)
            print("--- Overall Performance Statistics (ms) ---")
            summary_text = "Overall Performance (ms):\n\n| Controller | Mean (ms) | Max (ms) | Min (ms) | Std (ms) |\n|:---|:---|:---|:---|:---|\n"

            if all_rl_times_ms:
                rl_all_np = np.array(all_rl_times_ms)
                mean, pmax, pmin, std = np.mean(rl_all_np), np.max(rl_all_np), np.min(rl_all_np), np.std(rl_all_np)
                rl_stats = f"| RL Predict | {mean:.4f} | {pmax:.4f} | {pmin:.4f} | {std:.4f} |"
                print(rl_stats.replace("|", " ").strip()) # 콘솔 출력용
                summary_text += rl_stats + "\n"
            else:
                print("[RL Predict] No data recorded.")
                summary_text += "| RL Predict | N/A | N/A | N/A | N/A |\n"

            if all_classic_times_ms:
                cl_all_np = np.array(all_classic_times_ms)
                mean, pmax, pmin, std = np.mean(cl_all_np), np.max(cl_all_np), np.min(cl_all_np), np.std(cl_all_np)
                cl_stats = f"| Classic Plan | {mean:.4f} | {pmax:.4f} | {pmin:.4f} | {std:.4f} |"
                print(cl_stats.replace("|", " ").strip()) # 콘솔 출력용
                summary_text += cl_stats + "\n"
            else:
                print("[Classic Plan] No data recorded.")
                summary_text += "| Classic Plan | N/A | N/A | N/A | N/A |\n"

            print("="*30 + "\n")

            if self.writer:
                self.writer.add_text('Overall_Performance/Summary', summary_text, self.global_step)

            # --- 8. 종료 ---
            self.close()
