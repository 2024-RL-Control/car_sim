# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import pygame
import numpy as np
from typing import Dict, Any, Optional, Union
from collections import deque
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.results_plotter import load_results, ts2xy


def create_training_callbacks(config: Dict[str, Any], log_dir: str, models_dir: str) -> list:
    """
    설정에 따라 적절한 콜백 조합을 생성하는 팩토리 함수

    Args:
        config: 환경 및 훈련 설정
        log_dir: 로그 디렉토리
        models_dir: 모델 저장 디렉토리

    Returns:
        콜백 리스트
    """
    callbacks = []

    # 핵심 메트릭 콜백
    if config.get('num_vehicles', 1) > 1:
        # 다중 차량 환경
        vehicle_callback = VehicleMetricsCallback(
            num_vehicles=config['num_vehicles'],
            log_freq=config.get('vehicle_log_freq', 1000),
            verbose=config.get('verbose', 0)
        )
        callbacks.append(vehicle_callback)
    else:
        # 단일 차량 환경
        training_callback = TrainingMetricsCallback(
            verbose=config.get('verbose', 0)
        )
        callbacks.append(training_callback)

    # 성능 모니터링
    performance_callback = PerformanceCallback(
        log_freq=config.get('performance_log_freq', 1000),
        gpu_memory_limit=config.get('gpu_memory_limit', 3),
        verbose=config.get('verbose', 0)
    )
    callbacks.append(performance_callback)

    # 체크포인트 관리
    checkpoint_callback = EnhancedCheckpointCallback(
        save_freq=config.get('checkpoint_freq', 10000),
        save_path=models_dir,
        name_prefix=config.get('algorithm', 'model'),
        max_checkpoints=config.get('max_checkpoints', 5),
        save_best=config.get('save_best_model', True),
        verbose=config.get('verbose', 0)
    )
    callbacks.append(checkpoint_callback)

    # 하이퍼파라미터 로깅
    hparam_callback = HyperparameterCallback(
        env_config=config,
        verbose=config.get('verbose', 0)
    )
    callbacks.append(hparam_callback)

    # 통합 로깅
    logging_callback = ModularLoggingCallback(
        log_dir=log_dir,
        save_freq=config.get('logging_freq', 1000),
        verbose=config.get('verbose', 0)
    )
    callbacks.append(logging_callback)

    return callbacks

def get_callback_summary(callbacks: list) -> Dict[str, Any]:
    """
    콜백 리스트의 요약 정보를 반환
    """
    summary = {
        'total_callbacks': len(callbacks),
        'callback_types': [type(cb).__name__ for cb in callbacks],
        'core_callbacks': 0,
        'specialized_callbacks': 0,
        'utility_callbacks': 0
    }

    core_types = {TrainingMetricsCallback, PerformanceCallback, VehicleMetricsCallback}
    specialized_types = {EnhancedCheckpointCallback, HyperparameterCallback}
    utility_types = {ModularLoggingCallback}

    for callback in callbacks:
        cb_type = type(callback)
        if cb_type in core_types:
            summary['core_callbacks'] += 1
        elif cb_type in specialized_types:
            summary['specialized_callbacks'] += 1
        elif cb_type in utility_types:
            summary['utility_callbacks'] += 1

    return summary

class TrainingMetricsCallback(BaseCallback):
    """
    핵심 훈련 메트릭을 수집하고 관리하는 기본 콜백
    - 에피소드 보상, 길이, 성공률 등 기본 메트릭 추적
    - 다른 콜백들이 사용할 수 있는 공통 데이터 제공
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.start_time = time.time()

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if self.verbose >= 1:
            print("Training metrics collection started")

    def _on_step(self) -> bool:
        # 에피소드 진행 중 메트릭 업데이트는 하위 클래스에서 구현
        return True

    def get_recent_mean_reward(self, window: int = 100) -> float:
        """최근 N개 에피소드의 평균 보상 반환"""
        if len(self.episode_rewards) == 0:
            return 0.0
        recent_rewards = list(self.episode_rewards)[-window:]
        return np.mean(recent_rewards)

    def get_recent_mean_length(self, window: int = 100) -> float:
        """최근 N개 에피소드의 평균 길이 반환"""
        if len(self.episode_lengths) == 0:
            return 0.0
        recent_lengths = list(self.episode_lengths)[-window:]
        return np.mean(recent_lengths)

class PerformanceCallback(BaseCallback):
    """
    시스템 성능 및 리소스 사용량을 모니터링하는 콜백
    - GPU/CPU 메모리 사용량, 학습 속도, FPS 등
    """
    def __init__(self, log_freq: int = 1000, gpu_memory_limit: int = 3, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.gpu_memory_limit = gpu_memory_limit
        self.gpu_memory_usage = deque(maxlen=1000)
        self.training_start_time = None
        self.step_times = deque(maxlen=100)
        self.last_step_time = None

    def _on_training_start(self) -> None:
        self.training_start_time = time.time()
        self.last_step_time = time.time()

    def _on_step(self) -> bool:
        current_time = time.time()

        # 스텝 시간 기록
        if self.last_step_time is not None:
            step_time = current_time - self.last_step_time
            self.step_times.append(step_time)
        self.last_step_time = current_time

        # GPU 메모리 사용량 추적
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            self.gpu_memory_usage.append(gpu_memory_gb)

            # 메모리 정리 (지정된 GPU 메모리 이상 사용 시)
            if gpu_memory_gb > self.gpu_memory_limit:
                torch.cuda.empty_cache()
                if self.verbose >= 1:
                    print(f"GPU memory cleared at {gpu_memory_gb:.2f}GB")

        # 주기적 로깅
        if self.n_calls % self.log_freq == 0:
            self._log_performance_metrics()

        return True

    def _log_performance_metrics(self):
        """성능 메트릭을 로거에 기록"""
        if self.training_start_time:
            elapsed_time = time.time() - self.training_start_time
            steps_per_second = self.n_calls / elapsed_time if elapsed_time > 0 else 0

            self.logger.record("performance/steps_per_second", steps_per_second)
            self.logger.record("performance/elapsed_hours", elapsed_time / 3600)

            if len(self.step_times) > 0:
                avg_step_time = np.mean(self.step_times)
                self.logger.record("performance/avg_step_time_ms", avg_step_time * 1000)

            if torch.cuda.is_available() and len(self.gpu_memory_usage) > 0:
                current_gpu_memory = self.gpu_memory_usage[-1]
                max_gpu_memory = max(self.gpu_memory_usage)
                self.logger.record("performance/gpu_memory_current_gb", current_gpu_memory)
                self.logger.record("performance/gpu_memory_max_gb", max_gpu_memory)

class EnhancedCheckpointCallback(CheckpointCallback):
    """
    향상된 체크포인트 관리 콜백
    - 자동 정리, 최고 성능 모델 저장, 백업 관리
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = 'model',
                 max_checkpoints: int = 5, save_best: bool = True, verbose: int = 0):
        super().__init__(save_freq=save_freq, save_path=save_path,
                        name_prefix=name_prefix, verbose=verbose)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_mean_reward = -np.inf
        self.best_model_path = None

    def _on_step(self):
        result = super()._on_step()

        # 체크포인트 저장 후 정리
        if self.num_timesteps % self.save_freq == 0:
            self._cleanup_old_checkpoints()

            if self.save_best:
                self._save_best_model()

        return result

    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 정리"""
        try:
            files = sorted([
                f for f in os.listdir(self.save_path)
                if f.startswith(self.name_prefix) and f.endswith('.zip')
                and 'best' not in f  # best 모델은 제외
            ])

            for old_file in files[:-self.max_checkpoints]:
                old_path = os.path.join(self.save_path, old_file)
                os.remove(old_path)
                if self.verbose >= 1:
                    print(f"Removed old checkpoint: {old_file}")
        except Exception as e:
            if self.verbose >= 1:
                print(f"Checkpoint cleanup failed: {e}")

    def _save_best_model(self):
        """최고 성능 모델 저장 (TrainingMetricsCallback이 있는 경우)"""
        # 다른 콜백에서 메트릭 가져오기 시도
        try:
            # CallbackList에서 TrainingMetricsCallback 찾기
            if hasattr(self.parent, 'callbacks'):
                for callback in self.parent.callbacks:
                    if isinstance(callback, TrainingMetricsCallback):
                        current_reward = callback.get_recent_mean_reward(window=10)
                        if current_reward > self.best_mean_reward:
                            self.best_mean_reward = current_reward
                            self.best_model_path = os.path.join(
                                self.save_path, f"{self.name_prefix}_best.zip"
                            )
                            self.model.save(self.best_model_path)
                            if self.verbose >= 1:
                                print(f"New best model saved: {current_reward:.2f}")
                        break
        except Exception as e:
            if self.verbose >= 1:
                print(f"Best model saving failed: {e}")

class HyperparameterCallback(BaseCallback):
    """
    하이퍼파라미터와 모델 설정을 TensorBoard에 로깅하는 콜백
    - 알고리즘별 하이퍼파라미터 자동 감지 및 로깅
    - 환경별 설정 정보 포함
    """
    def __init__(self, env_config: Optional[Dict[str, Any]] = None, verbose: int = 0):
        super().__init__(verbose)
        self.env_config = env_config or {}

    def _on_training_start(self) -> None:
        hparam_dict = self._collect_hyperparameters()
        metric_dict = self._define_metrics()

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

        if self.verbose >= 1:
            print(f"Logged {len(hparam_dict)} hyperparameters and {len(metric_dict)} metrics")

    def _collect_hyperparameters(self) -> Dict[str, Any]:
        """모델과 환경의 하이퍼파라미터 수집"""
        hparams = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": float(self.model.learning_rate),
            "gamma": float(self.model.gamma),
        }

        # 알고리즘별 특화 파라미터
        algo_specific = {
            'batch_size': getattr(self.model, 'batch_size', None),
            'buffer_size': getattr(self.model, 'buffer_size', None),
            'tau': getattr(self.model, 'tau', None),
            'ent_coef': getattr(self.model, 'ent_coef', None),
            'n_steps': getattr(self.model, 'n_steps', None),
            'n_epochs': getattr(self.model, 'n_epochs', None),
            'clip_range': getattr(self.model, 'clip_range', None),
        }

        # None이 아닌 값만 추가
        for key, value in algo_specific.items():
            if value is not None:
                hparams[key] = float(value) if isinstance(value, (int, float)) else value

        # 환경 설정 추가
        if self.env_config:
            env_params = {
                "num_vehicles": self.env_config.get('num_vehicles', 1),
                "max_episode_steps": self.env_config.get('max_episode_steps', 1000),
                "num_static_obstacles": self.env_config.get('num_static_obstacles', 0),
                "num_dynamic_obstacles": self.env_config.get('num_dynamic_obstacles', 0),
            }
            hparams.update(env_params)

        return hparams

    def _define_metrics(self) -> Dict[str, float]:
        """추적할 메트릭 정의"""
        base_metrics = {
            "rollout/ep_rew_mean": 0.0,
            "rollout/ep_len_mean": 0.0,
            "performance/steps_per_second": 0.0,
            "performance/gpu_memory_current_gb": 0.0,
        }

        # 알고리즘별 메트릭
        if "SAC" in self.model.__class__.__name__:
            base_metrics.update({
                "train/actor_loss": 0.0,
                "train/critic_loss": 0.0,
                "train/ent_coef": 0.0,
            })
        elif "PPO" in self.model.__class__.__name__:
            base_metrics.update({
                "train/policy_loss": 0.0,
                "train/value_loss": 0.0,
                "train/explained_variance": 0.0,
            })

        return base_metrics

    def _on_step(self) -> bool:
        return True

class VehicleMetricsCallback(TrainingMetricsCallback):
    """
    차량 시뮬레이션 환경 전용 메트릭 콜백
    - 차량별 성능, 충돌률, 목표 도달률 등 추적
    - 멀티 에이전트 환경 지원
    """
    def __init__(self, num_vehicles: int = 1, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.num_vehicles = num_vehicles
        self.log_freq = log_freq

        # 차량별 메트릭
        self.vehicle_rewards = {i: deque(maxlen=100) for i in range(num_vehicles)}
        self.collision_count = 0
        self.goal_reached_count = 0
        self.outside_road_count = 0
        self.active_vehicles_history = deque(maxlen=100)

    def _on_step(self) -> bool:
        super()._on_step()

        # 차량 시뮬레이션 특화 메트릭 수집
        self._collect_vehicle_metrics()

        # 주기적 로깅
        if self.n_calls % self.log_freq == 0:
            self._log_vehicle_metrics()

        return True

    def _collect_vehicle_metrics(self):
        """차량 관련 메트릭 수집"""
        # 로컬 변수에서 정보 추출 시도
        info = self.locals.get('info', {})

        if isinstance(info, dict):
            # 충돌, 목표 도달, 도로 이탈 카운트
            collisions = info.get('collisions', {})
            reached_targets = info.get('reached_targets', {})
            outside_roads = info.get('outside_roads', {})

            if collisions:
                self.collision_count += sum(collisions.values())
            if reached_targets:
                self.goal_reached_count += sum(reached_targets.values())
            if outside_roads:
                self.outside_road_count += sum(outside_roads.values())

            # 개별 차량 보상 기록
            rewards = info.get('rewards', [])
            if isinstance(rewards, (list, np.ndarray)) and len(rewards) >= self.num_vehicles:
                for i in range(min(self.num_vehicles, len(rewards))):
                    self.vehicle_rewards[i].append(float(rewards[i]))

    def _log_vehicle_metrics(self):
        """차량 관련 메트릭을 로거에 기록"""
        # 전체 통계
        total_episodes = max(1, self.episode_count)

        self.logger.record("vehicles/collision_rate", self.collision_count / total_episodes)
        self.logger.record("vehicles/goal_reached_rate", self.goal_reached_count / total_episodes)
        self.logger.record("vehicles/outside_road_rate", self.outside_road_count / total_episodes)

        # 차량별 평균 보상 (처음 3대만)
        for i in range(min(3, self.num_vehicles)):
            if len(self.vehicle_rewards[i]) > 0:
                avg_reward = np.mean(self.vehicle_rewards[i])
                self.logger.record(f"vehicles/vehicle_{i}_avg_reward", avg_reward)

        # 활성 차량 수 (가능한 경우)
        try:
            # 모델에서 활성 에이전트 정보 가져오기
            if hasattr(self.model, 'rl_env') and hasattr(self.model.rl_env, 'active_agents'):
                active_count = sum(self.model.rl_env.active_agents)
                self.logger.record("vehicles/active_count", active_count)
                self.logger.record("vehicles/active_ratio", active_count / self.num_vehicles)
        except Exception:
            pass

    def update_episode_end(self, episode_reward: float, episode_length: int,
                          episode_info: Dict[str, Any]):
        """에피소드 종료 시 메트릭 업데이트"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_count += 1
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

class ModularLoggingCallback(BaseCallback):
    """
    모듈형 로깅 콜백 - 다른 콜백들의 데이터를 수집하여 통합 로깅
    """
    def __init__(self, log_dir: str, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.monitor_file = os.path.join(log_dir, "monitor.csv")
        self.start_time = time.time()
        self.last_episode_count = 0

        # CSV 파일 초기화
        os.makedirs(log_dir, exist_ok=True)
        self._init_csv_file()

    def _init_csv_file(self):
        """CSV 파일 헤더 초기화"""
        try:
            with open(self.monitor_file, 'w') as f:
                f.write(f'# Training monitor started at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write('episode,timesteps,reward,length,elapsed_time,active_vehicles,collision_rate\n')
        except Exception as e:
            if self.verbose >= 1:
                print(f"Failed to initialize CSV file: {e}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._collect_and_log_data()
        return True

    def _collect_and_log_data(self):
        """다른 콜백들로부터 데이터를 수집하여 기록"""
        try:
            # CallbackList에서 다른 콜백들 찾기
            training_callback = None
            vehicle_callback = None

            if hasattr(self.parent, 'callbacks'):
                for callback in self.parent.callbacks:
                    if isinstance(callback, TrainingMetricsCallback):
                        training_callback = callback
                    elif isinstance(callback, VehicleMetricsCallback):
                        vehicle_callback = callback

            # 데이터 수집 및 기록
            if training_callback and training_callback.episode_count > self.last_episode_count:
                self._log_episode_data(training_callback, vehicle_callback)
                self.last_episode_count = training_callback.episode_count

        except Exception as e:
            if self.verbose >= 1:
                print(f"Data collection failed: {e}")

    def _log_episode_data(self, training_callback, vehicle_callback):
        """에피소드 데이터를 CSV에 기록"""
        try:
            elapsed_time = time.time() - self.start_time
            episode = training_callback.episode_count
            timesteps = self.num_timesteps

            # 기본 메트릭
            reward = training_callback.get_recent_mean_reward(window=1)
            length = training_callback.get_recent_mean_length(window=1)

            # 차량 메트릭 (있는 경우)
            active_vehicles = "N/A"
            collision_rate = "N/A"
            if vehicle_callback:
                try:
                    if hasattr(self.model, 'rl_env') and hasattr(self.model.rl_env, 'active_agents'):
                        active_vehicles = sum(self.model.rl_env.active_agents)
                    collision_rate = vehicle_callback.collision_count / max(1, episode)
                except Exception:
                    pass

            # CSV에 기록
            with open(self.monitor_file, 'a') as f:
                f.write(f'{episode},{timesteps},{reward:.4f},{length:.1f},{elapsed_time:.2f},{active_vehicles},{collision_rate}\n')

        except Exception as e:
            if self.verbose >= 1:
                print(f"CSV logging failed: {e}")

class SACVehicleAlgorithm(SAC):
    """
    다중 차량을 위한 커스텀 강화학습 알고리즘 클래스
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 추가 속성 설정
        self.rl_env = None
        self.num_vehicles = None
        self.prev_observations = None
        self.prev_active_mask = None
        self.episode_steps = 0
        self.total_reward = 0

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
        success = False

        while not success:
            try:
                prev_observations, _ = self.rl_env.reset()
                success = True
            except Exception as e:
                print(f"Reset failed: {e}")
                continue

        self.prev_observations = prev_observations
        self.prev_active_mask = np.array(self.rl_env.active_agents)
        self.episode_steps = 0
        self.total_reward = 0.0

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

        # 2) 행동 결정
        active_mask = self.prev_active_mask
        actions = np.zeros((self.num_vehicles, 3))
        if self._n_updates < self.learning_starts:
            # 랜덤 행동 (벡터화)
            if active_mask.any():
                random_actions = np.stack([self.rl_env.env.action_space.sample() for _ in range(active_mask.sum())])
                random_actions[:, 1] = random_actions[:, 1] / 5
                # random_actions[:, 2] = random_actions[:, 2] / 3
                actions[active_mask] = random_actions
        else:
            # 활성화된 에이전트만 행동 예측 (벡터화)
            if active_mask.any():
                # 활성화된 에이전트의 관측값만 선택
                active_obs = self.prev_observations[active_mask]
                # 배치로 한번에 예측
                active_obs_tensor = torch.as_tensor(active_obs, device=self.device)
                with torch.no_grad():
                    actions_tensor, _ = self.policy.actor.action_log_prob(active_obs_tensor)
                # NumPy 배열로 변환
                actions[active_mask] = actions_tensor.cpu().numpy()

        # 3) 환경 스텝 & 보상 누적
        next_observations, reward, done, _, info = self.rl_env.step(actions)
        self.episode_steps += 1
        self.num_timesteps += 1
        self.total_reward += reward

        # 4) 렌더링
        self.rl_env.render()

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

        # 6) 내부 상태 갱신
        self.prev_observations = next_observations  # shape=(num_vehicles, obs_dim)
        self.prev_active_mask = np.array(self.rl_env.active_agents, dtype=bool)

        # 7) 콜백 업데이트 및 종료 체크
        callback.update_locals(locals())
        if not callback.on_step():
            return False

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

        return True

    def learn(
        self,
        total_timesteps: int,
        callback = None,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        **kwargs
    ):
        """
        다중 차량을 위한 학습 메소드 오버라이드
        """
        total_timesteps, callback = self._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            **kwargs,
        )

        callback.on_training_start(locals(), globals())

        # 환경 초기화
        self._reset()

        while self.num_timesteps < total_timesteps:
            continue_training = self._custom_rollout_step(callback)
            if not continue_training:
                break

            if self.num_timesteps > self.learning_starts and self.num_timesteps % self.train_freq.frequency == 0:
                self.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)

        callback.on_training_end()
        return self

class PPOVehicleAlgorithm(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 추가 속성 설정
        self.rl_env = None
        self.num_vehicles = None
        self.prev_observations = None
        self.prev_active_mask = None
        self.episode_steps = 0
        self.total_reward = 0
        self._is_new_episode = True

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
        success = False

        while not success:
            try:
                prev_observations, _ = self.rl_env.reset()
                success = True
            except Exception as e:
                print(f"Reset failed: {e}")
                continue

        self.prev_observations = prev_observations
        self.prev_active_mask = np.array(self.rl_env.active_agents)
        self.episode_steps = 0
        self.total_reward = 0.0
        self._is_new_episode = True

    def _custom_rollout_step(self, callback: BaseCallback, n_steps: int) -> bool:
        """
        실제 rollouts를 수집하는 함수.
        - n_steps가 되거나 에피소드 종료(done)가 발생하면 True를 리턴하여 학습으로 넘어가게 함.
        - callback이 False를 반환하면 False 리턴하여 학습 루프를 빠져나감.

        매 스텝마다:
        1) 키보드 이벤트 처리리
        2) 활성 에이전트(active_mask)만 policy.forward()를 통해 행동(action), value, log_prob 계산
        3) self.rl_env.step(actions)를 호출하여 (next_obs, reward, done, info) 획득 및 self.num_timesteps, self.episode_steps, self.total_reward 갱신
        4) 렌더링
        5) 활성 에이전트 하나하나에 대해 rollout_buffer.add(...) 호출
        6) self.prev_observations, self.prev_active_mask 갱신
        7) callback.on_step() 확인
        8) done=True이면 에피소드 종료 처리 후 즉시 학습으로 넘어감
        """
        step_count = 0
        # 수집한 transition 개수(Active agent 기준) 혹은, 활성 스텝을 셀 수도 있음
        # 여기서는 Active agent 하나당 한 개 transition으로 본다.
        while step_count < n_steps:
            # 1) 키보드 이벤트 처리
            self.rl_env.handle_keyboard_input()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False

            # 2) 행동(action), 값(value), 로그 확률(log_prob) 계산
            active_mask = self.prev_active_mask
            actions = np.zeros((self.num_vehicles, self.action_space.shape[-1]), dtype=np.float32)
            values = torch.zeros((self.num_vehicles,), dtype=torch.float32, device=self.device)
            log_probs = torch.zeros((self.num_vehicles,), dtype=torch.float32, device=self.device)

            # 활성 에이전트에 한해서만 정책 네트워크를 통과
            if active_mask.any():
                active_obs = self.prev_observations[active_mask]
                active_obs_tensor = torch.as_tensor(active_obs, device=self.device)
                with torch.no_grad():
                    actions_tensor, values_tensor, log_prob_tensor = self.policy.forward(active_obs_tensor)

                # NumPy 및 형태 변환환
                actions_np = actions_tensor.cpu().numpy()
                values_tensor = values_tensor.view(-1)
                log_prob_tensor = log_prob_tensor.view(-1)

                # 활성 에이전트 위치에 해당 값들을 대입
                actions[active_mask] = actions_np
                values[active_mask] = values_tensor
                log_probs[active_mask] = log_prob_tensor

            # 3) 환경 스텝: BasicRLDrivingEnv.step(actions)
            next_observations, reward, done, _, info = self.rl_env.step(actions)
            self.num_timesteps += 1
            self.episode_steps += 1
            self.total_reward += reward

            # 4) 렌더링
            self.rl_env.render()

            # 5) 활성 에이전트마다 RolloutBuffer에 transition 추가
            active_indices = np.where(active_mask)[0]
            for idx in active_indices:
                # 개별 차량(obs, action, reward, done, value, log_prob)을 rollout buffer에 저장
                self.rollout_buffer.add(
                    obs=self.prev_observations[idx].copy(),
                    action=actions[idx].copy(),
                    reward=info['rewards'][idx],
                    episode_start = self._is_new_episode,
                    value=values[idx],
                    log_prob=log_probs[idx],
                )
                step_count += 1
                if step_count >= n_steps:
                    return True

            # 6) 내부 상태 갱신
            if self._is_new_episode:
                self._is_new_episode = False
            self.prev_observations = next_observations  # shape=(num_vehicles, obs_dim)
            self.prev_active_mask = np.array(self.rl_env.active_agents, dtype=bool)

            # 7) 콜백 업데이트 및 종료 체크
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # 8) 에피소드 종료시점 처리
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

        # n_steps만큼 Transition을 모았으면, 학습 단계로 넘어간다.
        return True

    def learn(
            self,
            total_timesteps: int,
            callback: BaseCallback = None,
            tb_log_name: str = "PPO",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
            **kwargs
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            **kwargs,
        )

        callback.on_training_start(locals(), globals())

        self._reset()
        n_steps = self.n_steps

        while self.num_timesteps < total_timesteps:
            continue_training = self._custom_rollout_step(callback, n_steps)
            if not continue_training:
                break

            self.train()
            self.rollout_buffer.reset()

        callback.on_training_end()
        return self

class ResidualLayerNormMLP(nn.Module):
    """
    MLP with LayerNorm, Residual Connections, and Orthogonal Init + GELU Activation
    Improved version with proper residual connections and SB3-recommended initialization
    """
    def __init__(self, input_dim: int, net_arch: list[int]):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.projections = nn.ModuleList()

        in_dim = input_dim
        for hidden_dim in net_arch:
            block = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            self.blocks.append(block)

            # Add projection layer for residual connection when dimensions don't match
            if in_dim != hidden_dim:
                projection = nn.Linear(in_dim, hidden_dim)
                self.projections.append(projection)
            else:
                self.projections.append(None)

            in_dim = hidden_dim

        # Apply weight initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (block, projection) in enumerate(zip(self.blocks, self.projections)):
            residual = x
            x = block(x)

            # Apply projection if dimensions don't match, otherwise direct residual
            if projection is not None:
                residual = projection(residual)

            x = x + residual
        return x

    @staticmethod
    def _init_weights(m):
        # Orthogonal initialization for Linear layers (SB3 recommended)
        if isinstance(m, nn.Linear):
            # GELU는 gain이 정의되어 있지 않아서 기본값 1.0 사용 (또는 tanh 근사)
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, net_arch: list[int]):
        # last layer size → net_arch[-1]
        super().__init__(obs_space, features_dim=net_arch[-1])
        self.flatten = nn.Flatten()
        self.mlp     = ResidualLayerNormMLP(obs_space.shape[0], net_arch)

    def forward(self, obs):
        x = self.flatten(obs)
        return self.mlp(x)
