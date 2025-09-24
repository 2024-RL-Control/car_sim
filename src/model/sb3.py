# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import pygame
import numpy as np
import json
import gzip
from typing import Dict, Any, Optional, Union, List
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, EvalCallback
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import HParam, TensorBoardOutputFormat
from stable_baselines3.common.results_plotter import load_results, ts2xy


# =============================================================================
# 중앙화된 메트릭 스토어
# =============================================================================

@dataclass
class EpisodeData:
    """에피소드 데이터 구조"""
    episode_id: int
    timesteps: int
    reward: float
    length: int
    collisions: Dict[int, bool]
    goals_reached: Dict[int, bool]
    outside_roads: Dict[int, bool]
    individual_rewards: List[float]
    elapsed_time: float


class MetricsStore:
    """
    중앙화된 메트릭 저장소
    모든 콜백이 공유하는 단일 데이터 소스
    """

    def __init__(self, max_episodes: int = 1000, max_steps: int = 10000):
        # 기본 메트릭
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.total_timesteps = 0

        # 에피소드 기록 (순환 버퍼)
        self.episodes = deque(maxlen=max_episodes)

        # 성능 메트릭 (순환 버퍼)
        self.step_times = deque(maxlen=max_steps)
        self.gpu_memory_usage = deque(maxlen=max_steps)

        # 차량별 메트릭
        self.vehicle_metrics = {}

        # 시스템 정보
        self.training_start_time = None
        self.last_step_time = None

        # 베스트 성능 추적
        self.best_reward = -np.inf
        self.best_episode_id = -1

    def start_training(self):
        """훈련 시작 시 호출"""
        self.training_start_time = time.time()
        self.last_step_time = time.time()

    def record_step(self, timestep: int, gpu_memory: Optional[float] = None):
        """스텝 기록"""
        current_time = time.time()

        if self.last_step_time is not None:
            step_time = current_time - self.last_step_time
            self.step_times.append(step_time)

        if gpu_memory is not None:
            self.gpu_memory_usage.append(gpu_memory)

        self.last_step_time = current_time
        self.total_timesteps = timestep

    def record_episode(self, episode_data: EpisodeData):
        """에피소드 기록"""
        self.episodes.append(episode_data)
        self.episode_count += 1

        # 베스트 성능 업데이트
        if episode_data.reward > self.best_reward:
            self.best_reward = episode_data.reward
            self.best_episode_id = episode_data.episode_id

    def get_recent_mean_reward(self, window: int = 100) -> float:
        """최근 N개 에피소드 평균 보상"""
        if not self.episodes:
            return 0.0
        recent_episodes = list(self.episodes)[-window:]
        return np.mean([ep.reward for ep in recent_episodes])

    def get_recent_mean_length(self, window: int = 100) -> float:
        """최근 N개 에피소드 평균 길이"""
        if not self.episodes:
            return 0.0
        recent_episodes = list(self.episodes)[-window:]
        return np.mean([ep.length for ep in recent_episodes])

    def get_collision_rate(self, window: int = 100) -> float:
        """충돌률 계산"""
        if not self.episodes:
            return 0.0
        recent_episodes = list(self.episodes)[-window:]
        total_collisions = sum(
            sum(ep.collisions.values()) for ep in recent_episodes
        )
        return total_collisions / max(1, len(recent_episodes))

    def get_goal_success_rate(self, window: int = 100) -> float:
        """목표 달성률 계산"""
        if not self.episodes:
            return 0.0
        recent_episodes = list(self.episodes)[-window:]
        total_goals = sum(
            sum(ep.goals_reached.values()) for ep in recent_episodes
        )
        return total_goals / max(1, len(recent_episodes))

    def get_performance_metrics(self) -> Dict[str, float]:
        """성능 메트릭 반환"""
        metrics = {}

        if self.training_start_time:
            elapsed_time = time.time() - self.training_start_time
            metrics['elapsed_hours'] = elapsed_time / 3600
            metrics['steps_per_second'] = self.total_timesteps / elapsed_time if elapsed_time > 0 else 0

        if self.step_times:
            metrics['avg_step_time_ms'] = np.mean(self.step_times) * 1000

        if self.gpu_memory_usage:
            metrics['gpu_memory_current_gb'] = self.gpu_memory_usage[-1]
            metrics['gpu_memory_max_gb'] = max(self.gpu_memory_usage)

        return metrics


class MetricsCollectorCallback(BaseCallback):
    """
    중앙 메트릭 수집기
    모든 다른 콜백들이 의존하는 기본 데이터를 수집
    """

    def __init__(self, metrics_store: MetricsStore, monitor_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_store = metrics_store
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.monitor_freq = monitor_freq

    def _on_training_start(self) -> None:
        self.metrics_store.start_training()
        if self.verbose >= 1:
            print("Metrics collection started")

    def _on_step(self) -> bool:
        # GPU 메모리 추적
        gpu_memory = None
        if self.n_calls % self.monitor_freq == 0:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)

        self.metrics_store.record_step(self.num_timesteps, gpu_memory)
        return True


class VehicleSpecificCallback(BaseCallback):
    """
    차량 시뮬레이션 전용 메트릭
    다중 차량 환경의 충돌, 목표 달성 등
    """

    def __init__(self, metrics_store: MetricsStore, num_vehicles: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_store = metrics_store
        self.num_vehicles = num_vehicles
        self.last_episode_count = 0  # 마지막 로깅한 에피소드 수

    def _on_step(self) -> bool:
        # 에피소드 종료 시 즉시 로깅
        if self.metrics_store.episode_count > self.last_episode_count:
            self._log_vehicle_metrics()
            self.last_episode_count = self.metrics_store.episode_count

        return True

    def _log_vehicle_metrics(self, window: int = 50):
        """차량별 상세 메트릭 로깅"""
        if not self.metrics_store.episodes:
            return

        # 최근 에피소드들에서 차량별 데이터 분석
        recent_episodes = list(self.metrics_store.episodes)[-min(window, len(self.metrics_store.episodes)):]

        # 차량별 보상 통계
        for vehicle_id in range(self.num_vehicles):
            vehicle_rewards = []
            vehicle_collisions = 0
            vehicle_goals = 0
            vehicle_outside_roads = 0

            for episode in recent_episodes:
                if len(episode.individual_rewards) > vehicle_id:
                    vehicle_rewards.append(episode.individual_rewards[vehicle_id])

                if vehicle_id in episode.collisions:
                    vehicle_collisions += int(episode.collisions[vehicle_id])

                if vehicle_id in episode.goals_reached:
                    vehicle_goals += int(episode.goals_reached[vehicle_id])

                if vehicle_id in episode.outside_roads:
                    vehicle_outside_roads += int(episode.outside_roads[vehicle_id])

            if vehicle_rewards:
                # 차량별 보상 통계
                self.logger.record(f"vehicle_{vehicle_id}/mean_reward", np.mean(vehicle_rewards))
                self.logger.record(f"vehicle_{vehicle_id}/std_reward", np.std(vehicle_rewards))
                self.logger.record(f"vehicle_{vehicle_id}/max_reward", np.max(vehicle_rewards))
                self.logger.record(f"vehicle_{vehicle_id}/min_reward", np.min(vehicle_rewards))

                # 차량별 성능 지표
                num_episodes = len(recent_episodes)
                self.logger.record(f"vehicle_{vehicle_id}/collision_rate", vehicle_collisions / num_episodes)
                self.logger.record(f"vehicle_{vehicle_id}/goal_success_rate", vehicle_goals / num_episodes)
                self.logger.record(f"vehicle_{vehicle_id}/outside_road_rate", vehicle_outside_roads / num_episodes)


class SystemPerformanceCallback(BaseCallback):
    """
    시스템 성능 모니터링
    GPU/CPU 사용량, 메모리 관리
    """

    def __init__(self, metrics_store: MetricsStore, gpu_memory_limit: int = 5, monitor_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_store = metrics_store
        self.gpu_memory_limit = gpu_memory_limit
        self.monitor_freq = monitor_freq

    def _on_step(self) -> bool:
        # GPU 메모리 정리
        if self.n_calls % self.monitor_freq == 0:
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                if gpu_memory_gb > self.gpu_memory_limit:
                    torch.cuda.empty_cache()
                    if self.verbose >= 1:
                        print(f"GPU memory cleared at {gpu_memory_gb:.2f}GB")
        return True


class TensorBoardLogger(BaseCallback):
    """
    최적화된 TensorBoard 로깅
    구조화된 메트릭 그룹화 및 하이퍼파라미터 로깅
    알고리즘별 특화 메트릭 포함
    """

    def __init__(self, metrics_store: MetricsStore, log_freq: int = 1000,
                 env_config: Optional[Dict] = None, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_store = metrics_store
        self.log_freq = log_freq
        self.env_config = env_config or {}
        self.tb_formatter = None
        self.last_episode_count = 0  # 마지막 로깅한 에피소드 수

    def _on_training_start(self) -> None:
        # TensorBoard 포매터 찾기
        output_formats = self.logger.output_formats
        self.tb_formatter = next(
            (formatter for formatter in output_formats
             if isinstance(formatter, TensorBoardOutputFormat)),
            None
        )

        # 하이퍼파라미터 로깅
        self._log_hyperparameters()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()

        if self.metrics_store.episode_count > self.last_episode_count:
            episode = self.metrics_store.episodes[-1]

            self.logger.record("rollout/episode_reward", episode.reward)
            self.logger.record("rollout/episode_length", episode.length)
            self.logger.dump(episode.timesteps)

            self.last_episode_count = self.metrics_store.episode_count

        return True

    def _log_hyperparameters(self):
        """하이퍼파라미터 로깅"""
        hparam_dict = {
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
        }

        for key, value in algo_specific.items():
            if value is not None:
                hparam_dict[key] = float(value) if isinstance(value, (int, float)) else value

        # 환경 설정
        if self.env_config:
            env_params = {
                "num_vehicles": self.env_config.get('num_vehicles', 1),
                "max_episode_steps": self.env_config.get('max_episode_steps', 1000),
                "num_static_obstacles": self.env_config.get('num_static_obstacles', 0),
                "num_dynamic_obstacles": self.env_config.get('num_dynamic_obstacles', 0),
            }
            hparam_dict.update(env_params)

        # 메트릭 정의
        metric_dict = {
            "rollout/episode_reward_mean_10": 0.0,
            "rollout/episode_length_mean_10": 0.0,
            "vehicles/collision_rate": 0.0,
            "vehicles/goal_success_rate": 0.0,
            "performance/steps_per_second": 0.0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _log_metrics(self):
        """메트릭 로깅"""
        # 기본 RL 메트릭
        mean_reward = self.metrics_store.get_recent_mean_reward(window=10)
        mean_length = self.metrics_store.get_recent_mean_length(window=10)

        self.logger.record("rollout/episode_reward_mean_10", mean_reward)
        self.logger.record("rollout/episode_length_mean_10", mean_length)
        self.logger.record("rollout/best_reward", self.metrics_store.best_reward)

        # 차량 메트릭
        collision_rate = self.metrics_store.get_collision_rate(window=50)
        goal_rate = self.metrics_store.get_goal_success_rate(window=50)

        self.logger.record("vehicles/collision_rate", collision_rate)
        self.logger.record("vehicles/goal_success_rate", goal_rate)

        # 성능 메트릭
        perf_metrics = self.metrics_store.get_performance_metrics()
        for key, value in perf_metrics.items():
            self.logger.record(f"performance/{key}", value)

        # 알고리즘별 특화 메트릭 로깅
        self._log_algorithm_specific_metrics()

    def _log_algorithm_specific_metrics(self):
        """알고리즘별 특화 메트릭 로깅"""
        if hasattr(self.model, '_last_losses') and self.model._last_losses:
            losses = self.model._last_losses

            if hasattr(self.model, 'ent_coef'):  # SAC
                if 'ent_coef_loss' in losses:
                    self.logger.record("train/ent_coef_loss", losses['ent_coef_loss'])
                if 'actor_loss' in losses:
                    self.logger.record("train/actor_loss", losses['actor_loss'])
                if 'critic_loss' in losses:
                    self.logger.record("train/critic_loss", losses['critic_loss'])
                if 'ent_coef' in losses:
                    self.logger.record("train/ent_coef", losses['ent_coef'])

            elif hasattr(self.model, 'n_epochs'):  # PPO
                if 'policy_loss' in losses:
                    self.logger.record("train/policy_loss", losses['policy_loss'])
                if 'value_loss' in losses:
                    self.logger.record("train/value_loss", losses['value_loss'])
                if 'entropy_loss' in losses:
                    self.logger.record("train/entropy_loss", losses['entropy_loss'])
                if 'approx_kl' in losses:
                    self.logger.record("train/approx_kl", losses['approx_kl'])
                if 'clip_fraction' in losses:
                    self.logger.record("train/clip_fraction", losses['clip_fraction'])


class CustomCSVLogger(BaseCallback):
    """
    커스텀 CSV 로깅, 에피소드 종료시 기록으로 데이터 손실 방지
    """
    def __init__(self, metrics_store: MetricsStore, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_store = metrics_store
        self.log_dir = Path(log_dir)
        self.csv_file = self.log_dir / "monitor.csv"
        self.verbose = verbose
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_csv()
        self.last_episode_count = 0  # 마지막 로깅한 에피소드 수

    def _init_csv(self):
        """CSV 파일 초기화"""
        headers = [
            "elapsed_time", "episode", "timesteps", "reward", "length", "collision_rate",
            "success_rate", "best_reward"
        ]

        with open(self.csv_file, 'w', encoding='utf-8') as f:
            f.write(','.join(headers) + '\n')

    def _on_step(self) -> bool:
        if self.metrics_store.episode_count > self.last_episode_count:
            self._log_episode()
            self.last_episode_count = self.metrics_store.episode_count

        return True

    def _log_episode(self):
        """에피소드 데이터 즉시 기록"""
        try:
            episode = self.metrics_store.episodes[-1]

            collision_rate = sum(episode.collisions.values()) / max(1, len(episode.collisions))
            goal_rate = sum(episode.goals_reached.values()) / max(1, len(episode.goals_reached))

            row = [
                f"{episode.elapsed_time:.2f}",
                episode.episode_id,
                episode.timesteps,
                f"{episode.reward:.4f}",
                episode.length,
                f"{collision_rate:.4f}",
                f"{goal_rate:.4f}",
                f"{self.metrics_store.best_reward:.4f}"
            ]

            with open(self.csv_file, 'a', encoding='utf-8') as f:
                f.write(','.join(map(str, row)) + '\n')
                f.flush()

        except Exception as e:
            if self.verbose >= 1:
                print(f"CSV logging failed: {e}")


class SmartCheckpointManager(BaseCallback):
    """
    스마트 체크포인트 관리
    시간 기반 저장 정책
    """

    def __init__(self, metrics_store: MetricsStore, save_freq: int, save_path: str,
                 name_prefix: str = 'model', max_checkpoints: int = 5, save_best_model: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.metrics_store = metrics_store
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.max_checkpoints = max_checkpoints
        self.checkpoint_metadata = []
        self.save_best_model = save_best_model
        self.last_episode_count = 0  # 마지막 로깅한 에피소드 수

        # 저장 디렉토리 생성
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 저장 빈도에 따라 체크포인트 저장
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            self._save_checkpoint()
            self._save_checkpoint_metadata()
            self._cleanup_old_checkpoints()

        # 에피소드 종료 시 최고 성능 모델 저장
        if self.save_best_model:
            if self.metrics_store.episode_count > self.last_episode_count:
                episode = self.metrics_store.episodes[-1]
                if episode.reward >= self.metrics_store.best_reward:
                    self._save_best_model()
                self.last_episode_count = self.metrics_store.episode_count

        return True

    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
        self.model.save(checkpoint_path)

        if self.verbose >= 1:
            print(f"Checkpoint saved at timestep {self.num_timesteps}: {checkpoint_path}")

    def _save_checkpoint_metadata(self):
        """체크포인트 메타데이터 저장"""
        metadata = {
            'timestep': self.num_timesteps,
            'episode': self.metrics_store.episode_count,
            'mean_reward': self.metrics_store.get_recent_mean_reward(10),
            'best_reward': self.metrics_store.best_reward,
            'collision_rate': self.metrics_store.get_collision_rate(50),
            'goal_success_rate': self.metrics_store.get_goal_success_rate(50),
            'timestamp': time.time()
        }

        self.checkpoint_metadata.append(metadata)

        # 메타데이터 파일 저장
        metadata_file = os.path.join(self.save_path, "checkpoint_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.checkpoint_metadata, f, indent=2)

    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 정리"""
        try:
            files = [f for f in os.listdir(self.save_path)if f.startswith(self.name_prefix) and f.endswith('.zip') and 'best' not in f and 'replay_buffer' not in f and 'final' not in f and 'interrupted' not in f]
            files = sorted(files, key=lambda f: int(re.search(r'_(\d+)_steps', f).group(1)))

            for old_file in files[:-self.max_checkpoints]:
                old_path = os.path.join(self.save_path, old_file)
                os.remove(old_path)
                if self.verbose >= 1:
                    print(f"Removed old checkpoint: {old_file}")

        except Exception as e:
            if self.verbose >= 1:
                print(f"Checkpoint cleanup failed: {e}")

    def _save_best_model(self):
        """베스트 모델 저장"""
        checkpoint_path = os.path.join(self.save_path, f"{self.name_prefix}_best.zip")
        self.model.save(checkpoint_path)
        if self.name_prefix == 'sac':
            self.model.save_replay_buffer(os.path.join(self.save_path, f"{self.name_prefix}_best_replay_buffer"))


# 콜백 팩토리 함수
def create_optimized_callbacks(config: Dict[str, Any], log_dir: str, models_dir: str) -> List[BaseCallback]:
    """
    최적화된 콜백 시스템 생성 팩토리
    """
    # 중앙 메트릭 스토어 생성
    metrics_store = MetricsStore(
        max_episodes=config.get('max_episodes_history', 1000),
        max_steps=config.get('max_steps_history', 10000)
    )

    callbacks = []

    # 1. 메트릭 수집기 (가장 먼저)
    metrics_collector = MetricsCollectorCallback(
        metrics_store,
        config.get('monitoring_freq', 1000),
        config.get('verbose', 0)
    )
    callbacks.append(metrics_collector)

    # 2. 시스템 성능 모니터링
    system_perf = SystemPerformanceCallback(
        metrics_store,
        config.get('gpu_memory_limit', 5),
        config.get('monitoring_freq', 1000),
        config.get('verbose', 0)
    )
    callbacks.append(system_perf)

    # 3. 차량별 메트릭
    if config.get('num_vehicles', 1) >= 1:
        vehicle_metrics = VehicleSpecificCallback(
            metrics_store,
            config['num_vehicles'],
            config.get('verbose', 0)
        )
        callbacks.append(vehicle_metrics)

    # 4. 커스텀 CSV 로거
    csv_logger = CustomCSVLogger(
        metrics_store,
        log_dir,
        config.get('verbose', 0)
    )
    callbacks.append(csv_logger)

    # 5. TensorBoard 로거
    tensorboard_logger = TensorBoardLogger(
        metrics_store,
        config.get('logging_freq', 1000),
        config,
        config.get('verbose', 0)
    )
    callbacks.append(tensorboard_logger)

    # 6. 체크포인트 매니저
    checkpoint_manager = SmartCheckpointManager(
        metrics_store,
        config.get('checkpoint_freq', 10000),
        models_dir,
        config.get('algorithm', 'model'),
        config.get('max_checkpoints', 5),
        config.get('save_best_model', True),
        config.get('verbose', 0)
    )
    callbacks.append(checkpoint_manager)

    return callbacks, metrics_store


def get_callback_summary(callbacks: List[BaseCallback]) -> Dict[str, Any]:
    """콜백 요약 정보"""
    return {
        'total_callbacks': len(callbacks),
        'callback_types': [type(cb).__name__ for cb in callbacks],
        'optimized_system': True,
        'centralized_metrics': True
    }


# 커스텀 RL 알고리즘
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
        self.total_reward = 0
        self.individual_rewards = None
        self.metrics_store = None  # 메트릭 스토어 참조

    def set_env_info(self, rl_env):
        """
        BasicRLDrivingEnv 정보 설정
        """
        self.rl_env = rl_env
        self.num_vehicles = rl_env.num_vehicles

    def set_metrics_store(self, metrics_store: MetricsStore):
        """메트릭 스토어 설정"""
        self.metrics_store = metrics_store
        self._last_losses = {}  # 알고리즘 로스 추적용

    def _excluded_save_params(self) -> list[str]:
        # 제외된 저장 파라미터 목록
        return super()._excluded_save_params() + ["rl_env", "metrics_store"]

    def _reset(self):
        # 환경 리셋
        prev_observations, active_agents = self.rl_env.reset()
        self.prev_observations = prev_observations
        self.prev_active_mask = np.array(active_agents)
        self.total_reward = 0.0
        self.individual_rewards = np.zeros(self.num_vehicles)

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
        actions = np.zeros((self.num_vehicles, self.action_space.shape[-1]), dtype=np.float32)
        if self._n_updates < self.learning_starts:
            # 랜덤 행동 (벡터화)
            if active_mask.any():
                random_actions = np.stack([self.rl_env.env.action_space.sample() for _ in range(active_mask.sum())])
                # random_actions[:, 1] = random_actions[:, 1] / 5
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
        self.num_timesteps += 1
        self.total_reward += reward
        for vehicle_id in range(self.num_vehicles):
            if vehicle_id < len(info['rewards']):
                self.individual_rewards[vehicle_id] += info['rewards'][vehicle_id]

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

        # 7) 콜백 호출
        callback.update_locals(locals())
        if not callback.on_step():
            return False

        # 8) 에피소드 종료 시 처리
        if done:
            self._handle_episode_end(info)
            self._reset()

        return True

    def _handle_episode_end(self, info):
        """에피소드 종료 시 처리 (메트릭 스토어 업데이트 포함)"""
        # 기본 로깅
        print(f"Episode {info['episode_count']}, "f"Steps: {info['episode_length']}, "f"Total Reward: {self.total_reward:.2f}")

        # 메트릭 스토어에 에피소드 데이터 기록
        if self.metrics_store:
            episode_data = EpisodeData(
                episode_id=info['episode_count'],
                timesteps=self.num_timesteps,
                reward=self.total_reward,
                length=info['episode_length'],
                collisions=info['collisions'].copy(),
                goals_reached=info['reached_targets'].copy(),
                outside_roads=info['outside_roads'].copy(),
                individual_rewards=self.individual_rewards.tolist(),
                elapsed_time=time.time() - self.metrics_store.training_start_time if self.metrics_store.training_start_time else 0
            )

            self.metrics_store.record_episode(episode_data)

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
        self.total_reward = 0
        self.individual_rewards = None
        self._is_new_episode = True
        self.metrics_store = None

    def set_env_info(self, rl_env):
        """
        BasicRLDrivingEnv 정보 설정
        """
        self.rl_env = rl_env
        self.num_vehicles = rl_env.num_vehicles

    def set_metrics_store(self, metrics_store: MetricsStore):
        """메트릭 스토어 설정"""
        self.metrics_store = metrics_store
        self._last_losses = {}  # 알고리즘 로스 추적용

    def _excluded_save_params(self) -> list[str]:
        # 제외된 저장 파라미터 목록
        return super()._excluded_save_params() + ["rl_env", "metrics_store"]

    def _reset(self):
        # 환경 리셋
        prev_observations, active_agents = self.rl_env.reset()
        self.prev_observations = prev_observations
        self.prev_active_mask = np.array(active_agents)
        self.total_reward = 0.0
        self.individual_rewards = np.zeros(self.num_vehicles)
        self._is_new_episode = True

    def _custom_rollout_step(self, callback: BaseCallback, n_steps: int) -> bool:
        """
        실제 rollouts를 수집하는 함수.
        """
        step_count = 0
        while step_count < n_steps:
            # 1) 키보드 이벤트 처리
            self.rl_env.handle_keyboard_input()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False

            # 2) 행동, 값, 로그 확률 계산
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

                # NumPy 및 형태 변환
                actions_np = actions_tensor.cpu().numpy()
                values_tensor = values_tensor.view(-1)
                log_prob_tensor = log_prob_tensor.view(-1)

                # 활성 에이전트 위치에 해당 값들을 대입
                actions[active_mask] = actions_np
                values[active_mask] = values_tensor
                log_probs[active_mask] = log_prob_tensor

            # 3) 환경 스텝
            next_observations, reward, done, _, info = self.rl_env.step(actions)
            self.num_timesteps += 1
            self.total_reward += reward
            for vehicle_id in range(self.num_vehicles):
                if vehicle_id < len(info['rewards']):
                    self.individual_rewards[vehicle_id] += info['rewards'][vehicle_id]

            # 4) 렌더링
            self.rl_env.render()

            # 5) 활성 에이전트마다 RolloutBuffer에 transition 추가
            active_indices = np.where(active_mask)[0]
            for idx in active_indices:
                # 개별 차량을 rollout buffer에 저장
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

            # 7) 콜백 호출
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # 8) 에피소드 종료 시 처리
            if done:
                self._handle_episode_end(info)
                self._reset()

        return True

    def _handle_episode_end(self, info):
        """에피소드 종료 시 처리 (메트릭 스토어 업데이트 포함)"""
        # 기본 로깅
        print(f"Episode {info['episode_count']}, "f"Steps: {info['episode_length']}, "f"Total Reward: {self.total_reward:.2f}")

        # 메트릭 스토어에 에피소드 데이터 기록
        if self.metrics_store:
            episode_data = EpisodeData(
                episode_id=info['episode_count'],
                timesteps=self.num_timesteps,
                reward=self.total_reward,
                length=info['episode_length'],
                collisions=info['collisions'].copy(),
                goals_reached=info['reached_targets'].copy(),
                outside_roads=info['outside_roads'].copy(),
                individual_rewards=self.individual_rewards.tolist(),
                elapsed_time=time.time() - self.metrics_store.training_start_time if self.metrics_store.training_start_time else 0
            )

            self.metrics_store.record_episode(episode_data)

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


# 커스텀 피처 익스트랙터
class ResidualLayerNormMLP(nn.Module):
    """
    MLP with LayerNorm, Residual Connections, and Orthogonal Init + GELU Activation
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
