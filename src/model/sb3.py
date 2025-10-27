# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import pygame
import numpy as np
from math import isnan
import json
import gzip
import gymnasium as gym
from typing import Dict, Any, Optional, Union, List
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, EvalCallback, CallbackList
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
    early_termination: bool

@dataclass
class StepData:
    """스텝 데이터 구조"""
    timestep: int
    gpu_memory: Optional[float]

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

    def record_step(self, step_data: StepData):
        """스텝 기록"""
        current_time = time.time()

        if self.last_step_time is not None:
            step_time = current_time - self.last_step_time
            self.step_times.append(step_time)

        if step_data.gpu_memory is not None:
            self.gpu_memory_usage.append(step_data.gpu_memory)

        self.last_step_time = current_time
        self.total_timesteps = step_data.timestep

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

    def get_success_rate(self, window: int = 100) -> float:
        """목표 달성률 계산"""
        if not self.episodes:
            return 0.0
        recent_episodes = list(self.episodes)[-window:]
        total_goals = sum(
            sum(ep.goals_reached.values()) for ep in recent_episodes
        )
        return total_goals / max(1, len(recent_episodes))

    def get_outside_road_rate(self, window: int = 100) -> float:
        """도로 이탈률 계산"""
        if not self.episodes:
            return 0.0
        recent_episodes = list(self.episodes)[-window:]
        total_outside = sum(
            sum(ep.outside_roads.values()) for ep in recent_episodes
        )
        return total_outside / max(1, len(recent_episodes))

    def get_termination_rate(self, window: int = 100) -> float:
        """조기 종료 비율 계산"""
        if not self.episodes:
            return 0.0
        recent_episodes = list(self.episodes)[-window:]
        total_early_terminations = sum(
            1 for ep in recent_episodes if ep.early_termination
        )
        return total_early_terminations / max(1, len(recent_episodes))

    def get_rates(self, window: int = 100) -> tuple[float, float, float, float]:
        """목표 달성률, 조기 종료 비율, 도로 이탈율, 충돌률 반환"""
        if not self.episodes:
            return 0.0, 0.0, 0.0, 0.0
        recent_episodes = list(self.episodes)[-window:]
        length = len(recent_episodes)

        total_goals = sum(
            sum(ep.goals_reached.values()) for ep in recent_episodes
        )
        total_early_terminations = sum(
            1 for ep in recent_episodes if ep.early_termination
        )
        total_outside = sum(
            sum(ep.outside_roads.values()) for ep in recent_episodes
        )
        total_collisions = sum(
            sum(ep.collisions.values()) for ep in recent_episodes
        )

        success_rate = total_goals / max(1, length)
        termination_rate = total_early_terminations / max(1, length)
        outside_road_rate = total_outside / max(1, length)
        collision_rate = total_collisions / max(1, length)
        return success_rate, termination_rate, outside_road_rate, collision_rate

    def get_performance_metrics(self) -> Dict[str, float]:
        """성능 메트릭 반환"""
        metrics = {}

        if self.training_start_time:
            elapsed_time = time.time() - self.training_start_time
            metrics['time/elapsed(h)'] = elapsed_time / 3600
            metrics['step/per_second(num)'] = self.total_timesteps / elapsed_time if elapsed_time > 0 else 0

        if self.step_times:
            metrics['step/avg_time(ms)'] = np.mean(self.step_times) * 1000

        if self.gpu_memory_usage:
            metrics['gpu/current_memory(gb)'] = self.gpu_memory_usage[-1]
            metrics['gpu/max_memory(gb)'] = max(self.gpu_memory_usage)

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

        step_data = StepData(timestep=self.num_timesteps, gpu_memory=gpu_memory)
        self.metrics_store.record_step(step_data)
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
            self._log_vehicle_metrics(10)
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
                self.logger.record(f"vehicle_{vehicle_id}/reward/mean/{window}", np.mean(vehicle_rewards))
                self.logger.record(f"vehicle_{vehicle_id}/reward/std/{window}", np.std(vehicle_rewards))
                self.logger.record(f"vehicle_{vehicle_id}/reward/max/{window}", np.max(vehicle_rewards))
                self.logger.record(f"vehicle_{vehicle_id}/reward/min/{window}", np.min(vehicle_rewards))

                # 차량별 성능 지표
                num_episodes = len(recent_episodes)
                self.logger.record(f"vehicle_{vehicle_id}/rate/success/{window}", vehicle_goals / num_episodes)
                self.logger.record(f"vehicle_{vehicle_id}/rate/outside/{window}", vehicle_outside_roads / num_episodes)
                self.logger.record(f"vehicle_{vehicle_id}/rate/collision/{window}", vehicle_collisions / num_episodes)


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

    def _on_training_end(self) -> None:
        """훈련 종료 시 최종 성능과 하이퍼파라미터를 기록합니다."""
        if self.tb_formatter is None:
            return

        # 1. 하이퍼파라미터 로깅
        hparam_dict = self._get_hparam_dict()

        # 2. 최종 성능 지표 로깅
        mean_reward_10 = self.metrics_store.get_recent_mean_reward(window=10)
        mean_length_10 = self.metrics_store.get_recent_mean_length(window=10)

        success_rate_10, termination_rate_10, outside_road_rate_10, collision_rate_10 = self.metrics_store.get_rates(window=10)
        success_rate_30, termination_rate_30, outside_road_rate_30, collision_rate_30 = self.metrics_store.get_rates(window=30)
        success_rate_50, term_rate_50, out_rate_50, coll_rate_50 = self.metrics_store.get_rates(window=50)
        perf_metrics = self.metrics_store.get_performance_metrics()

        metric_dict = {
            "episode/reward/mean/10": mean_reward_10,
            "episode/length/mean/10": mean_length_10,
            "episode/reward/best": self.metrics_store.best_reward,
            "rate/success/10": success_rate_10,
            "rate/termination/10": termination_rate_10,
            "rate/outside/10": outside_road_rate_10,
            "rate/collision/10": collision_rate_10,
            "rate/success/30": success_rate_30,
            "rate/termination/30": termination_rate_30,
            "rate/outside/30": outside_road_rate_30,
            "rate/collision/30": collision_rate_30,
            "rate/success/50": success_rate_50,
            "rate/termination/50": term_rate_50,
            "rate/outside/50": out_rate_50,
            "rate/collision/50": coll_rate_50,
            "time/elapsed(h)": perf_metrics['time/elapsed(h)'],
            "step/per_second(num)": perf_metrics['step/per_second(num)'],
            "step/avg_time(ms)": perf_metrics['step/avg_time(ms)'],
        }

        cleaned_metric_dict = {}
        for key, value in metric_dict.items():
            if value is None or (isinstance(value, float) and isnan(value)):
                cleaned_metric_dict[key] = "N/A"
            elif isinstance(value, float):
                cleaned_metric_dict[key] = f"{value:.4f}" # 소수점 4자리까지
            else:
                cleaned_metric_dict[key] = value

        # 3. 딕셔너리를 Markdown 테이블 형식의 문자열로 변환
        hparam_md = "### Hyperparameters\n| Parameter | Value |\n|:---|:---|\n"
        for key, value in hparam_dict.items():
            hparam_md += f"| {key} | {value} |\n"

        metric_md = "\n### Metrics\n| Metric | Value |\n|:---|:---|\n"
        for key, value in cleaned_metric_dict.items():
            metric_md += f"| {key} | {value} |\n"

        # 4. add_text를 사용하여 TensorBoard에 기록
        final_summary_text = hparam_md + metric_md
        self.tb_formatter.writer.add_text("Final Summary", final_summary_text, self.num_timesteps)
        self.tb_formatter.writer.flush()

        if self.verbose > 0:
            print("\n" + "="*30)
            print("HParams logged to TensorBoard.")
            print("="*30 + "\n")

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()

        if self.metrics_store.episode_count > self.last_episode_count:
            # 최근 10개 에피소드 메트릭
            mean_reward = self.metrics_store.get_recent_mean_reward(window=10)
            mean_length = self.metrics_store.get_recent_mean_length(window=10)

            self.logger.record("rollout/episode/reward/mean/10", mean_reward)
            self.logger.record("rollout/episode/length/mean/10", mean_length)
            self.logger.record("rollout/episode/reward/best", self.metrics_store.best_reward)

            # 메트릭
            success_rate_30, termination_rate_30, outside_road_rate_30, collision_rate_30 = self.metrics_store.get_rates(window=30)
            self.logger.record("rollout/rate/success/30", success_rate_30)
            self.logger.record("rollout/rate/termination/30", termination_rate_30)
            self.logger.record("rollout/rate/outside/30", outside_road_rate_30)
            self.logger.record("rollout/rate/collision/30", collision_rate_30)

            success_rate_50, termination_rate_50, outside_road_rate_50, collision_rate_50 = self.metrics_store.get_rates(window=50)
            self.logger.record("rollout/rate/success/50", success_rate_50)
            self.logger.record("rollout/rate/termination/50", termination_rate_50)
            self.logger.record("rollout/rate/outside/50", outside_road_rate_50)
            self.logger.record("rollout/rate/collision/50", collision_rate_50)

            # 방금전 에피소드 메트릭
            episode = self.metrics_store.episodes[-1]
            self.logger.record("rollout/episode/reward", episode.reward)
            self.logger.record("rollout/episode/length", episode.length)
            self.logger.dump(episode.timesteps)

            self.last_episode_count = self.metrics_store.episode_count

        return True

    def _get_hparam_dict(self) -> Dict[str, Any]:
        """하이퍼파라미터 로깅"""
        hparam_dict = {}

        for key, value in self.env_config['hyperparameters'].items():
            if value is not None:
                hparam_dict[key] = float(value) if isinstance(value, (int, float)) else value

        # 환경 설정
        if self.env_config:
            env_params = {
                "env/num_vehicles": self.env_config.get('num_vehicles', 1),
                "env/max_episode_steps": self.env_config.get('max_episode_steps', 1000),
                "env/num_static_obstacles": self.env_config.get('num_static_obstacles', 0),
                "env/num_dynamic_obstacles": self.env_config.get('num_dynamic_obstacles', 0),
                "env/termination_check_step": self.env_config.get('termination_check_step', 0),
                "env/progress_change_threshold": self.env_config.get('progress_change_threshold', 0),
            }
            hparam_dict.update(env_params)

            reward_params = {
                "reward/success": self.env_config['reward_factor'].get('success', 0.0),
                "reward/collision": self.env_config['reward_factor'].get('collision', 0.0),
                "reward/outside": self.env_config['reward_factor'].get('outside', 0.0),
                "reward/termination": self.env_config['reward_factor'].get('termination', 0.0),
                "reward/progress": self.env_config['reward_factor'].get('w_progress', 0.0),
                "reward/lane": self.env_config['reward_factor'].get('w_lane', 0.0),
                "reward/speed": self.env_config['reward_factor'].get('w_speed', 0.0),
                "reward/speed/under": self.env_config['reward_factor'].get('s_speed_under', 0.0),
                "reward/speed/over": self.env_config['reward_factor'].get('s_speed_over', 0.0),
            }
            hparam_dict.update(reward_params)

        return hparam_dict

    def _log_metrics(self):
        """메트릭 로깅"""
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
            "elapsed_time", "episode", "timesteps", "reward", "length", "rate_success", "termination",
            "rate_collision", "best_reward"
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

            success_rate = sum(episode.goals_reached.values()) / max(1, len(episode.goals_reached))
            termination = 1.0 if episode.early_termination else 0.0
            collision_rate = sum(episode.collisions.values()) / max(1, len(episode.collisions))

            row = [
                f"{episode.elapsed_time:.2f}",
                episode.episode_id,
                episode.timesteps,
                f"{episode.reward:.4f}",
                episode.length,
                f"{success_rate:.4f}",
                f"{termination:.4f}",
                f"{collision_rate:.4f}",
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
        self.best_success_rate = -np.inf
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
                success_rate = self.metrics_store.get_success_rate(window=30)
                if success_rate > self.best_success_rate:
                    self.best_success_rate = success_rate
                    self._save_best_model()
                self.last_episode_count = self.metrics_store.episode_count
                # episode = self.metrics_store.episodes[-1]
                # if episode.reward >= self.metrics_store.best_reward:
                #     self._save_best_model()
                # self.last_episode_count = self.metrics_store.episode_count

        return True

    def _save_checkpoint(self):
        """체크포인트 저장"""
        checkpoint_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
        self.model.save(checkpoint_path)

        if self.verbose >= 1:
            print(f"Checkpoint saved at timestep {self.num_timesteps}: {checkpoint_path}")

    def _save_checkpoint_metadata(self):
        """체크포인트 메타데이터 저장"""
        success_rate, termination_rate, outside_road_rate, collision_rate = self.metrics_store.get_rates(window=50)
        metadata = {
            'timestep': self.num_timesteps,
            'episode': self.metrics_store.episode_count,
            'mean_reward': self.metrics_store.get_recent_mean_reward(window=50),
            'best_reward': self.metrics_store.best_reward,
            'rate_success': success_rate,
            'rate_termination': termination_rate,
            'rate_outside': outside_road_rate,
            'rate_collision': collision_rate,
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
        config.get('algorithm'),
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
    def __init__(self, *args, logging_freq: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        # 추가 속성 설정
        self.rl_env = None
        self.num_vehicles = None
        self.prev_observations = None
        self.prev_active_mask = None
        self.total_reward = 0
        self.individual_rewards = None
        self.metrics_store = None  # 메트릭 스토어 참조
        self.logging_freq = logging_freq

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

        # 5) 콜백 호출
        callback.update_locals(locals())
        if not callback.on_step():
            return False

        # 6) 활성화된 에이전트만 경험 저장
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

        # 7) 내부 상태 갱신
        self.prev_observations = next_observations  # shape=(num_vehicles, obs_dim)
        self.prev_active_mask = np.array(info['active_agents'], dtype=bool)

        # 8) 에피소드 종료 시 처리
        if done:
            self._handle_episode_end(info)
            self._reset()

        return True

    def _handle_episode_end(self, info):
        """에피소드 종료 시 처리 (메트릭 스토어 업데이트 포함)"""
        # 기본 로깅
        print(f"Episode {info['episode_count']}, "f"Steps: {info['episode_length']}, "f"Total Reward: {self.total_reward:.2f}, "f"Early Termination: {info['early_termination']}")

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
                early_termination=info['early_termination'],
                individual_rewards=self.individual_rewards.tolist(),
                elapsed_time=time.time() - self.metrics_store.training_start_time if self.metrics_store.training_start_time else 0,
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

    def _get_tb_writer(self):
        """
        로거에서 TensorBoard writer를 찾아 반환합니다.
        """
        output_formats = self.logger.output_formats
        tb_formatter = next(
            (formatter for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)),
            None
        )
        return tb_formatter.writer if tb_formatter else None

    def train(self, batch_size: int, gradient_steps: int) -> None:
        """
        SAC 훈련 스텝을 수행하고 추가 메트릭을 로깅합니다.
        """
        super().train(batch_size=batch_size, gradient_steps=gradient_steps)

        if self.num_timesteps % self.logging_freq == 0:
            # 텐서보드 writer 가져오기
            tb_writer = self._get_tb_writer()

            # 훈련 전에 리플레이 버퍼에서 데이터 샘플링
            if tb_writer:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                with torch.no_grad():
                    # Critic을 통해 Q-가치 예측
                    qf1_values, qf2_values = self.critic(replay_data.observations, replay_data.actions)
                    q_values = torch.min(qf1_values, qf2_values).cpu().numpy().flatten()
                    actions = replay_data.actions.cpu().numpy()
                    throttle_actions = actions[:, 0]
                    steering_actions = actions[:, 1]

                    # 스칼라 값 로깅 (Q-Value)
                    self.logger.record("sac/q_values/mean", np.mean(q_values))

                    # 스칼라 값 로깅 (가속/제동)
                    self.logger.record("sac/actions/throttle/mean", np.mean(throttle_actions))
                    self.logger.record("sac/actions/throttle/std", np.std(throttle_actions))

                    # 스칼라 값 로깅 (조향)
                    self.logger.record("sac/actions/steering/mean", np.mean(steering_actions))
                    self.logger.record("sac/actions/steering/std", np.std(steering_actions))

                    # 히스토그램 로깅
                    tb_writer.add_histogram("sac/q_values", q_values, self.num_timesteps)
                    tb_writer.add_histogram("sac/actions/throttle", throttle_actions, self.num_timesteps)
                    tb_writer.add_histogram("sac/actions/steering", steering_actions, self.num_timesteps)

class PPOVehicleAlgorithm(PPO):
    def __init__(self, *args, logging_freq: int = 100, **kwargs):
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
        self.logging_freq = logging_freq

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

            # 5) 콜백 호출
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            # 6) 활성 에이전트마다 RolloutBuffer에 transition 추가
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

            # 7) 내부 상태 갱신
            if self._is_new_episode:
                self._is_new_episode = False
            self.prev_observations = next_observations  # shape=(num_vehicles, obs_dim)
            self.prev_active_mask = np.array(info['active_agents'], dtype=bool)

            # 8) 에피소드 종료 시 처리
            if done:
                self._handle_episode_end(info)
                self._reset()

        return True

    def _handle_episode_end(self, info):
        """에피소드 종료 시 처리 (메트릭 스토어 업데이트 포함)"""
        # 기본 로깅
        print(f"Episode {info['episode_count']}, "f"Steps: {info['episode_length']}, "f"Total Reward: {self.total_reward:.2f}, "f"Early Termination: {info['early_termination']}")

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
                early_termination=info['early_termination'],
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

            if self.num_timesteps % self.logging_freq == 0:
                if self.rollout_buffer.full:
                    # 어드밴티지 계산
                    self.rollout_buffer.compute_returns_and_advantage(
                        last_values=torch.zeros(self.num_vehicles,),  # Dummy last values
                        dones=np.zeros(self.num_vehicles,) # Dummy dones
                    )
                    self._log_ppo_specific_metrics()

            self.train()
            self.rollout_buffer.reset()

        callback.on_training_end()
        return self

    def _get_tb_writer(self):
        """
        로거에서 TensorBoard writer를 찾아 반환합니다.
        """
        output_formats = self.logger.output_formats
        tb_formatter = next(
            (formatter for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)),
            None
        )
        return tb_formatter.writer if tb_formatter else None

    def _log_ppo_specific_metrics(self):
        """PPO 롤아웃 데이터를 텐서보드에 로깅합니다."""
        # 텐서보드 writer 가져오기
        tb_writer = self._get_tb_writer()
        if not tb_writer:
            return

        # 데이터 추출 (Numpy 배열로 변환)
        actions = self.rollout_buffer.actions
        throttle_actions = actions[:, 0]
        steering_actions = actions[:, 1]
        advantages = self.rollout_buffer.advantages.flatten()
        values = self.rollout_buffer.values.flatten()

        # 스칼라 값 로깅 (가속/제동)
        self.logger.record("ppo/actions/throttle/mean", np.mean(throttle_actions))
        self.logger.record("ppo/actions/throttle/std", np.std(throttle_actions))

        # 스칼라 값 로깅 (조향)
        self.logger.record("ppo/actions/steering/mean", np.mean(steering_actions))
        self.logger.record("ppo/actions/steering/std", np.std(steering_actions))

        # 스칼라 값 로깅 (Advantages & Values)
        self.logger.record("ppo/advantages/mean", np.mean(advantages))
        self.logger.record("ppo/values/mean", np.mean(values))

        # 히스토그램 로깅
        tb_writer.add_histogram("ppo/actions/throttle", throttle_actions, self.num_timesteps)
        tb_writer.add_histogram("ppo/actions/steering", steering_actions, self.num_timesteps)
        tb_writer.add_histogram("ppo/advantages", advantages, self.num_timesteps)
        tb_writer.add_histogram("ppo/values", values, self.num_timesteps)


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

class CustomFeatureExtractor2(BaseFeaturesExtractor):
    """
    RNN(GRU)을 사용하여 (k, features) 시퀀스를 처리하는 커스텀 피처 추출기
    입력: (Batch, k, obs_dim)
    출력: (Batch, features_dim)
    """
    def __init__(self, observation_space: gym.Space, net_arch: list[int]):
        # features_dim은 RNN의 은닉 크기(출력 크기)가 됩니다.
        super().__init__(observation_space, features_dim=net_arch[-1])

        # observation_space.shape는 (k, obs_dim)
        # (gym.Space는 배치 차원을 포함하지 않습니다)
        if len(observation_space.shape) != 2:
            raise ValueError(
                f"예상된 관측 공간 형태는 (k, obs_dim)이지만, "
                f"실제 형태는 {observation_space.shape}입니다."
            )

        k_frames = observation_space.shape[0] # 프레임 수 (k)
        obs_dim = observation_space.shape[1]  # 관측 차원 (obs_dim)

        self.rnn = nn.GRU(
            input_size=obs_dim,       # 27
            hidden_size=net_arch[-1], # 128 (지정된 features_dim)
            num_layers=1,             # RNN 레이어 수
            batch_first=True          # 입력 텐서 형태: (Batch, Seq, Feat)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations 텐서의 형태: (Batch, k, obs_dim)
        (Batch는 여기서 num_vehicles 또는 활성화된 에이전트 수)
        """
        # rnn_output 형태: (Batch, k, 128)
        # hidden_state 형태: (1, Batch, 128)
        # rnn(observations)는 (시퀀스 전체의 출력, 마지막 은닉 상태)를 반환합니다.
        rnn_output, hidden_state = self.rnn(observations)

        # 우리는 시퀀스의 마지막 타임스텝의 출력만 정책망에 전달합니다.
        # rnn_output[:, -1, :]는 (Batch, 128) 형태가 됩니다.
        return rnn_output[:, -1, :]
