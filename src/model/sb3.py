# -*- coding: utf-8 -*-
import os
import time
import torch
import torch.nn as nn
import pygame
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
                self.prev_active_mask = np.array(self.rl_env.active_agents)
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

        # 2) 행동 결정
        observations = self.prev_observations
        active_mask = self.prev_active_mask
        if self._n_updates < self.learning_starts:
            # 랜덤 행동 (벡터화)
            actions = np.zeros((self.num_vehicles, 3))
            if active_mask.any():
                random_actions = np.array([self.rl_env.env.action_space.sample() for _ in range(active_mask.sum())])
                random_actions[:, 1] = random_actions[:, 1] / 5
                # random_actions[:, 2] = random_actions[:, 2] / 3
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

        # 3) 환경 스텝 & 보상 누적
        next_observations, reward, done, _, info = self.rl_env.step(actions)
        self.total_reward += reward
        self.episode_steps += 1
        self.num_timesteps += 1

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
            # 활성화 마스크 업데이트
            self.prev_active_mask = np.array(self.rl_env.active_agents)
        return True

class ResidualLayerNormMLP(nn.Module):
    """
    MLP with LayerNorm, Residual Connections, and Normal Init (mean=0, std=0.02) + GELU Activation
    """
    def __init__(self, input_dim: int, net_arch: list[int]):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_dim = input_dim
        for hidden_dim in net_arch:
            block = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            self.blocks.append(block)
            in_dim = hidden_dim
        # Apply weight initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            residual = x
            x = block(x)
            if residual.shape == x.shape:
                x = x + residual
        return x

    @staticmethod
    def _init_weights(m):
        # Normal initialization for Linear layers: N(0, 0.02^2)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class LidarConvNet(nn.Module):
    """
    1D CNN for Lidar Data
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.lidar_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1),
            nn.Conv1d(1, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.AdaptiveAvgPool1d(1),
        )

        # 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)         # (B, 1, L)
        x = self.encoder(x)        # (B, out_dim, 1)
        return x[..., 0]           # (B, out_dim)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
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

class EnhancedFeatureExtractor(BaseFeaturesExtractor):
    """
    차량 상태, 궤적 데이터, LIDAR 데이터를 별도로 처리하는 향상된 특성 추출기
    - 차량 상태: MLP
    - 궤적 데이터: MLP
    - LIDAR 데이터: 1D CNN
    """
    def __init__(self, obs_space, net_arch: list[int]):
        # 입력 차원 정의
        self.vehicle_state_dim = 13
        self.trajectory_dim = 42
        self.lidar_dim = 36
        self.lidar_out = 8
        self.feature_dim = net_arch[-1] + self.lidar_out
        super().__init__(obs_space, features_dim=self.feature_dim)

        self.flatten = nn.Flatten()

        # 차량 상태 및 궤적 데이터 처리 네트워크
        self.mlp = ResidualLayerNormMLP(self.vehicle_state_dim + self.trajectory_dim, net_arch)

        # LIDAR 데이터 처리 네트워크 (1D CNN)
        self.cnn = LidarConvNet(self.lidar_dim, self.lidar_out)

    def forward(self, obs):
        # 입력 분리
        mlp_input = obs[:, :(self.vehicle_state_dim + self.trajectory_dim)]
        cnn_input = obs[:, (self.vehicle_state_dim + self.trajectory_dim):]

        # 각 네트워크 통과
        mlp_features = self.mlp(mlp_input)
        cnn_features = self.cnn(cnn_input)

        # 최종 특성 결합
        return torch.cat([mlp_features, cnn_features], dim=1)
