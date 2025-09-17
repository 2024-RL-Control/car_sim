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
from stable_baselines3.common.logger import HParam

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

class HyperparameterLoggingCallback(BaseCallback):
    """
    하이퍼파라미터와 핵심 메트릭을 TensorBoard HPARAMS 탭에 로깅하는 콜백
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        """학습 시작 시 하이퍼파라미터 로깅"""
        # 하이퍼파라미터 딕셔너리 구성
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }

        # 알고리즘별 특화 하이퍼파라미터 추가
        if hasattr(self.model, 'batch_size'):
            hparam_dict["batch_size"] = self.model.batch_size
        if hasattr(self.model, 'buffer_size'):
            hparam_dict["buffer_size"] = self.model.buffer_size
        if hasattr(self.model, 'tau'):
            hparam_dict["tau"] = self.model.tau
        if hasattr(self.model, 'ent_coef'):
            hparam_dict["ent_coef"] = self.model.ent_coef
        if hasattr(self.model, 'n_steps'):
            hparam_dict["n_steps"] = self.model.n_steps
        if hasattr(self.model, 'n_epochs'):
            hparam_dict["n_epochs"] = self.model.n_epochs
        if hasattr(self.model, 'clip_range'):
            hparam_dict["clip_range"] = self.model.clip_range

        # 환경 특화 하이퍼파라미터 추가 (가능한 경우)
        if hasattr(self.model, 'rl_env'):
            rl_env = self.model.rl_env
            hparam_dict.update({
                "num_vehicles": rl_env.num_vehicles,
                "max_episode_steps": rl_env.max_episode_steps,
                "num_static_obstacles": rl_env.num_static_obstacles,
                "num_dynamic_obstacles": rl_env.num_dynamic_obstacles,
            })

        # 메트릭 딕셔너리 (TensorBoard에서 추적할 핵심 지표들)
        metric_dict = {
            "train/episode_reward": 0,
            "train/episode_length": 0,
            "train/active_vehicles": 0,
            "resources/gpu_memory_gb": 0,
            "time/steps_per_second": 0,
        }

        # 알고리즘별 특화 메트릭 추가
        if "SAC" in self.model.__class__.__name__:
            metric_dict.update({
                "train/actor_loss": 0.0,
                "train/critic_loss": 0.0,
                "train/ent_coef": 0.0,
            })
        elif "PPO" in self.model.__class__.__name__:
            metric_dict.update({
                "train/policy_loss": 0.0,
                "train/value_loss": 0.0,
                "train/explained_variance": 0.0,
            })

        # HPARAMS 로깅
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

        if self.verbose >= 1:
            print(f"하이퍼파라미터 로깅 완료: {len(hparam_dict)}개 파라미터, {len(metric_dict)}개 메트릭")

    def _on_step(self) -> bool:
        return True

class VehiclePerformanceCallback(BaseCallback):
    """
    개별 차량 성능을 추적하고 로깅하는 콜백
    """
    def __init__(self, log_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.vehicle_rewards = {}
        self.vehicle_steps = {}
        self.vehicle_collisions = {}
        self.vehicle_goals_reached = {}

    def _on_step(self) -> bool:
        # 개별 차량 데이터 수집 (가능한 경우)
        if hasattr(self.model, 'rl_env') and self.n_calls % self.log_freq == 0:
            rl_env = self.model.rl_env

            # 현재 에피소드 정보가 있는 경우
            if hasattr(rl_env, 'episode_count') and hasattr(rl_env, 'num_vehicles'):
                num_vehicles = rl_env.num_vehicles

                # 활성 차량 수 로깅
                active_count = sum(rl_env.active_agents) if hasattr(rl_env, 'active_agents') else num_vehicles
                self.logger.record("vehicles/active_count", active_count)
                self.logger.record("vehicles/total_count", num_vehicles)
                self.logger.record("vehicles/active_ratio", active_count / num_vehicles if num_vehicles > 0 else 0)

                # 개별 차량 상태 로깅 (처음 3대만)
                for i in range(min(3, num_vehicles)):
                    is_active = rl_env.active_agents[i] if hasattr(rl_env, 'active_agents') else True
                    self.logger.record(f"vehicles/vehicle_{i}/active", int(is_active))

                if self.verbose >= 1 and self.n_calls % (self.log_freq * 10) == 0:
                    print(f"차량 성능 추적: {active_count}/{num_vehicles} 차량 활성")

        return True

class CustomMonitorCallback(BaseCallback):
    """
    Monitor.csv와 동일한 형태로 에피소드 데이터를 직접 기록하는 콜백
    """
    def __init__(self, log_dir: str, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.monitor_file = os.path.join(log_dir, "monitor.csv")
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.start_time = time.time()

        # CSV 헤더 작성
        os.makedirs(log_dir, exist_ok=True)
        with open(self.monitor_file, 'w') as f:
            f.write(f'#{{\"t_start\": {self.start_time}, \"env_id\": \"BasicRLDrivingEnv\"}}\n')
            f.write('r,l,t\n')

    def _on_step(self) -> bool:
        # 에피소드 완료 시 데이터 기록
        if hasattr(self.model, 'rl_env'):
            rl_env = self.model.rl_env

            # 새 에피소드가 시작되었는지 확인 (에피소드 카운터 변화)
            if hasattr(rl_env, 'episode_count'):
                current_episode = rl_env.episode_count

                # 이전 에피소드 데이터가 있다면 기록
                if hasattr(self, 'prev_episode_count') and current_episode > self.prev_episode_count:
                    # 이전 에피소드 정보 가져오기
                    if hasattr(self.model, 'total_reward') and hasattr(self.model, 'episode_steps'):
                        reward = self.model.total_reward
                        length = self.model.episode_steps
                        elapsed_time = time.time() - self.start_time

                        # CSV에 기록
                        with open(self.monitor_file, 'a') as f:
                            f.write(f'{reward},{length},{elapsed_time}\n')

                        if self.verbose >= 1:
                            print(f"Custom Monitor 기록: Episode {self.prev_episode_count}, Reward: {reward:.2f}, Length: {length}")

                self.prev_episode_count = current_episode

        return True

class EvaluationCallback(BaseCallback):
    """
    학습 중 주기적으로 모델 성능을 평가하는 콜백
    """
    def __init__(self, eval_freq: int = 10000, n_eval_episodes: int = 3, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and hasattr(self.model, 'rl_env'):
            try:
                rl_env = self.model.rl_env

                # 간단한 평가 수행
                episode_rewards = []
                episode_lengths = []

                for episode in range(self.n_eval_episodes):
                    obs, _ = rl_env.reset()
                    done = False
                    episode_reward = 0
                    episode_length = 0

                    while not done and episode_length < 500:  # 최대 500 스텝
                        # 모델로 행동 예측 (deterministic)
                        if hasattr(self.model, 'predict'):
                            actions = np.zeros((rl_env.num_vehicles, 3))
                            action, _ = self.model.predict(obs[0], deterministic=True)
                            actions[0] = action
                        else:
                            actions = np.zeros((rl_env.num_vehicles, 3))

                        obs, reward, done, _, _ = rl_env.step(actions)
                        episode_reward += reward
                        episode_length += 1

                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                # 평가 결과 로깅
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_length = np.mean(episode_lengths)

                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)
                self.logger.record("eval/mean_episode_length", mean_length)

                # 최고 성능 추적
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.logger.record("eval/best_mean_reward", self.best_mean_reward)

                    if self.verbose >= 1:
                        print(f"새로운 최고 평가 성능: {mean_reward:.2f}")

                if self.verbose >= 1:
                    print(f"평가 완료 - 평균 보상: {mean_reward:.2f} ± {std_reward:.2f}, 평균 길이: {mean_length:.1f}")

            except Exception as e:
                if self.verbose >= 1:
                    print(f"평가 실패: {e}")

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
