# -*- coding: utf-8 -*-
"""
test_local_2.py  (BENCHMARK - 2x2 COMBINATIONS)
- ✅ 통합 전략 완전 구현: Frenet 좌표계, BasicRLDrivingEnv, 관측/행동/보상 동일 적용
- 🔬 벤치마크 전용: 2 Planners (Lattice, TrajectoryRollout) x 2 Controllers (Stanley, EnhancedP)
- 각 조합당 100회 테스트하여 성능 비교

========================================
📘 빠른 실행 가이드
========================================

1️⃣ 기본 실행 (벤치마크 100회 자동):
   python test_local_2.py

2️⃣ 벤치마크 N회 실행:
   python test_local_2.py benchmark 50

3️⃣ Pygame 시각화 모드: default trajectoryrollout + stanley
   python test_local_2.py visual

자세한 내용은 파일 하단 __main__ 섹션 참고 (1457번째 줄)
========================================
"""

import os
import sys
import math
import numpy as np
import pygame
from collections import deque

# ============================
# FAST MODE 스위치/파라미터
# ============================
FAST_MODE = False  # False: Pygame 시각화 모드 (경로 확인 가능)

# 렌더/오버레이 (FPS 40 목표 - 경로 그리기 최소화)
SHOW_LOCAL_PATH = False if FAST_MODE else True
RENDER_EVERY     = 1  # 매 프레임 렌더 (부드러운 시각화)
DRAW_EVERY       = RENDER_EVERY
PATH_COLOR       = (40, 220, 120)
PATH_PT_COLOR    = (255, 255, 255)
PATH_WIDTH       = 1  # 얇은 선으로 빠르게
PATH_PT_STEP     = 12  # 포인트 간격 최적화 (최적 성능 설정)

# 로그 (디버그 출력 완전 제거로 FPS 최대화)
DEBUG_PRINT_LIMIT = 0  # 모든 디버그 출력 비활성화

# 플래너 설정 (최고 성능 조합: TrajectoryRollout + Stanley)
USE_LATTICE_PLANNER = False  # False: TrajectoryRollout (최고 성능)

# TrajectoryRollout 설정 (부드러운 경로 우선)
PLANNER_HORIZONS = (1.2, 2.8)  # 원래 설정
PLANNER_SPEEDS   = (9.0, 12.0, 15.0)  # 원래 설정
PLANNER_KAPPAS   = (-0.10, -0.07, -0.04, 0.0, 0.04, 0.07, 0.10)  # 부드러운 경로만! (±0.15 -> ±0.10)
PLANNER_DT_MIN   = 0.09
CENTER_SAMPLE_STEP = 10

# Lattice Planner 설정 (급커브 특화 + 초기 방향 정렬 강화) - 최고 성능 설정
LATTICE_TIME_HORIZON = 1.5  # 예측 시간 [s] (1.6 -> 1.5, 더 빠른 반응)
LATTICE_DT = 0.15  # 샘플링 간격 [s] (0.16 -> 0.15, 더 정밀)
LATTICE_LATERAL_SAMPLES = 9  # Lateral offset 샘플 수 (7 -> 9, 최대 다양성)
LATTICE_MAX_LATERAL_OFFSET = 5.0  # 최대 lateral offset [m] (4.5 -> 5.0, 최대 회피)
LATTICE_MAX_CURVATURE = 0.75  # 최대 곡률 [1/m] (0.65 -> 0.75, 극한 회전)

# 장애물 관련
DISABLE_OBS_REPULSION = True if FAST_MODE else False   # 장애물 반발 조향 비활성화(연산↓)
SPEED_BRAKE_WITH_OBS  = True if not FAST_MODE else True  # 속도만 안전 제동은 유지

# 도로 참조 lookahead (빠른 반응을 위해 단축)
LOOKAHEAD_M_FOR_REF = 12.0  # 원래 설정 복원  # 적당한 거리로 빠른 경로 변경

# 캐싱 전략 (반응성 최우선)
PATH_CACHE_STEPS = 1  # 매 스텝마다 경로 재계획 (빠른 반응)

# ============================
# sys.path 자동 추가
# ============================
def ensure_src_on_sys_path():
    start = os.path.abspath(os.path.dirname(__file__))
    cur = start
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "src")):
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    cur = os.path.abspath(os.getcwd())
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "src")):
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

ensure_src_on_sys_path()

# 시뮬레이터 환경
from src.env.env_rl import BasicRLDrivingEnv

# Lattice Planner & Controllers (test.py에서 import)
try:
    from src.model.test import LatticePlanner, PathSegment, StanleyController as TestStanleyController
    LATTICE_PLANNER_AVAILABLE = True
    STANLEY_CONTROLLER_AVAILABLE = True
except ImportError:
    LATTICE_PLANNER_AVAILABLE = False
    STANLEY_CONTROLLER_AVAILABLE = False
    print("[Warning] LatticePlanner/Stanley not available, using fallback")

# --- 안전 패치: VehicleState.encoding_angle(None) 방어 (외부 파일 수정 없이) ---
try:
    from src.model.vehicle import VehicleState
    import math as _math
    def _safe_encoding_angle(self, angle):
        if angle is None:
            angle = 0.0
        try:
            angle = float(angle)
        except Exception:
            angle = 0.0
        if _math.isnan(angle) or _math.isinf(angle):
            angle = 0.0
        return _math.cos(angle), _math.sin(angle)
    if hasattr(VehicleState, "encoding_angle"):
        VehicleState.encoding_angle = _safe_encoding_angle
        if DEBUG_PRINT_LIMIT > 0:
            print("[Patch] VehicleState.encoding_angle -> safe")
except Exception as e:
    if DEBUG_PRINT_LIMIT > 0:
        print("[Patch] encoding_angle monkey-patch failed:", repr(e))

# ============================
# 공통 유틸
# ============================
def wrap_to_pi(ang): return (ang + math.pi) % (2 * math.pi) - math.pi
def clamp(v, lo, hi): return lo if v < lo else (hi if v > hi else v)

def _log_once(state, key, msg, limit=5):
    if DEBUG_PRINT_LIMIT <= 0: return
    c = state.get(key, 0)
    if c < min(limit, DEBUG_PRINT_LIMIT):
        print(msg)
        state[key] = c + 1

def _log_every_n(counter, n, msg):
    if DEBUG_PRINT_LIMIT <= 0: return
    counter[0] += 1
    if counter[0] % n == 0: print(msg)

# ============================
# 관측값 파싱 (통합 전략: RL과 동일한 관측 공간 사용)
# ============================
def parse_observation(obs):
    """
    BasicRLDrivingEnv의 23차원 관측값을 파싱
    
    Observation structure (23-dim):
    [0]: progress (normalized)
    [1]: cos(yaw)
    [2]: sin(yaw)
    [3]: vel_long (normalized to max_speed=65)
    [4]: vel_lat (normalized to max_vel_lat=23)
    [5]: cos(goal_yaw_diff)
    [6]: sin(goal_yaw_diff)
    [7]: frenet_d (normalized to road_width/2=4)
    [8]: cos(heading_error)
    [9]: sin(heading_error)
    [10:23]: lidar_data (13 rays, normalized to max_range=30)
    
    Returns:
        dict: 파싱된 관측 정보
    """
    return {
        'progress': float(obs[0]),
        'yaw': math.atan2(obs[2], obs[1]),  # cos, sin -> angle
        'vel_long': float(obs[3]) * 65.0,    # denormalize
        'vel_lat': float(obs[4]) * 23.0,     # denormalize
        'goal_yaw_diff': math.atan2(obs[6], obs[5]),
        'frenet_d': float(obs[7]) * 4.0,     # denormalize
        'heading_error': math.atan2(obs[9], obs[8]),
        'lidar_data': obs[10:23] * 30.0,     # denormalize
        'raw_obs': obs
    }

# ============================
# Signed curvature helper
# ============================
def _signed_kappa_from_segment(seg, s, mag_hint=None, eps=0.4):
    try:
        s0 = max(0.0, s - eps)
        s1 = s + eps
        _, _, yaw0 = seg.evaluate_at_arc_length(s0)
        _, _, yaw1 = seg.evaluate_at_arc_length(s1)
        dpsi = wrap_to_pi(yaw1 - yaw0)
        k_signed = dpsi / max(1e-6, (s1 - s0))
        if mag_hint is not None:
            k_signed = math.copysign(abs(float(mag_hint)), k_signed)
        return float(k_signed)
    except Exception:
        return 0.0

# ============================
# Trajectory Rollout Local Planner (경량화 + 캐싱)
# ============================
class TrajectoryRolloutPlanner:
    def __init__(self, obstacle_api, road_api=None,
                 horizons=(1.2, 2.0), speeds=(4.0, 7.0, 10.0),
                 kappas=(-0.12, -0.08, -0.05, 0.0, 0.05, 0.08, 0.12),
                 wheelbase=2.6, dt_sim=0.06,
                 clearance_weight=2.0, smooth_weight=0.5,
                 heading_weight=1.0, kappa_align_weight=0.8,
                 progress_weight=0.4, min_clear=0.7,
                 center_weight=0.8, edge_weight=0.8, edge_ratio_warn=0.6,
                 debug=False):
        self.obs_api = obstacle_api
        self.road_api = road_api
        self.horizons = tuple(float(T) for T in horizons)
        self.speeds   = tuple(float(v) for v in speeds)
        self.kappas   = tuple(float(k) for k in kappas)
        self.L = float(wheelbase)
        self.dt = float(dt_sim)

        self.w_clear = float(clearance_weight)
        self.w_smooth = float(smooth_weight)
        self.w_head = float(heading_weight)
        self.w_kappa = float(kappa_align_weight)
        self.w_prog = float(progress_weight)

        self.w_center = float(center_weight)
        self.w_edge = float(edge_weight)
        self.edge_ratio_warn = float(edge_ratio_warn)

        self.min_clear = float(min_clear)
        self.debug = bool(debug)

        self._dbg = {"rollout": 0}
        self.last_choice = None
        self.last_goal = None
        self._center_sample_step = CENTER_SAMPLE_STEP
        
        # 경로 캐싱 (성능 향상)
        self._cached_path = None
        self._cache_counter = 0
        self._cache_lifetime = PATH_CACHE_STEPS

    @staticmethod
    def _nearest_on_poly(pose, poly):
        px, py, _ = pose
        best_pt, best_d2, best_i = None, 1e18, 0
        for i, (x, y, yaw, k) in enumerate(poly):
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best_d2:
                best_d2, best_pt, best_i = d2, (x, y, yaw, k), i
        return best_pt, best_d2, best_i

    def _sample_obstacles(self):
        try:
            arr = self.obs_api.get_all_outer_circles()
            return list(arr or [])
        except Exception:
            return []

    def _simulate(self, start_pose, v, kappa, T):
        x, y, yaw = map(float, start_pose)
        pts, t, dt = [], 0.0, self.dt
        while t <= T + 1e-6:
            pts.append((x, y, yaw, float(kappa)))
            yaw += v * kappa * dt
            x += math.cos(yaw) * v * dt
            y += math.sin(yaw) * v * dt
            t += dt
        return pts

    def _clearance_cost(self, path, obstacles):
        if not obstacles: return 0.0
        min_d, hit = 1e9, False
        for (x, y, _, _) in path[::2]:
            for (ox, oy, r) in obstacles:
                d = math.hypot(x - ox, y - oy) - r
                if d < 0.0: hit = True; break
                min_d = min(min_d, d)
            if hit: break
        if hit: return 1e5
        if min_d < self.min_clear:
            return 1200.0 * (self.min_clear - max(min_d, 0.0))
        if min_d < 2.0:
            return 1.5 * (2.0 - min_d)
        return 1.0 / (1.0 + min_d)

    def _smooth_cost(self, path):
        ks = [p[3] for p in path]
        if len(ks) < 2: return 0.0
        dks = [abs(ks[i] - ks[i-1]) for i in range(1, len(ks))]
        return float(np.mean(dks))

    def _heading_cost(self, path, desired_yaw):
        return abs(wrap_to_pi(path[-1][2] - desired_yaw))

    def _kappa_align_cost(self, path, desired_kappa):
        if desired_kappa is None: return 0.0
        return abs(path[-1][3] - desired_kappa)

    def _progress_reward(self, path, start_pose):
        x0, y0, yaw0 = start_pose
        xe, ye = path[-1][0], path[-1][1]
        forward = (xe - x0) * math.cos(yaw0) + (ye - y0) * math.sin(yaw0)
        return max(0.0, forward)

    def _center_and_edge_cost(self, path):
        if self.road_api is None or not path: return 0.0, 0.0
        try:
            x0, y0, yaw0, _ = path[0]
            info0 = self.road_api.get_vehicle_road_info((x0, y0, yaw0))
            seg = info0.get('closest_segment', None) if info0 else None
            if seg is None: return 0.0, 0.0
            half_w = max(0.5, float(getattr(seg, 'width', 6.0)) * 0.5)
        except Exception:
            return 0.0, 0.0

        ratios, edge_terms = [], []
        step = max(1, self._center_sample_step)
        for i in range(0, len(path), step):
            x, y, _, _ = path[i]
            try:
                _, dist_edge, _ = seg.project_point((x, y))
            except Exception:
                continue
            ratio = min(1.0, abs(dist_edge) / half_w)
            ratios.append(ratio * ratio)
            if ratio > self.edge_ratio_warn:
                edge_terms.append(math.exp(5.5 * (ratio - self.edge_ratio_warn)) - 1.0)
            else:
                edge_terms.append(0.0)
        if not ratios: return 0.0, 0.0
        return float(np.mean(ratios)), float(np.mean(edge_terms))

    def get_local_path(self, vehicle_pose, desired_yaw=None, desired_kappa=None, goal_xy=None):
        if desired_yaw is None: desired_yaw = vehicle_pose[2]
        self.last_goal = goal_xy

        # 캐싱 전략: 장애물이 없고 캐시가 유효하면 재사용
        if self._cached_path is not None and self._cache_counter < self._cache_lifetime:
            obstacles = self._sample_obstacles()
            if len(obstacles) == 0:  # 장애물 없으면 캐시 사용
                self._cache_counter += 1
                return self._cached_path
        
        # 캐시 만료 또는 장애물 존재시 재계산
        self._cache_counter = 0

        obstacles = self._sample_obstacles()
        kappas = list(self.kappas)
        if desired_kappa is not None:
            for dk in (-0.06, -0.03, 0.0, 0.03, 0.06):
                cand = desired_kappa + dk
                if min(self.kappas) <= cand <= max(self.kappas):
                    kappas.append(cand)
        kappas = sorted(set(round(k, 3) for k in kappas))

        best_path, bestJ, best_tag, best_costs = None, 1e18, None, None
        for T in self.horizons:
            for v in self.speeds:
                for kappa in kappas:
                    path = self._simulate(vehicle_pose, v, kappa, T)
                    J_clear = self._clearance_cost(path, obstacles)
                    J_smooth = self._smooth_cost(path)
                    J_head  = self._heading_cost(path, desired_yaw)
                    J_kappa = self._kappa_align_cost(path, desired_kappa)

                    if goal_xy is not None:
                        xe, ye = path[-1][0], path[-1][1]
                        J_goal = math.hypot(xe - goal_xy[0], ye - goal_xy[1])
                    else:
                        J_goal = 0.0

                    R_prog = self._progress_reward(path, vehicle_pose)
                    speed_penalty = 0.0   # 완전 제거(속도↑)

                    C_center, C_edge = self._center_and_edge_cost(path)

                    J = ( self.w_clear * J_clear
                        + self.w_smooth * J_smooth
                        + self.w_head  * J_head
                        + self.w_kappa * J_kappa
                        + 1.0 * J_goal
                        + self.w_center * C_center
                        + self.w_edge   * C_edge
                        - self.w_prog * R_prog
                        + speed_penalty )

                    if J < bestJ:
                        bestJ, best_path, best_tag = J, path, (T, v, kappa)
                        best_costs = (J_clear, J_smooth, J_head, J_kappa, J_goal, C_center, C_edge, R_prog, speed_penalty)

        if best_path and self.debug and DEBUG_PRINT_LIMIT > 0 and self._dbg["rollout"] < DEBUG_PRINT_LIMIT:
            T, v, k = best_tag
            (Jc, Js, Jh, Jk, Jg, Cc, Ce, Rp, Sp) = best_costs
            print(f"[TR] chosen | len={len(best_path)}, J={bestJ:.2f}, T={T:.2f}s, v={v:.1f}, k={k:.3f} | "
                  f"clear={Jc:.2f}, smooth={Js:.2f}, head={Jh:.2f}, kappa={Jk:.2f}, goal={Jg:.2f}, "
                  f"center={Cc:.2f}, edge={Ce:.2f}, -prog={-self.w_prog*Rp:.2f}, spdpen={Sp:.2f}")
            self._dbg["rollout"] += 1

        if not best_path:
            x, y, yaw = vehicle_pose
            fallback = [(x, y, yaw, 0.0), (x + math.cos(yaw)*2.0, y + math.sin(yaw)*2.0, yaw, 0.0)]
            self._cached_path = fallback
            return fallback
        
        # 경로를 캐시에 저장
        self._cached_path = best_path
        return best_path

# ============================
# 컨트롤러 베이스/구현 (통합 전략 + 성능 최적화)
# ============================
class BaseController:
    def __init__(self, env, action_low, action_high):
        self.env = env
        self.core = env.env
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = self.action_low.shape[0]

        self.max_steer_rad = math.radians(self._safe_get(self.core.config, ['vehicle', 'max_steer'], 30.0))
        self.wheelbase_m   = float(self._safe_get(self.core.config, ['vehicle', 'wheelbase'], 2.6))
        self.dt = float(self._safe_get(self.core.config, ['simulation', 'dt'], 1.0/60.0))
        
        # 통합 전략: 관측값 사용 모드 (기본적으로 직접 접근 모드 유지, 필요시 관측값 모드 활성화)
        self.use_observations = False  # True로 설정하면 observations를 파싱하여 사용
        self.last_observations = None

        # 속도 제어 (안전 우선 - 중심선 유지 집중)
        self.k_speed_p = 1.0  # 부드러운 속도 제어
        self.v_min, self.v_max = 3.5, 7.0  # 12.6~25.2 km/h (더 낮은 속도로 안전 주행)
        self.brake_on_negative_a = True
        self.a_brake_safe = 7.5  # 강한 제동

        # 장애물 옵션
        self.enable_obstacle_repulsion = (not DISABLE_OBS_REPULSION)
        self.obs_check_radius = 10.0
        self.obs_fov_deg = 110.0
        self.k_obs = 0.7
        self.v_obs_min = 3.0

        # 액션 포맷
        if self.action_dim == 2:
            self.format = "AS"; self.i_accel, self.i_steer = 0, 1
        elif self.action_dim == 3:
            self.format = "TBS"; self.i_throttle, self.i_brake, self.i_steer = 0, 1, 2
        else:
            self.format = "UNKNOWN"

        # 조향 후처리 (원래 설정 복원: Reward 1709.08)
        self.use_steer_rate_limit = True
        self.steer_rate_limit = math.radians(180.0)  # 원래 설정
        self.use_steer_lpf = True
        self.steer_lpf_alpha = 0.40  # 원래 설정
        self._steer_cmd_prev = 0.0

        # 플래너 선택: Lattice Planner (급커브 특화) 또는 TrajectoryRollout (폴백)
        if USE_LATTICE_PLANNER and LATTICE_PLANNER_AVAILABLE:
            # Lattice Planner: Quintic polynomial 기반, 급커브에 강함
            try:
                road_width = self._safe_get(self.core.config, ['simulation', 'path_planning', 'road_width'], 8.0)
                self.planner = LatticePlanner(
                    time_horizon=LATTICE_TIME_HORIZON,
                    dt=LATTICE_DT,
                    lateral_samples=LATTICE_LATERAL_SAMPLES,
                    road_width=road_width,
                    max_lateral_offset=LATTICE_MAX_LATERAL_OFFSET,
                    max_curvature=LATTICE_MAX_CURVATURE
                )
                self.planner_type = "Lattice"
                if DEBUG_PRINT_LIMIT > 0:
                    print(f"[Planner] Lattice Planner 초기화 완료 (급커브 특화)")
            except Exception as e:
                print(f"[Warning] Lattice Planner 초기화 실패: {e}, TrajectoryRollout 사용")
                self.planner_type = "TrajectoryRollout"
                planner_dt = max(self.dt, PLANNER_DT_MIN)
                self.planner = TrajectoryRolloutPlanner(
                    obstacle_api=self.core.obstacle_manager,
                    road_api=self.core.road_manager,
                    horizons=PLANNER_HORIZONS,
                    speeds=PLANNER_SPEEDS,
                    kappas=PLANNER_KAPPAS,
                    wheelbase=self.wheelbase_m,
                    dt_sim=planner_dt,
                    # 부드러운 경로 최우선 (급격한 커브 억제)
                    clearance_weight=5.0,    # 벽 회피 (Controller 안전 장치 보조)
                    smooth_weight=5.0,       # 부드러운 경로 최우선! (2.0 -> 5.0, 급격한 커브 억제)
                    heading_weight=5.0,      # 방향 유지 강화 (3.0 -> 5.0, 급격한 회전 억제)
                    kappa_align_weight=3.0,  # 곡률 정렬 (2.0 -> 3.0, 도로 곡률 따름)
                    progress_weight=1.5,     # 진행 감소 (2.0 -> 1.5, 부드러움 우선)
                    min_clear=1.0,           # 벽과 최소 1.0m 거리
                    center_weight=10.0,      # 중심선 유지 큰 보상!
                    edge_weight=30.0,        # 도로 가장자리 큰 손실!
                    edge_ratio_warn=0.45,    # 벽 경고
                    debug=False
                )
        else:
            # TrajectoryRollout Planner (폴백)
            self.planner_type = "TrajectoryRollout"
            planner_dt = max(self.dt, PLANNER_DT_MIN)
            self.planner = TrajectoryRolloutPlanner(
                obstacle_api=self.core.obstacle_manager,
                road_api=self.core.road_manager,
                horizons=PLANNER_HORIZONS,
                speeds=PLANNER_SPEEDS,
                kappas=PLANNER_KAPPAS,
                wheelbase=self.wheelbase_m,
                dt_sim=planner_dt,
                # 부드러운 경로 최우선 (급격한 커브 억제)
                clearance_weight=5.0,    # 벽 회피 (Controller 안전 장치 보조)
                smooth_weight=5.0,       # 부드러운 경로 최우선! (2.0 -> 5.0, 급격한 커브 억제)
                heading_weight=5.0,      # 방향 유지 강화 (3.0 -> 5.0, 급격한 회전 억제)
                kappa_align_weight=3.0,  # 곡률 정렬 (2.0 -> 3.0, 도로 곡률 따름)
                progress_weight=1.5,     # 진행 감소 (2.0 -> 1.5, 부드러움 우선)
                min_clear=1.0,           # 벽과 최소 1.0m 거리
                center_weight=10.0,      # 중심선 유지 큰 보상!
                edge_weight=30.0,        # 도로 가장자리 큰 손실!
                edge_ratio_warn=0.45,    # 벽 경고
                debug=False
            )

        # 필터 (원래 설정 복원: Reward 1709.08)
        self.e_alpha, self.psi_alpha, self.kappa_alpha = 0.6, 0.6, 0.7  # 원래 설정
        self._e_f = self._psi_f = self._kappa_f = 0.0

        self._dbg_state = {}
        self._dbg_counter = [0]
        # 곡률 피드포워드 게인 (최대 추종력)
        self.curvature_ff_gain = 1.2  # 강력한 곡률 추종 (0.80 -> 1.2)
        self.curvature_ff_sign = +1.0

    def _safe_get(self, d, keys, default=None):
        cur = d
        try:
            for k in keys: cur = cur[k]
            return cur
        except Exception:
            return default

    def _normalize_steer_to_unit(self, steer_rad):
        return clamp(steer_rad / (self.max_steer_rad + 1e-6), -1.0, 1.0)

    def _desired_pose_from_road(self, vehicle_pose):
        yaw_des = vehicle_pose[2]
        kappa_des = None
        goal_xy = None
        rm = self.core.road_manager
        info = None
        try:
            info = rm.get_vehicle_road_info(vehicle_pose)
        except Exception as e:
            _log_once(self._dbg_state, "road_info_err", f"[DBG] road_info error: {repr(e)}")

        if info and info.get('closest_segment', None) is not None:
            seg = info['closest_segment']
            fr = info.get('frenet_state', None)
            s0 = 0.0
            if fr is not None:
                try: s0 = float(fr.s)
                except Exception: s0 = 0.0
            else:
                try: _, _, s0 = seg.project_point((vehicle_pose[0], vehicle_pose[1]))
                except Exception: s0 = 0.0

            L = max(LOOKAHEAD_M_FOR_REF, 4.0)
            s_ref = s0 + L
            try:
                x_ref, y_ref, yaw_ref = seg.evaluate_at_arc_length(s_ref)
                yaw_des = float(yaw_ref)
                goal_xy = (float(x_ref), float(y_ref))
            except Exception:
                try:
                    yaw_des = float(info.get('road_yaw', yaw_des))
                    x0, y0 = info.get('road_center_point', (vehicle_pose[0], vehicle_pose[1]))
                    goal_xy = (float(x0) + math.cos(yaw_des)*L, float(y0) + math.sin(yaw_des)*L)
                except Exception:
                    goal_xy = (vehicle_pose[0] + math.cos(yaw_des)*L,
                               vehicle_pose[1] + math.sin(yaw_des)*L)
            try:
                k_mag_hint = None
                try: k_mag_hint = float(seg.get_curvature_at_s(s_ref))
                except Exception: pass
                kappa_des = _signed_kappa_from_segment(seg, s_ref, mag_hint=k_mag_hint, eps=0.4)
            except Exception:
                kappa_des = 0.0
        else:
            try:
                upd = rm.get_vehicle_update_data(vehicle_pose)
                for v in upd:
                    try:
                        f = float(v)
                        if -math.pi - 1e-3 <= f <= math.pi + 1e-3:
                            yaw_des = f; break
                    except Exception:
                        pass
            except Exception:
                pass

        if goal_xy is None:
            Ltmp = 8.0
            goal_xy = (vehicle_pose[0] + math.cos(yaw_des)*Ltmp,
                       vehicle_pose[1] + math.sin(yaw_des)*Ltmp)
        return yaw_des, kappa_des, goal_xy

    def speed_ref_from_update(self, vehicle_pose):
        # 기본 목표 속도 (도로 정보 기반)
        v_base = self.v_max
        try:
            upd = self.core.road_manager.get_vehicle_update_data(vehicle_pose)
            cand = []
            for v in upd:
                try:
                    f = float(v)
                    if 1.0 < f < 50.0: cand.append(f)
                except Exception:
                    pass
            if cand: v_base = clamp(min(cand), self.v_min, self.v_max)
        except Exception:
            pass
        
        # 경로 곡률에 따라 속도 조정 (원래 설정 복원: Reward 1709.08)
        if hasattr(self, '_viz_last_path') and self._viz_last_path:
            try:
                # 경로의 평균 곡률 계산
                kappas = [abs(p[3]) for p in self._viz_last_path[:min(10, len(self._viz_last_path))]]
                if kappas:
                    avg_kappa = sum(kappas) / len(kappas)
                    # 곡률이 클수록 속도 감소
                    if avg_kappa > 0.20:  # 극심한 급커브
                        v_base = clamp(self.v_min + 0.5, self.v_min, self.v_max)
                    elif avg_kappa > 0.12:  # 급커브
                        v_base = clamp(self.v_min + 1.5, self.v_min, self.v_max)
                    elif avg_kappa > 0.06:  # 중간 커브
                        v_base = clamp(self.v_min + 2.5, self.v_min, self.v_max)
                    # 직선/완만한 커브는 최대 속도
            except Exception:
                pass
        
        return v_base

    def speed_with_obstacle_brake(self, v_ref, st):
        if not SPEED_BRAKE_WITH_OBS: return v_ref
        try:
            oxyr_list = self.core.obstacle_manager.get_all_outer_circles() or []
        except Exception:
            oxyr_list = []
        if not oxyr_list: return v_ref
        vx, vy, vyaw = st.x, st.y, st.yaw
        cos_y, sin_y = math.cos(vyaw), math.sin(vyaw)
        fov_cos = math.cos(math.radians(110.0) / 2.0)
        v_safe_lim = v_ref
        for (ox, oy, r) in oxyr_list:
            dx, dy = ox - vx, oy - vy
            dist_edge = max(0.0, math.hypot(dx, dy) - r)
            if dist_edge > 10.0: continue
            front = dx * cos_y + dy * sin_y
            if front <= 0: continue
            dir_norm = math.hypot(dx, dy) + 1e-6
            cos_angle = (dx * cos_y + dy * sin_y) / dir_norm
            if cos_angle < fov_cos: continue
            v_safe = math.sqrt(max(0.0, 2.0 * self.a_brake_safe * dist_edge))
            v_safe_lim = min(v_safe_lim, max(self.v_obs_min, v_safe))
        return v_safe_lim

    def steer_obstacle_repulsion(self, st):
        if not self.enable_obstacle_repulsion: return 0.0
        try:
            oxyr_list = self.core.obstacle_manager.get_all_outer_circles() or []
        except Exception:
            oxyr_list = []
        if not oxyr_list: return 0.0
        vx, vy, vyaw = st.x, st.y, st.yaw
        cos_y, sin_y = math.cos(vyaw), math.sin(vyaw)
        fov_cos = math.cos(math.radians(self.obs_fov_deg) / 2.0)
        steer_obs = 0.0
        for (ox, oy, r) in oxyr_list:
            dx, dy = ox - vx, oy - vy
            dist = math.hypot(dx, dy)
            if dist - r > self.obs_check_radius: continue
            cos_angle = (dx * cos_y + dy * sin_y) / (dist + 1e-9)
            if cos_angle < fov_cos: continue
            side = -dx * sin_y + dy * cos_y
            edge = max(0.3, dist - r)
            steer_obs += 0.7 * math.atan2(side, edge**2)
        return clamp(steer_obs, -self.max_steer_rad*0.6, self.max_steer_rad*0.6)

    def postprocess_steer(self, steer_rad):
        s = steer_rad
        if self.use_steer_rate_limit:
            ds_max = self.steer_rate_limit * self.dt
            s = clamp(s, self._steer_cmd_prev - ds_max, self._steer_cmd_prev + ds_max)
        if self.use_steer_lpf:
            s = (1.0 - self.steer_lpf_alpha) * self._steer_cmd_prev + self.steer_lpf_alpha * s
        s = clamp(s, -self.max_steer_rad, self.max_steer_rad)
        self._steer_cmd_prev = s
        return s

    def ref_from_path(self, vehicle_pose):
        """
        🔄 ROLLBACK: Planner 기반 경로 추종 (가장 안전한 버전)
        - 도로 정보 + Planner 경로 = 커브 대응 가능!
        """
        desired_yaw, desired_kappa, goal_xy = self._desired_pose_from_road(vehicle_pose)
        
        # Planner 타입에 따라 다른 방식으로 경로 생성
        if hasattr(self, 'planner_type') and self.planner_type == "Lattice":
            path = self._get_lattice_path(vehicle_pose, desired_yaw)
        else:
            path = self.planner.get_local_path(vehicle_pose, desired_yaw, desired_kappa, goal_xy)
        
        if not path or len(path) < 2:
            x, y, yaw = vehicle_pose
            path = [(x, y, yaw, 0.0), (x + math.cos(yaw)*2.0, y + math.sin(yaw)*2.0, yaw, 0.0)]
        
        self._viz_last_path = path
        (cx, cy, cyaw, ck), _, idx = TrajectoryRolloutPlanner._nearest_on_poly(vehicle_pose, path)
        
        # 경로 추종 (매 스텝 새 경로 계획)
        steps_ahead = max(1, int(max(1.0, LOOKAHEAD_M_FOR_REF) / 1.0))
        look_idx = min(idx + steps_ahead, len(path) - 1)
        
        yaw_ref = path[look_idx][2]
        kappa_now = path[look_idx][3]
        
        # Cross-track error: 현재 가장 가까운 점 기준
        dx, dy = vehicle_pose[0] - cx, vehicle_pose[1] - cy
        e_now = -dx * math.sin(cyaw) + dy * math.cos(cyaw)
        psi_e_now = wrap_to_pi(yaw_ref - vehicle_pose[2])
        
        return psi_e_now, e_now, kappa_now
    
    def _get_lattice_path(self, vehicle_pose, desired_yaw):
        """Lattice Planner에서 경로를 가져와 전역 좌표로 변환"""
        try:
            # Frenet 좌표 가져오기
            rm = self.core.road_manager
            info = rm.get_vehicle_road_info(vehicle_pose)
            
            if not info or not info.get('closest_segment'):
                # Frenet 좌표를 얻을 수 없으면 폴백
                return None
            
            seg = info['closest_segment']
            fr = info.get('frenet_state')
            
            if fr is None:
                # Frenet state가 없으면 투영으로 계산
                _, d, s = seg.project_point((vehicle_pose[0], vehicle_pose[1]))
            else:
                s, d = float(fr.s), float(fr.d)
            
            # 차량 속도 가져오기
            vehicles = self.core.vehicle_manager.get_all_vehicles()
            if not vehicles:
                return None
            
            vel_long = float(vehicles[0].state.vel_long)
            vel_lat = float(vehicles[0].state.vel_lat) if hasattr(vehicles[0].state, 'vel_lat') else 0.0
            
            # Lattice planner로 경로 생성 (Frenet 좌표계)
            candidate_paths = self.planner.generate_candidate_paths(
                current_s=s,
                current_d=d,
                current_d_dot=vel_lat,
                current_speed=max(vel_long, 1.0)
            )
            
            if not candidate_paths:
                return None
            
            # 최적 경로 선택 (cost 기준)
            best_path = min(candidate_paths, key=lambda p: p.cost)
            
            # Frenet -> 전역 좌표 변환
            global_path = []
            for (s_p, d_p, t_p) in best_path.points:
                try:
                    # 도로 세그먼트에서 s 위치의 중심선 좌표 가져오기
                    x_c, y_c, yaw_c = seg.evaluate_at_arc_length(s_p)
                    
                    # Lateral offset 적용 (d 방향으로 이동)
                    x_global = x_c - d_p * math.sin(yaw_c)
                    y_global = y_c + d_p * math.cos(yaw_c)
                    
                    # 곡률 추정 (간단한 방법)
                    kappa = 0.0
                    if len(global_path) > 0:
                        dx = x_global - global_path[-1][0]
                        dy = y_global - global_path[-1][1]
                        ds = math.hypot(dx, dy)
                        if ds > 0.01:
                            dyaw = wrap_to_pi(yaw_c - global_path[-1][2])
                            kappa = dyaw / ds
                    
                    global_path.append((x_global, y_global, yaw_c, kappa))
                except Exception:
                    continue
            
            return global_path if len(global_path) >= 2 else None
            
        except Exception as e:
            if DEBUG_PRINT_LIMIT > 0:
                print(f"[Warning] Lattice path generation failed: {e}")
            return None

    def steer_cmd_core(self, vehicle_pose, v_long): raise NotImplementedError

    def act_single(self, veh, observation=None):
        """
        통합 전략: observation을 옵션으로 받아서 처리
        observation이 주어지고 use_observations=True이면 관측값 우선 사용
        """
        st = veh.state
        pose = (st.x, st.y, st.yaw)
        v_long = float(st.vel_long)
        
        # 관측값 기반 정보 추출 (통합 전략)
        if observation is not None and self.use_observations:
            parsed = parse_observation(observation)
            # 관측값에서 파싱한 정보 활용 가능 (현재는 직접 접근 방식과 병행)
            # v_long = parsed['vel_long']  # 필요시 활성화
            # pose = (st.x, st.y, parsed['yaw'])  # 필요시 활성화
        
        psi_e_now, e_now, kappa_now = self.ref_from_path(pose)

        steer_rad = self.steer_cmd_core_impl(psi_e_now, e_now, kappa_now, v_long)
        steer_rad += self.steer_obstacle_repulsion(st)
        steer_rad = wrap_to_pi(steer_rad)
        steer_rad = self.postprocess_steer(steer_rad)

        v_ref = self.speed_ref_from_update(pose)
        v_ref = self.speed_with_obstacle_brake(v_ref, st)
        # 속도 제어 게인 적용 (더 빠른 가속)
        a_cmd = self.k_speed_p * (v_ref - max(0.0, v_long))

        steer_unit = self._normalize_steer_to_unit(steer_rad)
        if self.format == "AS":
            act = np.array([a_cmd, steer_unit], dtype=np.float32)
        elif self.format == "TBS":
            if self.brake_on_negative_a and a_cmd < 0.0:
                throttle, brake = 0.0, -a_cmd
            else:
                throttle, brake = max(0.0, a_cmd), 0.0
            act = np.array([throttle, brake, steer_unit], dtype=np.float32)
        else:
            act = np.zeros(self.action_dim, dtype=np.float32)
        return np.clip(act, self.action_low, self.action_high)

    def act_batch(self, observations=None):
        """
        통합 전략: observations를 받아서 행동 생성
        observations가 None이면 직접 vehicle state 접근 (기존 방식)
        """
        vehicles = self.core.vehicle_manager.get_all_vehicles()
        acts = np.zeros((len(vehicles), self.action_dim), dtype=np.float32)
        
        if observations is not None and self.use_observations:
            # 통합 전략: 관측값 사용 모드
            self.last_observations = observations
            for i, veh in enumerate(vehicles):
                if i < len(observations):
                    # 관측값과 vehicle을 모두 전달 (하이브리드 모드)
                    acts[i] = self.act_single(veh, observations[i])
                else:
                    acts[i] = self.act_single(veh)
        else:
            # 기존 방식: 직접 vehicle state 접근
            for i, veh in enumerate(vehicles):
                acts[i] = self.act_single(veh)
        return acts

class EnhancedPController(BaseController):
    def __init__(self, env, action_low, action_high):
        super().__init__(env, action_low, action_high)
        # 게인 조정: Lattice Planner에 최적화 (부드러운 경로 추종)
        if hasattr(self, 'planner_type') and self.planner_type == "Lattice":
            self.k_h_base, self.k_d_base = 1.0, 0.28  # 부드러운 추종
        else:
            self.k_h_base, self.k_d_base = 1.1, 0.32  # 강력한 반응성
        self.vh, self.vd = 6.0, 6.0
        
        # 디버깅 (FPS 향상을 위해 비활성화)
        self._debug_steps = 0
        self._debug_print_limit = 0  # 항상 0

    def k_h(self, v): return self.k_h_base * (self.vh / (self.vh + max(0.0, v)))
    def k_d(self, v): return self.k_d_base * (self.vd / (self.vd + max(0.0, v)))

    def steer_cmd_core_impl(self, psi_e, e, kappa, v_long):
        v = max(0.0, float(v_long))
        self._e_f   = (1 - self.e_alpha)   * self._e_f   + self.e_alpha   * e
        self._psi_f = (1 - self.psi_alpha) * self._psi_f + self.psi_alpha * psi_e
        self._kappa_f = (1 - self.kappa_alpha) * self._kappa_f + self.kappa_alpha * kappa
        
        # 조향 계산
        steer_heading = self.k_h(v) * self._psi_f
        steer_lateral = self.k_d(v) * self._e_f
        # 곡률 피드포워드에 게인 적용 (0.5로 약화)
        steer_ff = self.curvature_ff_gain * self.curvature_ff_sign * math.atan(self.wheelbase_m * self._kappa_f)
        steer = steer_heading + steer_lateral + steer_ff
        
        # 디버깅: FPS 향상을 위해 제거 (항상 비활성화)
        
        return clamp(steer, -self.max_steer_rad, self.max_steer_rad)


class StanleyController(BaseController):
    """
    Stanley Controller: 산업 표준 경로 추종 알고리즘
    - Cross-track error와 heading error를 결합
    - 속도에 적응적인 제어
    - Lattice Planner와 최적 조합
    """
    def __init__(self, env, action_low, action_high):
        super().__init__(env, action_low, action_high)
        # Stanley 게인 (중심선 강력 유지)
        self.k_stanley = 12.0  # 중심선 추종 강화 (8.0 -> 12.0)
        self.k_soft = 0.10     # 빠른 반응 (0.15 -> 0.10)
        self.min_speed = 0.5   # Minimum speed for control
        
        # 초기 방향 정렬 모드 (연장)
        self.initial_alignment_steps = 60  # 처음 60 스텝 동안 특별 제어
        self.step_count = 0
        
        # 디버깅
        self._debug_steps = 0
        self._debug_print_limit = 0

    def speed_ref_from_update(self, vehicle_pose):
        """
        점진적 전환: 초기 정렬 모드 → 정상 모드로 부드럽게 전환
        """
        # 부모 클래스의 속도 계산
        v_base = super().speed_ref_from_update(vehicle_pose)
        
        # 점진적 전환: 3단계 속도 제어
        if self.step_count < self.initial_alignment_steps:
            # Phase 1: 초기 정렬 (0~60 스텝)
            progress = self.step_count / self.initial_alignment_steps
            # 부드러운 곡선 전환 (cubic easing)
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            v_initial = 2.5  # 원래 설정 복원
            v_base = v_initial + (v_base - v_initial) * smooth_progress
        elif self.step_count < self.initial_alignment_steps + 30:
            # Phase 2: 전환 단계 (60~90 스텝) - 제어 게인도 점진적으로 완화
            transition_progress = (self.step_count - self.initial_alignment_steps) / 30.0
            # 이 단계에서는 속도는 정상, 제어만 약간 강화
            pass
        # Phase 3: 정상 모드 (90 스텝 이후)
        
        return v_base
    
    def act_single(self, veh, observation=None):
        """
        초기 방향 정렬을 위한 스텝 카운터 증가
        """
        self.step_count += 1
        return super().act_single(veh, observation)

    def steer_cmd_core_impl(self, psi_e, e, kappa, v_long):
        """
        Stanley Control Law + 벽 회피 안전 장치:
        δ = θe + arctan(-k × e / (ks + v)) + safety_correction
        
        where:
        - θe: heading error (psi_e)
        - e: cross-track error (lateral offset)
        - v: vehicle speed
        - safety_correction: 벽 근처에서 중심선 쪽으로 강제 조향
        
        초기 정렬 모드에서는 heading error에 더 큰 가중치 부여
        """
        v = max(self.min_speed, abs(float(v_long)))
        
        # Stanley control law
        # Heading error component
        heading_term = psi_e
        
        # 점진적 전환: heading error 가중치 부드럽게 감소
        if self.step_count < self.initial_alignment_steps:
            # Phase 1: 초기 정렬 (0~60 스텝) - 2.0배 강조
            heading_weight = 2.0
        elif self.step_count < self.initial_alignment_steps + 30:
            # Phase 2: 전환 단계 (60~90 스텝) - 2.0 → 1.0으로 점진 감소
            transition_progress = (self.step_count - self.initial_alignment_steps) / 30.0
            heading_weight = 2.0 - (2.0 - 1.0) * transition_progress
        else:
            # Phase 3: 정상 모드 (90+ 스텝)
            heading_weight = 1.0
        
        # Cross-track error component
        # Note: negative sign because positive e means left of path
        cross_track_term = math.atan(-self.k_stanley * e / (self.k_soft + v))
        
        # Curvature feedforward (optional, improves performance)
        feedforward_term = self.curvature_ff_gain * math.atan(self.wheelbase_m * kappa)
        
        # 🛡️ 벽 회피 안전 장치: 중심선에서 벗어나면 강제로 중심선 쪽으로 조향!
        safety_correction = 0.0
        try:
            # 도로 정보 확인
            vehicle_pose = self.env.vehicles[0].get_pose()
            rm = self.core.road_manager
            info = rm.get_vehicle_road_info(vehicle_pose)
            
            if info and 'lateral_offset' in info and 'lane_width' in info:
                lateral_offset = float(info['lateral_offset'])  # 중심선으로부터의 거리 (+ = 왼쪽, - = 오른쪽)
                lane_width = float(info['lane_width'])  # 도로 전체 폭
                
                # 도로 가장자리까지의 거리 비율 계산
                edge_ratio = abs(lateral_offset) / (lane_width / 2.0) if lane_width > 0 else 0.0
                
                # 벽 근처 경고 (edge_ratio > 0.5: 도로 폭의 50% 이상 벗어남)
                if edge_ratio > 0.5:
                    # 중심선 쪽으로 강제 조향 (lateral_offset의 부호 반대 방향)
                    safety_gain = 10.0  # 극강 보정! (5.0 -> 10.0)
                    safety_correction = -math.copysign(safety_gain * (edge_ratio - 0.5), lateral_offset)
                    # edge_ratio가 1.0(벽)에 가까울수록 더 강한 보정
                # 약간 벗어남 (edge_ratio > 0.3)
                elif edge_ratio > 0.3:
                    # 부드럽게 중심선으로 유도
                    safety_gain = 3.0
                    safety_correction = -math.copysign(safety_gain * (edge_ratio - 0.3), lateral_offset)
        except Exception:
            pass  # 도로 정보 없으면 무시
        
        # Total steering (안전 장치 포함!)
        steer = heading_weight * heading_term + cross_track_term + feedforward_term + safety_correction
        
        return clamp(steer, -self.max_steer_rad, self.max_steer_rad)


# 컨트롤러 선택 (최고 성능 조합: TrajectoryRollout + Stanley)
USE_STANLEY_CONTROLLER = True  # True: Stanley (최고 성능)
SELECTED_CONTROLLER = StanleyController if USE_STANLEY_CONTROLLER else EnhancedPController

# ============================
# 렌더 유틸 (최적화 버전)
# ============================
def _get_draw_surface(env_like):
    """Drawing surface 가져오기"""
    try:
        # BasicRLDrivingEnv -> CarSimulatorEnv -> Renderer 경로
        if hasattr(env_like, 'env') and hasattr(env_like.env, 'renderer'):
            renderer = env_like.env.renderer
            if hasattr(renderer, 'screen'):
                return renderer.screen
        # 직접 접근
        if hasattr(env_like, 'renderer') and hasattr(env_like.renderer, 'screen'):
            return env_like.renderer.screen
    except Exception:
        pass
    return pygame.display.get_surface()

def _get_camera(env_like):
    """Camera 객체 가져오기 (제대로 된 world_to_screen 변환용)"""
    try:
        # BasicRLDrivingEnv -> CarSimulatorEnv -> Camera 경로
        if hasattr(env_like, 'env') and hasattr(env_like.env, 'camera'):
            return env_like.env.camera
        # 직접 접근
        if hasattr(env_like, 'camera'):
            return env_like.camera
    except Exception:
        pass
    return None

def _world_to_screen(env_like, x, y):
    """
    제대로 된 world_to_screen 변환 (Camera 객체 활용)
    성능 최적화: Camera의 변환 메커니즘 사용
    """
    camera = _get_camera(env_like)
    if camera is not None:
        try:
            # Camera의 world_to_screen 메서드 사용
            return camera.world_to_screen(np.array([[x, y]]))[0]
        except Exception:
            pass
    
    # 폴백: 간단한 변환 (Camera가 없는 경우)
    surf = pygame.display.get_surface()
    if surf:
        W, H = surf.get_width(), surf.get_height()
        sx = int(W/2 + x*10.0)
        sy = int(H/2 - y*10.0)
        return sx, sy
    return int(x), int(y)

# 배치 변환 (성능 최적화)
def _world_to_screen_batch(env_like, points):
    """여러 점을 한번에 변환 (성능 향상)"""
    camera = _get_camera(env_like)
    if camera is not None:
        try:
            pts_array = np.array(points)
            return camera.world_to_screen(pts_array)
        except Exception:
            pass
    
    # 폴백: 개별 변환
    return [_world_to_screen(env_like, x, y) for x, y in points]

def draw_local_path(env_like, path, color=PATH_COLOR, pt_color=PATH_PT_COLOR,
                    width=PATH_WIDTH, pt_step=PATH_PT_STEP):
    """경로 그리기 (배치 변환으로 최적화)"""
    if not SHOW_LOCAL_PATH or not path or len(path) < 2: return
    surf = _get_draw_surface(env_like)
    if surf is None: return
    
    # 배치 변환으로 성능 향상
    world_pts = [(x, y) for (x, y, _, _) in path]
    pts = _world_to_screen_batch(env_like, world_pts)
    
    try: 
        pygame.draw.lines(surf, color, False, pts, width)
    except Exception: 
        pass
    
    # 포인트 마커 (더 간격을 넓혀서 그리기)
    step = max(1, pt_step)
    for i in range(0, len(pts), step):
        try: 
            pygame.draw.circle(surf, pt_color, pts[i], 2)
        except Exception: 
            pass

def draw_goal_point(env_like, goal_xy, color=(200, 80, 255)):
    """목표 지점 그리기"""
    if not SHOW_LOCAL_PATH or not goal_xy: return
    surf = _get_draw_surface(env_like)
    if surf is None: return
    
    sx, sy = _world_to_screen(env_like, goal_xy[0], goal_xy[1])
    try:
        pygame.draw.line(surf, color, (sx-6, sy), (sx+6, sy), 2)
        pygame.draw.line(surf, color, (sx, sy-6), (sx, sy+6), 2)
    except Exception: 
        pass

# ============================
# 메인 루프 (통합 전략 + 성능 최적화)
# ============================
def main():
    """
    통합 전략 완전 구현:
    - BasicRLDrivingEnv 사용 ✅
    - Frenet 좌표계 및 road_manager 활용 ✅  
    - 동일한 관측/행동 공간 적용 ✅
    - 동일한 보상 함수 적용 ✅
    """
    env = BasicRLDrivingEnv()
    observations, active_agents = env.reset()

    action_low  = env.action_space.low[0]
    action_high = env.action_space.high[0]

    controller = SELECTED_CONTROLLER(env, action_low, action_high)
    controller.enable_obstacle_repulsion = (not DISABLE_OBS_REPULSION)
    # controller.use_observations = True  # 관측값 사용 모드 활성화 (필요시)

    # HUD 활성화 (속도 표시를 위해)
    env.env.config['visualization']['visualize_hud'] = True
    
    # 초기 렌더링은 FAST_MODE와 관계없이 필수 (pygame 초기화)
    env.render()
    
    if not FAST_MODE:
        env.print_basic_controls()
        print(f"[Controller] {SELECTED_CONTROLLER.__name__}")
        print(f"[Planner] {getattr(controller, 'planner_type', 'Unknown')}")
        print(f"[통합 전략] Frenet 좌표계, BasicRLDrivingEnv, 관측/행동/보상 통합 완료\n")
    else:
        print(f"[FAST_MODE] 렌더링 최소화, 성능 최적화 모드")
        print(f"[Controller] {SELECTED_CONTROLLER.__name__}")
        print(f"[Planner] {getattr(controller, 'planner_type', 'Unknown')}\n")

    clock = pygame.time.Clock()
    done, ep_reward, steps = False, 0.0, 0
    last_render_step = -RENDER_EVERY
    
    print(f"[Info] 시작 - ESC로 종료 가능")
    print(f"[Info] Steps: 0, Reward: 0.00, FPS: 0.0")
    
    # 성능 모니터링 (출력 빈도 조정)
    perf_window = deque(maxlen=60)
    fps_display_interval = 120 if FAST_MODE else 60  # FAST_MODE에서 더 적게 출력

    try:
        while not done:
            step_start = pygame.time.get_ticks()
            
            # 이벤트 최소 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    done = True
            env.handle_keyboard_input()

            # 통합 전략: 관측값을 컨트롤러에 전달
            actions = controller.act_batch(observations)
            
            # 통합 전략: 동일한 step 인터페이스 및 보상 함수 사용
            observations, reward, done, _, info = env.step(actions)
            ep_reward += float(reward)
            steps += 1
            
            # 종료 원인 출력 (항상 표시)
            if done:
                reason = info.get('reason', 'Unknown')
                print(f"\n[Episode End] Reason: {reason}")
                print(f"[Episode End] Steps: {steps}, Reward: {ep_reward:.2f}")

            # 렌더 프레임 스킵 (성능 최적화)
            if steps - last_render_step >= RENDER_EVERY:
                env.render()
                if SHOW_LOCAL_PATH and getattr(controller, "_viz_last_path", None):
                    draw_local_path(env, controller._viz_last_path)
                    draw_goal_point(env, getattr(controller.planner, "last_goal", None))
                
                try: 
                    pygame.display.flip()
                except Exception: 
                    pass
                last_render_step = steps

            # FPS 제한 (렌더링 속도 최적화)
            # FAST_MODE: 무제한 (0), 일반 모드: 30 FPS (안정적인 성능)
            clock.tick(0 if FAST_MODE else 30)
            
            # 성능 모니터링
            step_time = pygame.time.get_ticks() - step_start
            perf_window.append(step_time)
            
            # FPS 및 속도 출력 (FAST_MODE에서도 가끔 출력)
            if steps % fps_display_interval == 0 and len(perf_window) > 0:
                avg_step_time = sum(perf_window) / len(perf_window)
                avg_fps = 1000.0 / avg_step_time if avg_step_time > 0 else 0
                
                # 현재 차량 속도 표시
                try:
                    vehicles = env.env.vehicle_manager.get_all_vehicles()
                    if vehicles:
                        v_kmh = vehicles[0].state.vel_long * 3.6
                        print(f"[Performance] Steps: {steps}, FPS: {avg_fps:.1f}, Speed: {v_kmh:.1f} km/h, Reward: {ep_reward:.2f}")
                    else:
                        print(f"[Performance] Steps: {steps}, FPS: {avg_fps:.1f}, Reward: {ep_reward:.2f}")
                except Exception:
                    print(f"[Performance] Steps: {steps}, FPS: {avg_fps:.1f}, Reward: {ep_reward:.2f}")

        # 종료 메시지 (FAST_MODE에서도 표시)
        print(f"\n[Run End] Steps: {steps}, Total Reward: {ep_reward:.2f}")
        print(f"[Controller] {SELECTED_CONTROLLER.__name__}")
        if len(perf_window) > 0:
            avg_step_time = sum(perf_window) / len(perf_window)
            avg_fps = 1000.0 / avg_step_time if avg_step_time > 0 else 0
            print(f"[Performance] Average FPS: {avg_fps:.1f}")
    finally:
        env.close()

def benchmark(num_episodes=5):
    """여러 에피소드 실행해서 평균 성능 측정"""
    import time
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"[BENCHMARK] 벤치마크 시작: {num_episodes}개 에피소드")
    print(f"   Controller: {SELECTED_CONTROLLER.__name__}")
    print(f"   Planner: {'Lattice' if USE_LATTICE_PLANNER else 'TrajectoryRollout'}")
    print(f"{'='*70}\n")
    
    results = []
    successes = 0  # 성공 카운트 (긍정적 reward)
    
    ep_count = 0
    while ep_count < num_episodes:
        # 10 에피소드마다 중간 통계 출력
        if ep_count > 0 and ep_count % 10 == 0:
            temp_avg_reward = sum(r['reward'] for r in results) / len(results)
            temp_avg_steps = sum(r['steps'] for r in results) / len(results)
            elapsed = time.time() - start_time
            eta = (elapsed / ep_count) * (num_episodes - ep_count)
            print(f"\n[Progress {ep_count}/{num_episodes}] 평균 Reward: {temp_avg_reward:.2f}, 평균 Steps: {temp_avg_steps:.1f}, ETA: {eta:.1f}s")
        
        try:
            env = BasicRLDrivingEnv()
            observations, _ = env.reset()
            
            action_low = env.action_space.low[0]
            action_high = env.action_space.high[0]
            controller = SELECTED_CONTROLLER(env, action_low, action_high)
            
            env.env.config['visualization']['visualize_hud'] = False
            
            done, ep_reward, steps = False, 0.0, 0
            
            try:
                while not done and steps < 500:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            break
                    
                    actions = controller.act_batch(observations)
                    observations, reward, done, _, info = env.step(actions)
                    ep_reward += float(reward)
                    steps += 1
                    
            finally:
                env.close()
            
            if ep_reward > 0:
                successes += 1
            
            results.append({
                'steps': steps,
                'reward': ep_reward,
                'reason': info.get('reason', 'Unknown') if done else 'MaxSteps'
            })
            
            # 간결한 출력 (매 에피소드)
            status = "OK" if ep_reward > 0 else "FAIL"
            print(f"[Ep {ep_count+1:3d}] {status:4s} Steps: {steps:3d}, Reward: {ep_reward:7.2f}")
            
            ep_count += 1
            
        except RuntimeError as e:
            if "RRT path planning failed" in str(e):
                print(f"[Ep {ep_count+1:3d}] SKIP (RRT failed, 재시도...)")
                continue
            else:
                raise
        except Exception as e:
            print(f"[Ep {ep_count+1:3d}] ERROR: {e}, 재시도...")
            continue
    
    # 최종 통계
    total_time = time.time() - start_time
    avg_steps = sum(r['steps'] for r in results) / len(results)
    avg_reward = sum(r['reward'] for r in results) / len(results)
    max_steps = max(r['steps'] for r in results)
    max_reward = max(r['reward'] for r in results)
    min_reward = min(r['reward'] for r in results)
    success_rate = (successes / num_episodes) * 100
    
    # Reward 분포
    positive_rewards = [r['reward'] for r in results if r['reward'] > 0]
    negative_rewards = [r['reward'] for r in results if r['reward'] < 0]
    
    print(f"\n{'='*70}")
    print("[RESULTS] 벤치마크 최종 결과:")
    print(f"{'='*70}")
    print(f"총 에피소드: {num_episodes}")
    print(f"총 실행 시간: {total_time:.1f}s ({total_time/60:.1f}분)")
    print(f"에피소드당 평균 시간: {total_time/num_episodes:.2f}s")
    print(f"\n{'-'*70}")
    print(f"평균 Steps:  {avg_steps:.1f}")
    print(f"평균 Reward: {avg_reward:.2f}")
    print(f"최대 Steps:  {max_steps}")
    print(f"최대 Reward: {max_reward:.2f}")
    print(f"최소 Reward: {min_reward:.2f}")
    print(f"\n{'-'*70}")
    print(f"성공률 (Reward > 0): {success_rate:.1f}% ({successes}/{num_episodes})")
    if positive_rewards:
        print(f"성공 시 평균 Reward: {sum(positive_rewards)/len(positive_rewards):.2f}")
    if negative_rewards:
        print(f"실패 시 평균 Reward: {sum(negative_rewards)/len(negative_rewards):.2f}")
    print(f"{'='*70}\n")
    
    return results

def run_full_benchmark(episodes_per_combo=100):
    """
    2x2 조합 벤치마크 실행
    - Lattice + Stanley
    - Lattice + EnhancedP
    - TrajectoryRollout + Stanley  
    - TrajectoryRollout + EnhancedP
    """
    import time
    
    combinations = [
        ("Lattice", "Stanley", True, True),
        ("Lattice", "EnhancedP", True, False),
        ("TrajectoryRollout", "Stanley", False, True),
        ("TrajectoryRollout", "EnhancedP", False, False),
    ]
    
    all_results = {}
    total_start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"[FULL BENCHMARK] 2x2 조합 벤치마크 시작")
    print(f"   각 조합당 {episodes_per_combo}회 테스트")
    print(f"   총 {len(combinations) * episodes_per_combo}회 실행")
    print(f"{'='*80}\n")
    
    for idx, (planner_name, controller_name, use_lattice, use_stanley) in enumerate(combinations, 1):
        print(f"\n{'#'*80}")
        print(f"# 조합 {idx}/4: {planner_name} Planner + {controller_name} Controller")
        print(f"{'#'*80}\n")
        
        # 글로벌 설정 변경
        global USE_LATTICE_PLANNER, USE_STANLEY_CONTROLLER, SELECTED_CONTROLLER
        USE_LATTICE_PLANNER = use_lattice
        USE_STANLEY_CONTROLLER = use_stanley
        SELECTED_CONTROLLER = StanleyController if use_stanley else EnhancedPController
        
        # 벤치마크 실행
        combo_start = time.time()
        results = benchmark(episodes_per_combo)
        combo_time = time.time() - combo_start
        
        # 통계 계산
        avg_reward = sum(r['reward'] for r in results) / len(results)
        avg_steps = sum(r['steps'] for r in results) / len(results)
        max_reward = max(r['reward'] for r in results)
        min_reward = min(r['reward'] for r in results)
        success_rate = (sum(1 for r in results if r['reward'] > 0) / len(results)) * 100
        
        combo_key = f"{planner_name}+{controller_name}"
        all_results[combo_key] = {
            'results': results,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'success_rate': success_rate,
            'time': combo_time
        }
        
        print(f"\n[조합 {idx} 완료] 시간: {combo_time:.1f}s")
        print(f"   평균 Reward: {avg_reward:.2f}, 성공률: {success_rate:.1f}%\n")
    
    # 최종 비교 결과
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("[FINAL COMPARISON] 전체 조합 성능 비교")
    print(f"{'='*80}")
    print(f"총 실행 시간: {total_time:.1f}s ({total_time/60:.1f}분)\n")
    print(f"{'플래너+컨트롤러':<30} {'평균 Reward':<15} {'평균 Steps':<12} {'성공률':<10} {'시간(s)':<10}")
    print(f"{'-'*80}")
    
    # 성능순으로 정렬
    sorted_combos = sorted(all_results.items(), key=lambda x: x[1]['avg_reward'], reverse=True)
    
    for combo_key, stats in sorted_combos:
        print(f"{combo_key:<30} {stats['avg_reward']:>12.2f}   {stats['avg_steps']:>10.1f}   "
              f"{stats['success_rate']:>7.1f}%   {stats['time']:>8.1f}")
    
    print(f"{'='*80}")
    
    # 최고 성능 조합
    best_combo = sorted_combos[0]
    print(f"\n🏆 최고 성능: {best_combo[0]}")
    print(f"   평균 Reward: {best_combo[1]['avg_reward']:.2f}")
    print(f"   최대 Reward: {best_combo[1]['max_reward']:.2f}")
    print(f"   성공률: {best_combo[1]['success_rate']:.1f}%")
    print(f"{'='*80}\n")
    
    return all_results

if __name__ == "__main__":
    """
    ========================================
    📘 실행 방법 가이드
    ========================================
    
    1️⃣ [기본 실행] 벤치마크 100회 자동 실행 (4가지 조합)
       python src\\model\\test_local_2.py
       
       → 자동으로 100회씩 4가지 조합을 테스트하고 성능 비교 결과 출력
       → 조합: Lattice+Stanley, Lattice+EnhancedP, TrajectoryRollout+Stanley, TrajectoryRollout+EnhancedP
    
    
    2️⃣ [벤치마크 횟수 지정] N회 벤치마크 실행
       python src\\model\\test_local_2.py benchmark 50
       python src\\model\\test_local_2.py benchmark 200
       
       → 원하는 횟수만큼 각 조합 테스트
    
    
    3️⃣ [Pygame 시각화 모드] 실시간 경로 확인 (녹색 경로 표시)
       python src\\model\\test_local_2.py visual
       
       → Pygame 창에서 차량 주행 + 플래너 경로(녹색선) 실시간 확인
       → 현재 조합: TrajectoryRollout + Stanley (최고 성능)
       → ESC 키로 종료
    
    
    4️⃣ [특정 조합만 테스트] 코드 수정 필요
       - USE_LATTICE_PLANNER = True/False (34번째 줄)
       - SELECTED_CONTROLLER = StanleyController / EnhancedPController (선택 필요)
    
    ========================================
    """
    import sys
    
    # 실행 모드 선택 (기본값: 벤치마크 100회)
    mode = "benchmark"  # 기본값 변경: 벤치마크 모드
    num_episodes = 100  # 기본 100회
    
    # sys.argv에서 우리의 커스텀 인자 체크 및 제거 (env 초기화 전에 처리)
    if len(sys.argv) > 1:
        if sys.argv[1] == "visual":
            mode = "visual"
            sys.argv = [sys.argv[0]]  # visual 인자 제거
        elif sys.argv[1] == "benchmark":
            mode = "benchmark"
            # benchmark 인자와 횟수 저장 후 제거
            if len(sys.argv) > 2:
                try:
                    num_episodes = int(sys.argv[2])
                except:
                    pass
            sys.argv = [sys.argv[0]]  # benchmark 관련 인자 제거
    
    if mode == "benchmark":
        # ====== 벤치마크 모드 (4가지 조합 x N회) ======
        print("\n" + "="*80)
        print("🔬 4가지 조합 벤치마크 시작!")
        print(f"  각 조합당 {num_episodes}회 실행")
        print("  1) Lattice + Stanley")
        print("  2) Lattice + EnhancedP")
        print("  3) TrajectoryRollout + Stanley")
        print("  4) TrajectoryRollout + EnhancedP")
        print("="*80 + "\n")
        
        run_full_benchmark(episodes_per_combo=num_episodes)
        
    else:
        # ====== 시각화 모드 (Pygame) ======
        print("\n" + "="*70)
        print("[시각화 모드] TrajectoryRollout Planner + Stanley Controller")
        print("  Pygame에서 실시간 경로 확인 가능")
        print("  - 녹색 선: 플래너가 계획한 로컬 경로")
        print("  - ESC: 종료")
        print("  - F1~F9: 환경 제어 (자세한 내용은 시작시 출력 참고)")
        print("="*70 + "\n")
        
        main()
