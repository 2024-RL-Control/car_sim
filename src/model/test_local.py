# -*- coding: utf-8 -*-
"""
test_local.py  (Hybrid Local Planner + Advanced Stanley / Enhanced P)
- BasicRLDrivingEnv 시뮬레이터 그대로 사용
- 로컬 경로 계획:
  1) 세그먼트 기반 Lattice (가능시): centerline + lateral offsets
  2) 세그먼트가 약하면: centerline 불완전 샘플 뒤 yaw-synth 로 lookahead 연장
  3) road_info 불가/세그먼트 전무: Arc Rollout(상수 곡률 아크 후보)로 항상 경로 생성
  => 어떤 상황에서도 path를 반환하므로 pygame 화면/주행이 안정적으로 시작됨
- 컨트롤러는 하이브리드 플래너의 경로를 lookahead 참조해 ψ_e, e, κ를 계산
"""

import os
import sys
import math
import numpy as np
import pygame

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
        print("[Patch] VehicleState.encoding_angle -> safe")
except Exception as e:
    print("[Patch] encoding_angle monkey-patch failed:", repr(e))

# ============================
# 전역 설정
# ============================
USE_PLANNER = True            # 하이브리드 로컬 플래너 사용
LOOKAHEAD_M_FOR_REF = 4.0     # 경로 상 lookahead 참조 거리
DEBUG_PRINT_LIMIT = 10        # 동일 유형 로그 반복 출력 제한

SHOW_LOCAL_PATH = True        # 로컬 경로 시각화 on/off
PATH_COLOR = (40, 220, 120)   # 경로 라인 색
PATH_PT_COLOR = (255, 255, 255)  # 경로 점 색
PATH_WIDTH = 2                # 라인 두께(px)
PATH_PT_STEP = 2              # 점 간격(샘플 스킵)

def wrap_to_pi(ang):
    return (ang + math.pi) % (2 * math.pi) - math.pi

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

# ============================
# 하이브리드 로컬 플래너
# ============================
class HybridLocalPlanner:
    """
    목표: 어떤 상황에서도 경로 반환
    1) road_info + closest_segment 있으면: 세그먼트 라티스(센터라인 + lateral offsets)
       - 세그 샘플 중간 실패 시: 마지막 yaw로 yaw-synth 이어붙여 lookahead 채움
    2) road_info만 있고 세그먼트가 약하면: road_yaw 기반 yaw-synth
    3) road_info가 전무하면: Arc Rollout (상수 곡률)로 후보 생성 후 비용 최소
    반환 형식: [(x,y,yaw,kappa), ...]
    """

    def __init__(self, road_api, obstacle_api,
                 lookahead=28.0, ds=1.0,
                 lateral_offsets=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
                 max_curv_lattice=0.35,    # lattice 시 곡률 상한
                 arc_rollout_ks=None,      # 아크 곡률 후보 세트
                 wheelbase=2.6,
                 debug=True):
        self.road = road_api
        self.obs_api = obstacle_api
        self.lookahead = float(lookahead)
        self.ds = max(0.5, float(ds))
        self.offsets = list(lateral_offsets)
        self.max_curv_lat = float(max_curv_lattice)
        self.wheelbase = float(wheelbase)
        self.debug = debug

        if arc_rollout_ks is None:
            # 직진 포함, 좌우 다양한 곡률(반경 대략 6~∞ m)
            self.arc_rollout_ks = np.array([-0.18, -0.12, -0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08, 0.12, 0.18], dtype=float)
        else:
            self.arc_rollout_ks = np.array(arc_rollout_ks, dtype=float)

        self._dbg = {"noinfo":0, "segpose":0, "center":0, "yaw":0, "arc":0, "lattice_fail":0}

    # ---------- 공통 유틸 ----------
    @staticmethod
    def _nvec(yaw):  # 도로 좌법선(좌측 +)
        return -math.sin(yaw), math.cos(yaw)

    @staticmethod
    def _nearest_on_poly(pose, poly):
        px, py, _ = pose
        best = (None, 1e18, 0)
        for i, (x, y, yaw, k) in enumerate(poly):
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best[1]:
                best = ((x, y, yaw, k), d2, i)
        return best

    def _print_once(self, key, msg):
        if not self.debug:
            return
        if self._dbg.get(key, 0) < DEBUG_PRINT_LIMIT:
            print(msg)
            self._dbg[key] = self._dbg.get(key, 0) + 1

    def _sample_obstacles(self):
        try:
            arr = self.obs_api.get_all_outer_circles()
            return list(arr or [])
        except Exception:
            return []

    # ---------- 1) 세그먼트 라티스 ----------
    def _segpose(self, seg, s):
        try:
            x, y = seg.get_position_at_s(s)
            yaw = seg.get_yaw_at_s(s)
            kappa = seg.get_curvature_at_s(s)
            return float(x), float(y), float(yaw), float(kappa)
        except Exception:
            return None

    def _build_centerline(self, info):
        seg = info.get('closest_segment', None)
        if seg is None:
            return None
        s0 = float(info.get('s', 0.0))
        seg_len = float(info.get('segment_length', 0.0))
        la = min(self.lookahead, max(3.0, seg_len - s0))
        s_end = s0 + la

        pts = []
        s = s0
        while s <= s_end + 1e-6:
            base = self._segpose(seg, s)
            if base is None:
                self._print_once("segpose", f"[LAT] seg pose failed @ s={s:.2f}")
                break
            x, y, yaw, k = base
            if abs(k) > self.max_curv_lat * 1.8:
                # 너무 급커브면 컷; yaw-synth로 이어붙일 것
                break
            pts.append((x, y, yaw, k))
            s += self.ds
        return pts if len(pts) >= 2 else None

    def _extend_with_yaw_synth(self, path, final_yaw):
        """centerline이 짧거나 샘플 실패 시, 현재 마지막 점/방향으로 lookahead까지 이어붙임"""
        if not path:
            return None
        x, y, yaw, k = path[-1]
        dist = 0.0
        while dist < self.lookahead and len(path) < int(self.lookahead / self.ds) + 2:
            # k=0으로 직선으로만 보간; yaw는 마지막 세그먼트 yaw 유지 또는 final_yaw로 보정
            yaw = final_yaw
            x += math.cos(yaw) * self.ds
            y += math.sin(yaw) * self.ds
            path.append((x, y, yaw, 0.0))
            dist += self.ds
        return path

    def _lattice_from_centerline(self, centerline, offsets):
        """센터라인을 좌/우로 평행 이동한 라티스 후보들 생성"""
        if not centerline:
            return []
        paths = []
        for d in offsets:
            samples = []
            for (x, y, yaw, k) in centerline:
                nx, ny = self._nvec(yaw)
                x2 = x + d * nx
                y2 = y + d * ny
                samples.append((x2, y2, yaw, k))
            if len(samples) >= 2:
                paths.append((samples, d))
        return paths  # [(path, offset), ...]

    # ---------- 2) road_yaw 기반 yaw-synth ----------
    def _path_from_road_yaw(self, yaw, start_pose):
        x, y, _ = start_pose
        pts = []
        dist = 0.0
        while dist <= self.lookahead + 1e-6:
            pts.append((float(x), float(y), float(yaw), 0.0))
            x += math.cos(yaw) * self.ds
            y += math.sin(yaw) * self.ds
            dist += self.ds
        return pts if len(pts) >= 2 else None

    # ---------- 3) Arc Rollout (세그먼트 불요) ----------
    def _simulate_arc(self, start_pose, kappa):
        """상수 곡률 kappa로 self.lookahead까지 전개"""
        x, y, yaw = start_pose
        pts = []
        dist = 0.0
        while dist <= self.lookahead + 1e-6:
            pts.append((float(x), float(y), float(yaw), float(kappa)))
            yaw = yaw + kappa * self.ds
            x += math.cos(yaw) * self.ds
            y += math.sin(yaw) * self.ds
            dist += self.ds
        return pts

    def _arc_rollout_candidates(self, start_pose):
        return [self._simulate_arc(start_pose, k) for k in self.arc_rollout_ks]

    # ---------- 비용 함수 ----------
    def _clearance_cost(self, path, obstacles, min_clear=0.6):
        if not obstacles:
            return 0.0
        min_d = 1e9
        hit = False
        for (x, y, _, _) in path[::2]:
            for (ox, oy, r) in obstacles:
                d = math.hypot(x - ox, y - oy) - r
                if d < 0.0:
                    hit = True
                    min_d = -1.0
                    break
                min_d = min(min_d, d)
            if hit:
                break
        if hit:
            return 1e5  # 충돌은 큰 패널티
        if min_d < min_clear:
            return 1200.0 * (min_clear - max(min_d, 0.0))
        # 여유가 클수록 비용 작게
        return 1.0 / (1.0 + min_d)

    def _smooth_cost(self, path):
        ks = [p[3] for p in path]
        dks = [abs(ks[i] - ks[i-1]) for i in range(1, len(ks))]
        return float(np.mean(dks)) if dks else 0.0

    def _heading_align_cost(self, path, desired_yaw):
        yaw_end = path[-1][2]
        return abs(wrap_to_pi(yaw_end - desired_yaw))

    def _center_cost_from_offset(self, offset):
        return abs(offset)  # 중앙 정렬 선호

    # ---------- 메인: path 생성 ----------
    def get_local_path(self, vehicle_pose):
        obstacles = self._sample_obstacles()

        # 0) road_info 질의
        info = None
        try:
            info = self.road.get_vehicle_road_info(vehicle_pose)
        except Exception:
            info = None

        # A) 세그먼트 라티스 시도
        if info and info.get('closest_segment', None) is not None:
            centerline = self._build_centerline(info)
            if centerline:
                # 필요시 centerline을 yaw-synth로 lookahead까지 확장
                yaw_final = float(centerline[-1][2] if centerline else info.get('road_yaw', vehicle_pose[2]))
                if len(centerline) * self.ds < self.lookahead - 1e-6:
                    centerline = self._extend_with_yaw_synth(centerline, yaw_final)
                # 라티스 후보 생성
                lat_paths = self._lattice_from_centerline(centerline, self.offsets)
                if lat_paths:
                    # 비용으로 최적 경로 선택
                    desired_yaw = float(info.get('road_yaw', vehicle_pose[2]))
                    best, bestJ = None, 1e18
                    for path, off in lat_paths:
                        J = ( 1.8 * self._clearance_cost(path, obstacles)
                            + 0.8 * self._smooth_cost(path)
                            + 0.8 * self._center_cost_from_offset(off)
                            + 0.9 * self._heading_align_cost(path, desired_yaw))
                        if J < bestJ:
                            best, bestJ = path, J
                    if best:
                        return best
            # 세그먼트 존재했지만 샘플이 나쁘면 yaw-synth 폴백
            yaw = float(info.get('road_yaw', vehicle_pose[2]))
            path = self._path_from_road_yaw(yaw, vehicle_pose)
            if path:
                self._print_once("yaw", f"[LAT] fallback yaw-synth len={len(path)}")
                return path
            self._print_once("lattice_fail", "[LAT] lattice centerline failed & yaw-synth failed")

        # B) road_info는 있으나 세그먼트가 없을 때 → yaw-synth
        if info and info.get('closest_segment', None) is None:
            yaw = float(info.get('road_yaw', vehicle_pose[2]))
            path = self._path_from_road_yaw(yaw, vehicle_pose)
            if path:
                self._print_once("yaw", f"[LAT] yaw-synth len={len(path)} (no segment)")
                return path

        # C) road_info 전무 → Arc Rollout
        self._print_once("noinfo", "[LAT] road_info None → arc rollout")
        cands = self._arc_rollout_candidates(vehicle_pose)
        # desired_yaw을 알면 헤딩 정렬 비용에 반영
        desired_yaw = vehicle_pose[2]
        try:
            upd = self.road.get_vehicle_update_data(vehicle_pose)
            # update_data 안에서 [-pi, pi] 범위의 값 하나를 road_yaw로 추정
            for val in upd:
                try:
                    f = float(val)
                    if -math.pi - 1e-3 <= f <= math.pi + 1e-3:
                        desired_yaw = f
                        break
                except Exception:
                    pass
        except Exception:
            pass

        best, bestJ = None, 1e18
        for path in cands:
            J = ( 1.8 * self._clearance_cost(path, obstacles)
                + 0.8 * self._smooth_cost(path)
                + 1.0 * self._heading_align_cost(path, desired_yaw))
            if J < bestJ:
                best, bestJ = path, J
        if best:
            self._print_once("arc", f"[LAT] arc rollout chosen (len={len(best)})")
            return best

        # D) 최후 직진 (이론상 도달하지 않음)
        x, y, yaw = vehicle_pose
        pts = []
        d = 0.0
        while d <= self.lookahead + 1e-6:
            pts.append((x, y, yaw, 0.0))
            x += math.cos(yaw) * self.ds
            y += math.sin(yaw) * self.ds
            d += self.ds
        return pts

# ============================
# 공통 베이스 컨트롤러
# ============================
class BaseController:
    """
    공통:
      - 로컬 경로(하이브리드) 참조로 ψ_e, e, κ 추정 → 조향
      - 권장속도(도로/업데이트) 기반 P 제어 + 장애물 간이 감속/반발
      - 액션포맷 자동 인식([accel, steer] 또는 [throttle, brake, steer])
      - 라디안 조향 → [-1,1] 정규화
    """
    def __init__(self, env, action_low, action_high):
        self.env = env
        self.core = env.env  # CarSimulatorEnv
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_dim = self.action_low.shape[0]

        # 차량 파라미터
        self.max_steer_rad = math.radians(self._safe_get(self.core.config, ['vehicle', 'max_steer'], default=30.0))
        self.wheelbase_m = float(self._safe_get(self.core.config, ['vehicle', 'wheelbase'], default=2.6))

        # dt
        self.dt = float(self._safe_get(self.core.config, ['simulation', 'dt'], default=1.0/60.0))

        # 속도 제어
        self.k_speed_p = 1.2
        self.v_min = 2.0
        self.v_max = 18.0
        self.brake_on_negative_a = True
        self.a_brake_safe = 3.0  # [m/s^2]

        # 장애물 반응형 회피(옵션)
        self.enable_obstacle_repulsion = True
        self.obs_check_radius = 12.0
        self.obs_fov_deg = 120.0
        self.k_obs = 0.9
        self.v_obs_min = 3.0

        # 액션 포맷
        if self.action_dim == 2:
            self.format = "AS"; self.i_accel, self.i_steer = 0, 1
        elif self.action_dim == 3:
            self.format = "TBS"; self.i_throttle, self.i_brake, self.i_steer = 0, 1, 2
        else:
            self.format = "UNKNOWN"

        # 조향 후처리
        self.use_steer_rate_limit = True
        self.steer_rate_limit = math.radians(140.0)
        self.use_steer_lpf = True
        self.steer_lpf_alpha = 0.35
        self._steer_cmd_prev = 0.0

        # 하이브리드 플래너
        self.use_planner = USE_PLANNER
        self.planner = HybridLocalPlanner(
            road_api=self.core.road_manager,
            obstacle_api=self.core.obstacle_manager,
            lookahead=28.0, ds=1.0,
            lateral_offsets=(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
            max_curv_lattice=0.35,
            wheelbase=self.wheelbase_m,
            debug=True
        )
        self._viz_last_path = None
        # 필터
        self.e_alpha = 0.35
        self.psi_alpha = 0.35
        self.kappa_alpha = 0.5
        self._e_f = 0.0
        self._psi_f = 0.0
        self._kappa_f = 0.0

    # ---------- 유틸 ----------
    def _safe_get(self, d, keys, default=None):
        cur = d
        try:
            for k in keys:
                cur = cur[k]
            return cur
        except Exception:
            return default

    def _normalize_steer_to_unit(self, steer_rad):
        return clamp(steer_rad / (self.max_steer_rad + 1e-6), -1.0, 1.0)

    def speed_ref_from_update(self, vehicle_pose):
        # road_manager.get_vehicle_update_data()로 권장속도 추정(스펙 불명 대비)
        try:
            upd = self.core.road_manager.get_vehicle_update_data(vehicle_pose)
            # float이며 적당히 빠른 값(>1.0) 후보 중 최소를 택함
            cand = []
            for v in upd:
                try:
                    f = float(v)
                    if f > 1.0 and f < 50.0:
                        cand.append(f)
                except Exception:
                    pass
            if cand:
                return clamp(min(cand), self.v_min, self.v_max)
        except Exception:
            pass
        return self.v_max

    def speed_with_obstacle_brake(self, v_ref, st):
        if not self.enable_obstacle_repulsion:
            return v_ref
        try:
            oxyr_list = self.core.obstacle_manager.get_all_outer_circles() or []
        except Exception:
            oxyr_list = []
        if not oxyr_list:
            return v_ref

        vx, vy, vyaw = st.x, st.y, st.yaw
        cos_y, sin_y = math.cos(vyaw), math.sin(vyaw)
        fov_cos = math.cos(math.radians(self.obs_fov_deg) / 2.0)

        v_safe_lim = v_ref
        for (ox, oy, r) in oxyr_list:
            dx, dy = ox - vx, oy - vy
            dist_edge = max(0.0, math.hypot(dx, dy) - r)
            if dist_edge > self.obs_check_radius:
                continue
            front = dx * cos_y + dy * sin_y
            if front <= 0:
                continue
            dir_norm = math.hypot(dx, dy) + 1e-6
            cos_angle = (dx * cos_y + dy * sin_y) / dir_norm
            if cos_angle < fov_cos:
                continue
            v_safe = math.sqrt(max(0.0, 2.0 * self.a_brake_safe * dist_edge))
            v_safe_lim = min(v_safe_lim, max(self.v_obs_min, v_safe))
        return v_safe_lim

    def steer_obstacle_repulsion(self, st):
        if not self.enable_obstacle_repulsion:
            return 0.0
        try:
            oxyr_list = self.core.obstacle_manager.get_all_outer_circles() or []
        except Exception:
            oxyr_list = []
        if not oxyr_list:
            return 0.0

        vx, vy, vyaw = st.x, st.y, st.yaw
        cos_y, sin_y = math.cos(vyaw), math.sin(vyaw)
        fov_cos = math.cos(math.radians(self.obs_fov_deg) / 2.0)

        steer_obs = 0.0
        for (ox, oy, r) in oxyr_list:
            dx, dy = ox - vx, oy - vy
            dist = math.hypot(dx, dy)
            if dist - r > self.obs_check_radius:
                continue
            cos_angle = (dx * cos_y + dy * sin_y) / (dist + 1e-9)
            if cos_angle < fov_cos:
                continue
            side = -dx * sin_y + dy * cos_y
            edge = max(0.3, dist - r)
            steer_obs += 0.9 * math.atan2(side, edge**2)
        return clamp(steer_obs, -self.max_steer_rad*0.7, self.max_steer_rad*0.7)

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

    # ---------- 경로 참조로 ψ_e, e, κ 계산 ----------
    def ref_from_path(self, vehicle_pose):
        """
        로컬 경로(하이브리드)로부터 (psi_e_now, e_now, kappa_now) 계산
        항상 유효한 path를 보장
        """
        if not self.use_planner:
            # 플래너 미사용: road yaw만으로 대략 헤딩 참조, CTE=0, κ=0
            try:
                info = self.core.road_manager.get_vehicle_road_info(vehicle_pose)
                yaw_ref = float(info.get('road_yaw', vehicle_pose[2])) if info else vehicle_pose[2]
            except Exception:
                yaw_ref = vehicle_pose[2]
            return wrap_to_pi(yaw_ref - vehicle_pose[2]), 0.0, 0.0

        path = self.planner.get_local_path(vehicle_pose)
        self._viz_last_path = path
        # 가장 가까운 점 + lookahead 인덱스
        (cx, cy, cyaw, ck), _, idx = self.planner._nearest_on_poly(vehicle_pose, path)
        # lookahead index
        steps_ahead = int(max(1.0, LOOKAHEAD_M_FOR_REF) / max(0.5, self.planner.ds))
        look_idx = min(idx + steps_ahead, len(path) - 1)
        yaw_ref = path[look_idx][2]
        kappa_now = path[look_idx][3]

        # 횡오차 (노말 성분)
        dx, dy = vehicle_pose[0] - cx, vehicle_pose[1] - cy
        e_now = -dx * math.sin(cyaw) + dy * math.cos(cyaw)
        psi_e_now = wrap_to_pi(yaw_ref - vehicle_pose[2])
        return psi_e_now, e_now, kappa_now

    # ---------- 자식이 구현 ----------
    def steer_cmd_core(self, vehicle_pose, v_long):
        raise NotImplementedError

    # ---------- 1대/배치 ----------
    def act_single(self, veh):
        st = veh.state
        pose = (st.x, st.y, st.yaw)
        v_long = float(st.vel_long)

        # 경로 참조
        psi_e_now, e_now, kappa_now = self.ref_from_path(pose)

        # 자식 컨트롤러가 ψ_e, e, κ 사용
        steer_rad = self.steer_cmd_core_impl(psi_e_now, e_now, kappa_now, v_long)

        # 장애물 반발항
        steer_rad += self.steer_obstacle_repulsion(st)
        steer_rad = wrap_to_pi(steer_rad)
        steer_rad = self.postprocess_steer(steer_rad)

        # 권장 속도
        v_ref = self.speed_ref_from_update(pose)
        v_ref = self.speed_with_obstacle_brake(v_ref, st)

        # 속도 P
        a_cmd = 1.2 * (v_ref - max(0.0, v_long))

        # 액션 패킹
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

    def act_batch(self):
        vehicles = self.core.vehicle_manager.get_all_vehicles()
        acts = np.zeros((len(vehicles), self.action_dim), dtype=np.float32)
        for i, veh in enumerate(vehicles):
            acts[i] = self.act_single(veh)
        return acts

# ============================
# Advanced Stanley Controller
# ============================
class AdvancedStanleyController(BaseController):
    def __init__(self, env, action_low, action_high):
        super().__init__(env, action_low, action_high)
        self.k0 = 2.0
        self.v0 = 4.0
        self.eps_v = 0.3

        # 내부 필터 상태는 BaseController에 존재

    def _k_cte(self, v):
        return self.k0 * (v / (v + self.v0))

    def steer_cmd_core_impl(self, psi_e, e, kappa, v_long):
        v = max(0.0, float(v_long))
        # 필터
        self._e_f   = (1 - self.e_alpha)   * self._e_f   + self.e_alpha   * e
        self._psi_f = (1 - self.psi_alpha) * self._psi_f + self.psi_alpha * psi_e
        self._kappa_f = (1 - self.kappa_alpha) * self._kappa_f + self.kappa_alpha * kappa

        # Stanley + Curvature FF
        steer = self._psi_f + math.atan2(self._k_cte(v) * self._e_f, (v + self.eps_v)) + math.atan(self.wheelbase_m * self._kappa_f)
        return clamp(steer, -self.max_steer_rad, self.max_steer_rad)

# ============================
# Enhanced P Controller
# ============================
class EnhancedPController(BaseController):
    def __init__(self, env, action_low, action_high):
        super().__init__(env, action_low, action_high)
        self.k_h_base = 1.2
        self.k_d_base = 0.3
        self.vh = 6.0
        self.vd = 6.0

    def k_h(self, v):
        return self.k_h_base * (self.vh / (self.vh + max(0.0, v)))

    def k_d(self, v):
        return self.k_d_base * (self.vd / (self.vd + max(0.0, v)))

    def steer_cmd_core_impl(self, psi_e, e, kappa, v_long):
        v = max(0.0, float(v_long))
        self._e_f   = (1 - self.e_alpha)   * self._e_f   + self.e_alpha   * e
        self._psi_f = (1 - self.psi_alpha) * self._psi_f + self.psi_alpha * psi_e
        self._kappa_f = (1 - self.kappa_alpha) * self._kappa_f + self.kappa_alpha * kappa
        steer = self.k_h(v) * self._psi_f + self.k_d(v) * self._e_f + math.atan(self.wheelbase_m * self._kappa_f)
        return clamp(steer, -self.max_steer_rad, self.max_steer_rad)

# ============================
# 컨트롤러 선택 (여기 한 줄만 바꾸세요)
# ============================
#SELECTED_CONTROLLER = EnhancedPController
SELECTED_CONTROLLER = AdvancedStanleyController

# ============================
# 메인 루프
# ============================
def _get_draw_surface(env_like):
    """env 객체에서 pygame Surface를 최대한 유연하게 얻는다."""
    surf = None
    # 1) 환경이 화면 surface를 직접 노출
    for attr in ("surface", "screen", "window", "canvas"):
        if hasattr(env_like, attr):
            ok = getattr(env_like, attr)
            if hasattr(ok, "blit"):  # pygame.Surface heuristic
                surf = ok; break
    # 2) 내부 env/env.renderer 쪽
    if surf is None and hasattr(env_like, "env"):
        for holder in (env_like.env, getattr(env_like.env, "renderer", None)):
            if holder is None: continue
            for attr in ("surface", "screen", "window", "canvas"):
                if hasattr(holder, attr):
                    ok = getattr(holder, attr)
                    if hasattr(ok, "blit"):
                        surf = ok; break
            if surf is not None: break
    # 3) pygame 전역에서 현재 디스플레이 가져오기
    if surf is None:
        try:
            surf = pygame.display.get_surface()
        except Exception:
            surf = None
    return surf

def _world_to_screen(env_like, x, y):
    """
    환경이 제공하는 월드→스크린 변환을 최대한 시도.
    없으면 대충 현재 디스플레이 좌표계로 근사(줌/팬 없음).
    """
    # 1) 가장 직접적인 변환기들
    for obj in (env_like, getattr(env_like, "env", None),
                getattr(getattr(env_like, "env", None), "renderer", None),
                getattr(env_like, "renderer", None),
                getattr(env_like, "camera", None),
                getattr(getattr(env_like, "env", None), "camera", None)):
        if obj is None: 
            continue
        for fn in ("world_to_screen", "to_screen", "w2s"):
            if hasattr(obj, fn):
                try:
                    out = getattr(obj, fn)((x, y)) if fn in ("to_screen",) else getattr(obj, fn)(x, y)
                    # 다양한 형태 허용
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        return int(out[0]), int(out[1])
                except Exception:
                    pass
    # 2) Fallback: 화면 중앙을 원점으로 가정(단순 근사)
    surf = pygame.display.get_surface()
    if surf:
        W, H = surf.get_width(), surf.get_height()
        sx = int(W/2 + x*10.0)   # 1m ≈ 10px 가정 (필요시 조절)
        sy = int(H/2 - y*10.0)   # y축 반전
        return sx, sy
    return int(x), int(y)
def draw_local_path(env_like, path, color=(40,220,120), pt_color=(255,255,255),
                    width=2, pt_step=2):
    if not path or len(path) < 2:
        return
    surf = _get_draw_surface(env_like)
    if surf is None:  # 화면 표면을 못 찾으면 패스
        return

    # 라인 그리기
    pts_screen = []
    for (x, y, yaw, k) in path:
        sx, sy = _world_to_screen(env_like, x, y)
        pts_screen.append((sx, sy))
    try:
        pygame.draw.lines(surf, color, False, pts_screen, width)
    except Exception:
        pass

    # 점 찍기 (가독성)
    for i in range(0, len(pts_screen), max(1, pt_step)):
        try:
            pygame.draw.circle(surf, pt_color, pts_screen[i], 2)
        except Exception:
            pass
def main():
    env = BasicRLDrivingEnv()
    observations, active_agents = env.reset()

    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]

    controller = SELECTED_CONTROLLER(env, action_low, action_high)
    controller.enable_obstacle_repulsion = True  # 원하면 False

    env.render()
    # --- 로컬 경로 시각화 ---
    if SHOW_LOCAL_PATH and hasattr(controller, "_viz_last_path"):
        draw_local_path(env, controller._viz_last_path,
                        color=PATH_COLOR, pt_color=PATH_PT_COLOR,
                        width=PATH_WIDTH, pt_step=PATH_PT_STEP)
    # 화면 갱신(일부 렌더러는 env.render 내부에서 flip하지만, 안전하게 한 번 더)
    try:
        pygame.display.flip()
    except Exception:
        pass
    env.print_basic_controls()
    print(f"[Controller] {SELECTED_CONTROLLER.__name__}")
    print("ESC로 종료.\n")

    clock = pygame.time.Clock()
    done = False
    ep_reward = 0.0
    steps = 0

    try:
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True

            env.handle_keyboard_input()

            actions = controller.act_batch()
            observations, reward, done, _, info = env.step(actions)
            ep_reward += float(reward)
            steps += 1

            env.render()
            clock.tick(60)

        print(f"\n[Run End] Steps: {steps}, Total Reward: {ep_reward:.2f} (Controller: {SELECTED_CONTROLLER.__name__})")

    finally:
        env.close()

if __name__ == "__main__":
    main()
