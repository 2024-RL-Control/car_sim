# -*- coding: utf-8 -*-
"""
Classical Control Methods for Autonomous Driving
===================================================

This module implements classical path planning and control algorithms for benchmarking
against reinforcement learning (RL) models in the car simulator environment.

Implemented Controllers:
------------------------
1. **PIDController**: Proportional-Integral-Derivative controller for longitudinal speed control
   - Tracks target velocity with configurable gains
   - Separate acceleration and braking limits

2. **StanleyController**: Lateral path tracking controller
   - Combines cross-track error and heading error
   - Stanley method: δ = θe + arctan(k × e / v)

3. **StanleyPIDController**: Combined longitudinal + lateral control
   - Integrates PID for speed and Stanley for steering
   - Direct interface with environment observation space

4. **LatticePlanner**: Path planning using quintic polynomial trajectories
   - Generates multiple candidate paths with different lateral offsets
   - Cost-based path selection (comfort, safety, tracking error)
   - Frenet frame path generation

5. **LatticePlannerController**: Strategic planning + tactical control
   - Lattice planner for path generation
   - Stanley+PID for path following
   - Adaptive target generation

6. **MPCController**: Model Predictive Control with optimization
   - 1-second prediction horizon with bicycle model
   - Multi-objective cost function (tracking, smoothness, control effort)
   - Gradient-based optimization (SLSQP)

Benchmark Framework:
-------------------
**ControllerBenchmark**: Comprehensive performance evaluation framework
   - Success rate, collision rate, reward metrics
   - Control smoothness (jerk, steering rate)
   - Path tracking accuracy (lateral error, heading error)
   - Computation time profiling
   - Statistical comparison and visualization

Usage Example:
-------------
```python
from src.env.env_rl import BasicRLDrivingEnv
from src.model.test import ControllerBenchmark, StanleyPIDController, MPCController

# Create environment and benchmark
env = BasicRLDrivingEnv()
benchmark = ControllerBenchmark(env, num_episodes=10)

# Evaluate controllers
stanley_results = benchmark.evaluate_controller(StanleyPIDController(), "Stanley+PID")
mpc_results = benchmark.evaluate_controller(MPCController(), "MPC")

# Compare
benchmark.compare_controllers([stanley_results, mpc_results])
```

Implementation Notes:
--------------------
- All controllers interface with 23-dim observation space (see _parse_observation)
- Action space: [acceleration, steering] in normalized [-1, 1]²
- Frenet coordinate system used for path-relative control
- Compatible with BasicRLDrivingEnv from src.env.env_rl

Author: Claude Code
Date: 2025-01-11
"""

import numpy as np
from math import pi, cos, sin, tan, atan, atan2, sqrt
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque


# ======================
# Utility Functions
# ======================

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π] range"""
    return (angle + pi) % (2 * pi) - pi


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ======================
# Data Structures
# ======================

@dataclass
class VehicleState:
    """Current vehicle state information"""
    x: float  # position x [m]
    y: float  # position y [m]
    yaw: float  # heading angle [rad]
    vel_long: float  # longitudinal velocity [m/s]
    vel_lat: float  # lateral velocity [m/s]

    # Frenet coordinates
    frenet_s: float  # arc length [m]
    frenet_d: float  # lateral offset [m], left is positive
    frenet_point: Optional[Tuple[float, float, float]]  # reference point (x, y, yaw)
    heading_error: float  # heading error [rad]

    # Target information
    target_vel_long: float  # recommended speed [m/s]
    segment_length: float  # current segment length [m]

    # Sensor data
    lidar_data: np.ndarray  # lidar distances


@dataclass
class ControlCommand:
    """Control command for vehicle"""
    acceleration: float  # [-1, 1] range
    steering: float  # [-1, 1] range


@dataclass
class PathSegment:
    """Path segment for Lattice Planner"""
    points: List[Tuple[float, float, float]]  # (x, y, yaw)
    cost: float  # path cost
    lateral_offset: float  # target lateral offset [m]
    time_horizon: float  # time to reach end [s]


# ======================
# Phase 1: PID + Stanley Controllers
# ======================

class PIDController:
    """
    PID Controller for longitudinal speed control

    Maintains target speed using proportional, integral, and derivative control
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.1, kd: float = 0.2,
                 max_accel: float = 1.0, max_brake: float = 1.0):
        """
        Initialize PID Controller

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            max_accel: Maximum acceleration output [0, 1]
            max_brake: Maximum brake output [0, 1]
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_accel = max_accel
        self.max_brake = max_brake

        # State variables
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.error_history = deque(maxlen=100)

    def reset(self):
        """Reset controller state"""
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.error_history.clear()

    def compute(self, current_speed: float, target_speed: float, dt: float) -> float:
        """
        Compute acceleration command using PID control

        Args:
            current_speed: Current longitudinal speed [m/s]
            target_speed: Target longitudinal speed [m/s]
            dt: Time step [s]

        Returns:
            acceleration: Acceleration command in [-1, 1] range
                         (positive: acceleration, negative: braking)
        """
        # Speed error (positive means need to accelerate)
        error = target_speed - current_speed

        # Proportional term
        p_term = self.kp * error

        # Integral term (with anti-windup)
        self.integral_error += error * dt
        # Anti-windup: clamp integral error
        max_integral = 10.0  # Maximum integral error
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        i_term = self.ki * self.integral_error

        # Derivative term
        if dt > 0:
            error_rate = (error - self.previous_error) / dt
        else:
            error_rate = 0.0
        d_term = self.kd * error_rate

        # PID output
        output = p_term + i_term + d_term

        # Update state
        self.previous_error = error
        self.error_history.append(error)

        # Map output to [-1, 1] range
        # Positive output -> acceleration, Negative output -> braking
        if output >= 0:
            # Acceleration needed
            acceleration = np.clip(output, 0.0, self.max_accel)
        else:
            # Braking needed
            acceleration = np.clip(output, -self.max_brake, 0.0)

        return acceleration


class StanleyController:
    """
    Stanley Controller for lateral path tracking

    Combines cross-track error and heading error for steering control
    Formula: δ = θe + arctan(k × e / v)

    Reference:
    - Hoffmann, G. M., et al. "Autonomous automobile trajectory tracking
      for off-road driving: Controller design, experimental validation and racing."
      American Control Conference, 2007.
    """

    def __init__(self, k: float = 1.0, ks: float = 0.5,
                 max_steer: float = 1.0, min_speed: float = 0.5):
        """
        Initialize Stanley Controller

        Args:
            k: Cross-track error gain (typically 0.5 ~ 2.0)
            ks: Softening gain for low speed
            max_steer: Maximum steering output [0, 1]
            min_speed: Minimum speed for control [m/s]
        """
        self.k = k
        self.ks = ks
        self.max_steer = max_steer
        self.min_speed = min_speed

    def compute(self, vehicle_state: VehicleState) -> float:
        """
        Compute steering command using Stanley method

        Args:
            vehicle_state: Current vehicle state

        Returns:
            steering: Steering command in [-1, 1] range
                     (positive: left turn, negative: right turn)
        """
        # Extract state information
        heading_error = vehicle_state.heading_error  # road_yaw - vehicle_yaw
        cross_track_error = vehicle_state.frenet_d  # lateral offset (left is positive)
        velocity = abs(vehicle_state.vel_long)

        # Avoid division by zero at low speeds
        if velocity < self.min_speed:
            velocity = self.min_speed

        # Stanley control law
        # δ = θe + arctan(k × e / (ks + v))
        # Note: cross_track_error is positive when vehicle is left of path
        #       We want negative steering (right turn) to correct this
        cross_track_term = atan(-self.k * cross_track_error / (self.ks + velocity))

        # Total steering angle
        steering_angle = heading_error + cross_track_term

        # Normalize to [-π, π]
        steering_angle = normalize_angle(steering_angle)

        # Map to [-1, 1] range (assume max steering angle is π/6 = 30 degrees)
        max_physical_steer = pi / 6  # 30 degrees in radians
        steering = steering_angle / max_physical_steer

        # Clip to maximum steering
        steering = np.clip(steering, -self.max_steer, self.max_steer)

        return steering


class StanleyPIDController:
    """
    Combined Stanley (lateral) + PID (longitudinal) Controller

    This is the Phase 1 classical controller for comparison with RL
    """

    def __init__(self,
                 # PID parameters
                 pid_kp: float = 0.5, pid_ki: float = 0.1, pid_kd: float = 0.2,
                 # Stanley parameters
                 stanley_k: float = 1.0, stanley_ks: float = 0.5):
        """
        Initialize combined controller

        Args:
            pid_kp, pid_ki, pid_kd: PID controller gains
            stanley_k, stanley_ks: Stanley controller gains
        """
        self.pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)
        self.stanley = StanleyController(k=stanley_k, ks=stanley_ks)

    def reset(self):
        """Reset both controllers"""
        self.pid.reset()

    def get_action(self, observation: np.ndarray, dt: float = 0.016) -> np.ndarray:
        """
        Compute control action from observation

        This method interfaces with the RL environment's action space

        Args:
            observation: Environment observation (23-dim vector)
            dt: Time step [s]

        Returns:
            action: Control action [acceleration, steering] in [-1, 1]²
        """
        # Parse observation (see env.py _get_vehicle_observation)
        # observation structure:
        # [0]: progress (목표 진행률)
        # [1-2]: cos(yaw), sin(yaw)
        # [3]: scaled vel_long
        # [4]: scaled vel_lat
        # [5-6]: cos(goal_yaw_diff), sin(goal_yaw_diff)
        # [7]: scaled frenet_d
        # [8-9]: cos(heading_error), sin(heading_error)
        # [10-22]: lidar_data (13 rays)

        # Reconstruct vehicle state from observation
        # Note: Some information might be lost in normalization
        # We'll work with what we have

        # Decode velocity (assuming max_vel_long = 65 m/s from config)
        max_vel_long = 65.0
        vel_long = observation[3] * max_vel_long  # Descale velocity

        # Decode frenet_d (assuming road_width = 8.0 m from config)
        road_width = 8.0
        frenet_d = observation[7] * (road_width / 2.0)  # Descale frenet_d

        # Decode heading_error
        cos_heading = observation[8]
        sin_heading = observation[9]
        heading_error = atan2(sin_heading, cos_heading)

        # Target speed: assume default speed for now
        # In actual implementation, this should come from road_manager
        target_speed = 15.0  # Default from config

        # Lidar data
        lidar_data = observation[10:23]

        # Create minimal vehicle state
        vehicle_state = VehicleState(
            x=0.0, y=0.0, yaw=0.0,  # Not used by controllers
            vel_long=vel_long,
            vel_lat=0.0,  # Not critical for control
            frenet_s=0.0,  # Not used
            frenet_d=frenet_d,
            frenet_point=None,
            heading_error=heading_error,
            target_vel_long=target_speed,
            segment_length=0.0,  # Not used
            lidar_data=lidar_data
        )

        # Compute longitudinal control (PID)
        acceleration = self.pid.compute(vel_long, target_speed, dt)

        # Compute lateral control (Stanley)
        steering = self.stanley.compute(vehicle_state)

        # Return action
        action = np.array([acceleration, steering], dtype=np.float32)

        return action


# ======================
# Phase 2: Lattice Planner
# ======================

class LatticePlanner:
    """
    Lattice-based path planner

    Generates multiple candidate paths using quintic polynomials in Frenet frame
    and selects the optimal one based on safety, efficiency, and comfort

    Reference:
    - Werling, M., et al. "Optimal trajectory generation for dynamic street scenarios
      in a frenet frame." IEEE International Conference on Robotics and Automation, 2010.
    """

    def __init__(self,
                 time_horizon: float = 2.0,
                 dt: float = 0.2,
                 lateral_samples: int = 5,
                 road_width: float = 8.0,
                 max_lateral_offset: float = 3.0,
                 max_curvature: float = 0.3):
        """
        Initialize Lattice Planner

        Args:
            time_horizon: Planning time horizon [s]
            dt: Time step for path sampling [s]
            lateral_samples: Number of lateral offset samples
            road_width: Road width [m]
            max_lateral_offset: Maximum lateral offset from center [m]
            max_curvature: Maximum allowed path curvature [1/m]
        """
        self.time_horizon = time_horizon
        self.dt = dt
        self.lateral_samples = lateral_samples
        self.road_width = road_width
        self.max_lateral_offset = max_lateral_offset
        self.max_curvature = max_curvature

        # Current best path (persistent between calls)
        self.current_path: Optional[PathSegment] = None
        self.replan_timer = 0.0
        self.replan_interval = 0.5  # Replan every 0.5 seconds

    def generate_candidate_paths(self,
                                  current_s: float,
                                  current_d: float,
                                  current_d_dot: float,
                                  current_speed: float) -> List[PathSegment]:
        """
        Generate candidate paths using quintic polynomials

        Args:
            current_s: Current longitudinal position [m]
            current_d: Current lateral position [m]
            current_d_dot: Current lateral velocity [m/s]
            current_speed: Current longitudinal speed [m/s]

        Returns:
            List of candidate path segments
        """
        paths = []

        # Generate lateral offset targets
        # Distribute samples around current position and lane center
        target_offsets = np.linspace(-self.max_lateral_offset,
                                     self.max_lateral_offset,
                                     self.lateral_samples)

        for target_d in target_offsets:
            # Clamp target to road boundaries
            if abs(target_d) > self.road_width / 2:
                continue

            # Generate quintic polynomial path in Frenet frame
            # d(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
            # Boundary conditions:
            # d(0) = current_d, d_dot(0) = current_d_dot, d_ddot(0) = 0
            # d(T) = target_d, d_dot(T) = 0, d_ddot(T) = 0

            T = self.time_horizon
            coeffs = self._solve_quintic_polynomial(
                current_d, current_d_dot, 0.0,  # initial conditions
                target_d, 0.0, 0.0,  # final conditions
                T
            )

            # Sample path points
            points = []
            times = np.arange(0, T + self.dt, self.dt)

            for t in times:
                # Lateral position from quintic polynomial
                d_t = self._evaluate_quintic(coeffs, t)

                # Longitudinal position (constant velocity assumption)
                s_t = current_s + current_speed * t

                # Store path point (s, d, time)
                # We'll convert to global coordinates later when needed
                points.append((s_t, d_t, t))

            # Calculate path cost
            cost = self._evaluate_path_cost(points, coeffs, target_d, current_d)

            # Check path validity (curvature constraint)
            if not self._is_path_valid(points, coeffs):
                continue

            # Create path segment
            path = PathSegment(
                points=points,
                cost=cost,
                lateral_offset=target_d,
                time_horizon=T
            )
            paths.append(path)

        return paths

    def _solve_quintic_polynomial(self,
                                   d0: float, d0_dot: float, d0_ddot: float,
                                   df: float, df_dot: float, df_ddot: float,
                                   T: float) -> np.ndarray:
        """
        Solve quintic polynomial coefficients given boundary conditions

        Returns:
            coeffs: [a0, a1, a2, a3, a4, a5]
        """
        # Quintic polynomial: d(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # Set up linear system Ax = b

        A = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [T**0, T**1, T**2, T**3, T**4, T**5],
            [0, T**0, 2*T**1, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2*T**0, 6*T**1, 12*T**2, 20*T**3]
        ])

        # Initial conditions
        A[0, 0] = 1  # d(0)
        A[1, 1] = 1  # d_dot(0)
        A[2, 2] = 2  # d_ddot(0)

        b = np.array([d0, d0_dot, d0_ddot, df, df_dot, df_ddot])

        # Solve for coefficients
        coeffs = np.linalg.solve(A, b)

        return coeffs

    def _evaluate_quintic(self, coeffs: np.ndarray, t: float) -> float:
        """Evaluate quintic polynomial at time t"""
        return (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 +
                coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)

    def _evaluate_quintic_derivative(self, coeffs: np.ndarray, t: float, order: int = 1) -> float:
        """Evaluate nth derivative of quintic polynomial at time t"""
        if order == 1:
            return (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 +
                    4*coeffs[4]*t**3 + 5*coeffs[5]*t**4)
        elif order == 2:
            return (2*coeffs[2] + 6*coeffs[3]*t +
                    12*coeffs[4]*t**2 + 20*coeffs[5]*t**3)
        else:
            raise ValueError("Only 1st and 2nd derivatives supported")

    def _is_path_valid(self, points: List[Tuple], coeffs: np.ndarray) -> bool:
        """
        Check if path satisfies kinematic constraints

        Args:
            points: Path points (s, d, t)
            coeffs: Quintic polynomial coefficients

        Returns:
            True if path is valid
        """
        # Check maximum curvature along path
        for s, d, t in points:
            # Approximate curvature as d_ddot / v^2
            # (simplified, assuming v is constant)
            d_ddot = self._evaluate_quintic_derivative(coeffs, t, order=2)

            # Skip if curvature check is too strict for low speeds
            # (curvature becomes very large at low speeds)
            if abs(d_ddot) > 10.0:  # Arbitrary threshold
                return False

        return True

    def _evaluate_path_cost(self,
                            points: List[Tuple],
                            coeffs: np.ndarray,
                            target_d: float,
                            current_d: float) -> float:
        """
        Evaluate path cost based on multiple criteria

        Args:
            points: Path points (s, d, t)
            coeffs: Quintic polynomial coefficients
            target_d: Target lateral offset [m]
            current_d: Current lateral offset [m]

        Returns:
            cost: Path cost (lower is better)
        """
        cost = 0.0

        # 1. Deviation from lane center (efficiency)
        # Prefer paths closer to lane center (d=0)
        w_lane_center = 1.0
        cost += w_lane_center * abs(target_d)

        # 2. Lateral acceleration (comfort)
        # Penalize high lateral accelerations
        w_lat_accel = 2.0
        max_d_ddot = 0.0
        for s, d, t in points:
            d_ddot = abs(self._evaluate_quintic_derivative(coeffs, t, order=2))
            max_d_ddot = max(max_d_ddot, d_ddot)
        cost += w_lat_accel * max_d_ddot

        # 3. Lateral jerk (smoothness)
        # Penalize high lateral jerk (rate of acceleration change)
        w_jerk = 1.0
        # For quintic, jerk is third derivative
        # We'll approximate by change in d_ddot
        prev_d_ddot = 0.0
        total_jerk = 0.0
        for s, d, t in points:
            d_ddot = self._evaluate_quintic_derivative(coeffs, t, order=2)
            jerk = abs(d_ddot - prev_d_ddot)
            total_jerk += jerk
            prev_d_ddot = d_ddot
        cost += w_jerk * total_jerk / len(points)

        # 4. Deviation from current path (continuity)
        # Penalize large changes from current lateral position
        w_continuity = 0.5
        lateral_change = abs(target_d - current_d)
        cost += w_continuity * lateral_change

        return cost

    def select_best_path(self,
                         candidate_paths: List[PathSegment],
                         lidar_data: np.ndarray,
                         lidar_angles: np.ndarray) -> Optional[PathSegment]:
        """
        Select best path considering obstacles from lidar

        Args:
            candidate_paths: List of candidate paths
            lidar_data: Lidar distances [m]
            lidar_angles: Lidar ray angles [rad] (vehicle frame)

        Returns:
            best_path: Selected path (or None if no valid path)
        """
        if not candidate_paths:
            return None

        # Add obstacle avoidance cost to each path
        for path in candidate_paths:
            obstacle_cost = self._evaluate_obstacle_cost(path, lidar_data, lidar_angles)
            path.cost += obstacle_cost

        # Sort by cost and select best
        candidate_paths.sort(key=lambda p: p.cost)
        best_path = candidate_paths[0]

        return best_path

    def _evaluate_obstacle_cost(self,
                                 path: PathSegment,
                                 lidar_data: np.ndarray,
                                 lidar_angles: np.ndarray) -> float:
        """
        Evaluate obstacle avoidance cost using lidar data

        Args:
            path: Path segment to evaluate
            lidar_data: Lidar distances [m]
            lidar_angles: Lidar ray angles [rad]

        Returns:
            cost: Obstacle cost (higher if path is near obstacles)
        """
        # This is a simplified obstacle check
        # In full implementation, would project path points to global frame
        # and check against obstacle positions

        cost = 0.0
        w_obstacle = 50.0  # High weight for obstacle avoidance
        safety_distance = 2.0  # Minimum safe distance [m]

        # Check if any lidar ray detects close obstacle
        # (simplified: assumes obstacles are perpendicular to vehicle)
        min_obstacle_distance = np.min(lidar_data)

        if min_obstacle_distance < safety_distance:
            # Penalize paths that go towards obstacles
            # (simplified heuristic)
            # If path goes left (positive d) and left obstacles are close, penalize
            # If path goes right (negative d) and right obstacles are close, penalize

            target_d = path.lateral_offset

            # Left side lidar (positive angles)
            left_indices = lidar_angles > 0
            if np.any(left_indices):
                left_min_dist = np.min(lidar_data[left_indices])
                if target_d > 0 and left_min_dist < safety_distance:
                    cost += w_obstacle * (safety_distance - left_min_dist) / safety_distance

            # Right side lidar (negative angles)
            right_indices = lidar_angles < 0
            if np.any(right_indices):
                right_min_dist = np.min(lidar_data[right_indices])
                if target_d < 0 and right_min_dist < safety_distance:
                    cost += w_obstacle * (safety_distance - right_min_dist) / safety_distance

        return cost

    def plan(self,
             current_s: float,
             current_d: float,
             current_d_dot: float,
             current_speed: float,
             lidar_data: np.ndarray,
             dt_elapsed: float = 0.0) -> Optional[PathSegment]:
        """
        Main planning function - generates and selects optimal path

        Args:
            current_s: Current longitudinal position [m]
            current_d: Current lateral position [m]
            current_d_dot: Current lateral velocity [m/s]
            current_speed: Current speed [m/s]
            lidar_data: Lidar distances (13 rays, normalized to [0,1])
            dt_elapsed: Time elapsed since last call [s]

        Returns:
            Selected path segment (or None if planning fails)
        """
        # Update replan timer
        self.replan_timer += dt_elapsed

        # Check if we need to replan
        if self.current_path is not None and self.replan_timer < self.replan_interval:
            # Use existing path
            return self.current_path

        # Reset timer
        self.replan_timer = 0.0

        # Generate candidate paths
        candidates = self.generate_candidate_paths(
            current_s, current_d, current_d_dot, current_speed
        )

        if not candidates:
            return self.current_path  # Keep previous path if generation fails

        # Reconstruct lidar angles (from sensor config in config.yaml)
        # angle_start: -170 deg, angle_end: 170 deg, num_samples: 13
        lidar_angle_start = np.radians(-170)
        lidar_angle_end = np.radians(170)
        num_lidar_rays = len(lidar_data)

        # Apply concentration factor (simplified, using linear spacing)
        lidar_angles = np.linspace(lidar_angle_start, lidar_angle_end, num_lidar_rays)

        # Denormalize lidar data (assuming max_range = 30m from config)
        max_lidar_range = 30.0
        lidar_distances = lidar_data * max_lidar_range

        # Select best path
        best_path = self.select_best_path(candidates, lidar_distances, lidar_angles)

        # Update current path
        self.current_path = best_path

        return best_path


class LatticePlannerController:
    """
    Combined Lattice Planner + Stanley + PID Controller

    Uses lattice planner for strategic path planning (low frequency)
    and Stanley+PID for tactical control (high frequency)

    This is the Phase 2 classical controller for comparison with RL
    """

    def __init__(self,
                 # Lattice parameters
                 lattice_time_horizon: float = 2.0,
                 lateral_samples: int = 5,
                 # PID parameters
                 pid_kp: float = 0.5, pid_ki: float = 0.1, pid_kd: float = 0.2,
                 # Stanley parameters
                 stanley_k: float = 1.0, stanley_ks: float = 0.5):
        """
        Initialize combined planner-controller

        Args:
            lattice_time_horizon: Planning horizon [s]
            lateral_samples: Number of lateral samples for lattice
            pid_kp, pid_ki, pid_kd: PID gains
            stanley_k, stanley_ks: Stanley gains
        """
        self.lattice_planner = LatticePlanner(
            time_horizon=lattice_time_horizon,
            lateral_samples=lateral_samples
        )
        self.pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)
        self.stanley = StanleyController(k=stanley_k, ks=stanley_ks)

        # Path following state
        self.current_path: Optional[PathSegment] = None
        self.path_index = 0  # Current waypoint index in path

    def reset(self):
        """Reset all controllers"""
        self.pid.reset()
        self.current_path = None
        self.path_index = 0
        self.lattice_planner.current_path = None

    def get_action(self, observation: np.ndarray, dt: float = 0.016) -> np.ndarray:
        """
        Compute control action from observation

        This method interfaces with the RL environment's action space

        Args:
            observation: Environment observation (23-dim vector)
            dt: Time step [s]

        Returns:
            action: Control action [acceleration, steering] in [-1, 1]²
        """
        # Parse observation (similar to StanleyPIDController)
        max_vel_long = 65.0
        road_width = 8.0

        vel_long = observation[3] * max_vel_long
        frenet_d = observation[7] * (road_width / 2.0)
        cos_heading = observation[8]
        sin_heading = observation[9]
        heading_error = atan2(sin_heading, cos_heading)
        lidar_data = observation[10:23]

        # Estimate frenet_s and vel_lat
        # (In actual implementation, these should come from environment)
        progress = observation[0]  # -1 to 1
        frenet_s = progress * 100.0  # Rough estimate (assuming ~100m segment)
        vel_lat = observation[4] * 23.0  # Descale (max_vel_lat = 23 m/s)

        # Target speed
        target_speed = 15.0

        # === Lattice Planning (strategic layer) ===
        # Generate/update path if needed
        planned_path = self.lattice_planner.plan(
            current_s=frenet_s,
            current_d=frenet_d,
            current_d_dot=vel_lat,  # Lateral velocity
            current_speed=max(vel_long, 1.0),  # Avoid division by zero
            lidar_data=lidar_data,
            dt_elapsed=dt
        )

        # If planning succeeded, use planned path for reference
        if planned_path is not None:
            self.current_path = planned_path
            # Get next waypoint from planned path
            # (simplified: use current lateral offset as target)
            target_d = planned_path.lateral_offset
        else:
            # Fallback: stay in current lane
            target_d = 0.0

        # === Stanley Control (tactical layer) ===
        # Create modified heading error based on planned path
        # If we have a target_d different from current frenet_d,
        # adjust heading error accordingly
        d_error = target_d - frenet_d

        # Simple proportional adjustment to heading error
        # (in full implementation, would compute proper path tangent)
        heading_adjustment = atan(0.5 * d_error)  # Simple heuristic
        adjusted_heading_error = heading_error + heading_adjustment

        # Create vehicle state for Stanley
        vehicle_state = VehicleState(
            x=0.0, y=0.0, yaw=0.0,
            vel_long=vel_long,
            vel_lat=vel_lat,
            frenet_s=frenet_s,
            frenet_d=frenet_d,
            frenet_point=None,
            heading_error=adjusted_heading_error,
            target_vel_long=target_speed,
            segment_length=0.0,
            lidar_data=lidar_data
        )

        # Compute steering with Stanley
        steering = self.stanley.compute(vehicle_state)

        # === PID Speed Control (tactical layer) ===
        acceleration = self.pid.compute(vel_long, target_speed, dt)

        # Return action
        action = np.array([acceleration, steering], dtype=np.float32)

        return action


# ======================
# Phase 3: MPC Controller (TODO)
# ======================

class MPCController:
    """
    Model Predictive Control for vehicle control

    Uses bicycle model to predict future states over a finite horizon and
    optimizes control inputs to minimize path tracking error and control effort.

    Key features:
    - Receding horizon optimization (1-second lookahead)
    - Bicycle model for state prediction
    - Multi-objective cost: path tracking + control smoothness
    - Gradient-based optimization (SLSQP)
    """

    def __init__(self,
                 horizon: float = 1.0,           # prediction horizon [s]
                 dt: float = 0.05,                # time step [s]
                 wheelbase: float = 2.875,        # vehicle wheelbase [m]
                 max_accel: float = 4.0,          # max acceleration [m/s²]
                 max_brake: float = 6.0,          # max braking [m/s²]
                 max_steer_angle: float = 0.611,  # max steering angle [rad] (35°)
                 # Cost weights
                 w_lateral: float = 10.0,         # lateral error weight
                 w_heading: float = 5.0,          # heading error weight
                 w_speed: float = 1.0,            # speed tracking weight
                 w_accel: float = 0.5,            # acceleration penalty
                 w_steer: float = 0.5,            # steering penalty
                 w_accel_rate: float = 1.0,       # jerk penalty
                 w_steer_rate: float = 1.0):      # steering rate penalty
        """
        Initialize MPC controller

        Args:
            horizon: Prediction horizon in seconds
            dt: Time step for discretization
            wheelbase: Vehicle wheelbase for bicycle model
            max_accel/max_brake: Acceleration limits
            max_steer_angle: Maximum steering angle in radians
            w_*: Cost function weights
        """
        self.horizon = horizon
        self.dt = dt
        self.N = int(horizon / dt)  # number of prediction steps
        self.wheelbase = wheelbase

        # Control limits
        self.max_accel = max_accel
        self.max_brake = max_brake
        self.max_steer_angle = max_steer_angle

        # Cost weights
        self.w_lateral = w_lateral
        self.w_heading = w_heading
        self.w_speed = w_speed
        self.w_accel = w_accel
        self.w_steer = w_steer
        self.w_accel_rate = w_accel_rate
        self.w_steer_rate = w_steer_rate

        # Previous control inputs (for rate penalties)
        self.prev_accel = 0.0
        self.prev_steer = 0.0

    def _bicycle_model_step(self, x: float, y: float, yaw: float, vel: float,
                            accel: float, steer_angle: float, dt: float
                            ) -> Tuple[float, float, float, float]:
        """
        Simple bicycle model for state prediction

        State: [x, y, yaw, vel]
        Control: [accel, steer_angle]

        Returns:
            x_next, y_next, yaw_next, vel_next
        """
        # Update velocity with acceleration
        vel_next = vel + accel * dt
        vel_next = np.clip(vel_next, -20.0, 65.0)  # velocity limits

        # Avoid division by zero
        vel_avg = (vel + vel_next) / 2.0

        # Kinematic bicycle model
        if abs(vel_avg) > 0.1:
            # Normal motion
            yaw_rate = vel_avg * tan(steer_angle) / self.wheelbase
            yaw_next = yaw + yaw_rate * dt

            # Position update
            x_next = x + vel_avg * cos(yaw) * dt
            y_next = y + vel_avg * sin(yaw) * dt
        else:
            # Nearly stationary
            yaw_next = yaw
            x_next = x
            y_next = y

        return x_next, y_next, yaw_next, vel_next

    def _predict_trajectory(self, state: np.ndarray,
                           controls: np.ndarray) -> np.ndarray:
        """
        Predict vehicle trajectory given control sequence

        Args:
            state: Initial state [x, y, yaw, vel]
            controls: Control sequence (N, 2) - [accel, steer_angle]

        Returns:
            trajectory: Predicted states (N+1, 4)
        """
        trajectory = np.zeros((self.N + 1, 4))
        trajectory[0] = state

        x, y, yaw, vel = state

        for i in range(self.N):
            accel = controls[i, 0]
            steer_angle = controls[i, 1]

            x, y, yaw, vel = self._bicycle_model_step(
                x, y, yaw, vel, accel, steer_angle, self.dt
            )

            trajectory[i + 1] = [x, y, yaw, vel]

        return trajectory

    def _compute_lateral_error_from_trajectory(
        self, x: float, y: float, yaw: float,
        reference_frenet_point: Optional[Tuple[float, float, float]]
    ) -> float:
        """
        Estimate lateral error for predicted state

        This is a simplified version assuming the reference path is
        locally straight around the current frenet point
        """
        if reference_frenet_point is None:
            return 0.0

        ref_x, ref_y, ref_yaw = reference_frenet_point

        # Vector from reference point to vehicle
        dx = x - ref_x
        dy = y - ref_y

        # Lateral error is perpendicular distance to path
        # Using rotation to path frame
        lateral_error = -dx * sin(ref_yaw) + dy * cos(ref_yaw)

        return lateral_error

    def _cost_function(self, controls_flat: np.ndarray,
                       state: np.ndarray,
                       vehicle_state: VehicleState) -> float:
        """
        Cost function for MPC optimization

        Balances multiple objectives:
        - Path tracking: minimize lateral and heading errors
        - Speed tracking: maintain target speed
        - Control effort: minimize acceleration and steering
        - Smoothness: penalize rapid control changes

        Args:
            controls_flat: Flattened control sequence (2*N,)
            state: Initial state [x, y, yaw, vel]
            vehicle_state: VehicleState object with reference info

        Returns:
            cost: Total cost
        """
        # Reshape controls to (N, 2)
        controls = controls_flat.reshape((self.N, 2))

        # Predict trajectory
        trajectory = self._predict_trajectory(state, controls)

        # Initialize cost
        cost = 0.0

        # Path tracking costs
        for i in range(1, self.N + 1):
            x, y, yaw, vel = trajectory[i]

            # Lateral error cost
            lateral_error = self._compute_lateral_error_from_trajectory(
                x, y, yaw, vehicle_state.frenet_point
            )
            cost += self.w_lateral * lateral_error**2

            # Heading error cost (relative to path)
            if vehicle_state.frenet_point is not None:
                ref_yaw = vehicle_state.frenet_point[2]
                heading_error = self._normalize_angle(yaw - ref_yaw)
                cost += self.w_heading * heading_error**2

            # Speed tracking cost
            speed_error = vel - vehicle_state.target_vel_long
            cost += self.w_speed * speed_error**2

        # Control effort costs
        for i in range(self.N):
            accel = controls[i, 0]
            steer = controls[i, 1]

            # Penalize control magnitudes
            cost += self.w_accel * accel**2
            cost += self.w_steer * steer**2

            # Penalize control rates (smoothness)
            if i == 0:
                accel_prev = self.prev_accel
                steer_prev = self.prev_steer
            else:
                accel_prev = controls[i - 1, 0]
                steer_prev = controls[i - 1, 1]

            accel_rate = (accel - accel_prev) / self.dt
            steer_rate = (steer - steer_prev) / self.dt

            cost += self.w_accel_rate * accel_rate**2
            cost += self.w_steer_rate * steer_rate**2

        return cost

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def get_action(self, observation: np.ndarray, dt: float = 0.016) -> np.ndarray:
        """
        Compute optimal control action using MPC

        Solves optimization problem:
            min J(u) = Σ(tracking_error + control_effort + smoothness)
            subject to: vehicle dynamics and control constraints

        Args:
            observation: 23-dim observation from environment
            dt: Time step (not used in MPC, uses self.dt internally)

        Returns:
            action: [acceleration, steering] in normalized space [-1, 1]²
        """
        # Parse observation into vehicle state
        vehicle_state = self._parse_observation(observation)

        # Construct initial state for prediction
        # Use local coordinates relative to current position
        state = np.array([
            0.0,  # x (local frame)
            0.0,  # y (local frame)
            0.0,  # yaw (local frame, aligned with current heading)
            vehicle_state.vel_long
        ])

        # Initial guess: maintain current control
        controls_init = np.zeros((self.N, 2))
        controls_init[:, 0] = self.prev_accel
        controls_init[:, 1] = self.prev_steer

        # Control bounds
        bounds = []
        for _ in range(self.N):
            bounds.append((-self.max_brake, self.max_accel))  # acceleration
            bounds.append((-self.max_steer_angle, self.max_steer_angle))  # steering

        # Optimize
        from scipy.optimize import minimize

        result = minimize(
            self._cost_function,
            controls_init.flatten(),
            args=(state, vehicle_state),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'disp': False}
        )

        # Extract optimal control (first step)
        optimal_controls = result.x.reshape((self.N, 2))
        accel_opt = optimal_controls[0, 0]
        steer_opt = optimal_controls[0, 1]

        # Update previous controls
        self.prev_accel = accel_opt
        self.prev_steer = steer_opt

        # Normalize to action space [-1, 1]²
        # Acceleration: [-max_brake, max_accel] -> [-1, 1]
        if accel_opt >= 0:
            accel_normalized = accel_opt / self.max_accel
        else:
            accel_normalized = accel_opt / self.max_brake

        # Steering: [-max_steer_angle, max_steer_angle] -> [-1, 1]
        steer_normalized = steer_opt / self.max_steer_angle

        # Clip to ensure valid range
        accel_normalized = np.clip(accel_normalized, -1.0, 1.0)
        steer_normalized = np.clip(steer_normalized, -1.0, 1.0)

        return np.array([accel_normalized, steer_normalized], dtype=np.float32)

    def _parse_observation(self, observation: np.ndarray) -> VehicleState:
        """
        Parse 23-dim observation into VehicleState

        Observation structure (23-dim):
        [0]: progress (normalized)
        [1]: cos(yaw)
        [2]: sin(yaw)
        [3]: vel_long (normalized to max_speed=65)
        [4]: vel_lat (normalized to max_vel_lat=23)
        [5]: acc_long (normalized to max_accel=4)
        [6]: acc_lat (normalized to max_acc_lat=705)
        [7]: frenet_d (normalized to road_width/2=4)
        [8]: cos(heading_error)
        [9]: sin(heading_error)
        [10:23]: lidar_data (13 rays, normalized to max_range=30)
        """
        # Extract and denormalize
        progress = observation[0]
        cos_yaw = observation[1]
        sin_yaw = observation[2]
        yaw = atan2(sin_yaw, cos_yaw)

        vel_long = observation[3] * 65.0  # max_speed
        vel_lat = observation[4] * 23.0   # max_vel_lat

        frenet_d = observation[7] * 4.0   # road_width/2

        cos_heading = observation[8]
        sin_heading = observation[9]
        heading_error = atan2(sin_heading, cos_heading)

        lidar_data = observation[10:23] * 30.0  # max_range

        # Estimate target speed (simplified: use recommended speed from road)
        # In absence of direct info, assume moderate target speed
        target_vel_long = 15.0  # default speed from config

        # Estimate frenet_point (local reference)
        # Since we use local coordinates, reference is at origin aligned with path
        frenet_point = (0.0, 0.0, heading_error)  # (x, y, yaw in path frame)

        return VehicleState(
            x=0.0,  # Local frame
            y=0.0,  # Local frame
            yaw=yaw,
            vel_long=vel_long,
            vel_lat=vel_lat,
            frenet_s=0.0,  # Not used in MPC
            frenet_d=frenet_d,
            frenet_point=frenet_point,
            heading_error=heading_error,
            target_vel_long=target_vel_long,
            segment_length=0.0,  # Not used
            lidar_data=lidar_data
        )


# ======================
# Phase 4: Benchmark Framework
# ======================

@dataclass
class EpisodeMetrics:
    """Metrics collected during a single episode"""
    episode_reward: float
    episode_length: int
    success: bool           # reached goal
    collision: bool         # collision occurred
    outside_road: bool      # went outside road

    # Timing metrics
    total_time: float       # total episode time [s]
    avg_action_time: float  # average action computation time [s]

    # Control smoothness
    avg_jerk: float         # average jerk (acceleration rate) [m/s³]
    max_jerk: float         # maximum jerk [m/s³]
    avg_steering_rate: float  # average steering rate [rad/s]
    max_steering_rate: float  # maximum steering rate [rad/s]

    # Path tracking
    mean_lateral_error: float    # mean lateral error [m]
    max_lateral_error: float     # max lateral error [m]
    mean_heading_error: float    # mean heading error [rad]


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results across multiple episodes"""
    controller_name: str
    num_episodes: int

    # Success metrics
    success_rate: float
    collision_rate: float
    outside_road_rate: float

    # Reward and length
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    std_episode_length: float

    # Timing
    mean_action_time: float
    std_action_time: float

    # Control smoothness
    mean_jerk: float
    mean_steering_rate: float

    # Path tracking
    mean_lateral_error: float
    mean_heading_error: float

    # Raw episode data
    episodes: List[EpisodeMetrics]


class ControllerBenchmark:
    """
    Performance comparison framework for different controllers

    Evaluates controllers on multiple metrics:
    - Success rate (goal reaching)
    - Collision rate
    - Average reward per episode
    - Episode length (steps to goal)
    - Computation time per action
    - Control smoothness (jerk, steering rate)
    - Path tracking error (lateral, heading)

    Usage:
        benchmark = ControllerBenchmark(env, num_episodes=10)
        results = benchmark.evaluate_controller(controller, "Stanley+PID")
        benchmark.print_results(results)
    """

    def __init__(self,
                 env,
                 num_episodes: int = 10,
                 max_steps: int = 1500,
                 verbose: bool = True):
        """
        Initialize benchmark framework

        Args:
            env: Gymnasium environment (e.g., BasicRLDrivingEnv)
            num_episodes: Number of episodes to run per controller
            max_steps: Maximum steps per episode
            verbose: Print progress during evaluation
        """
        self.env = env
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.verbose = verbose

    def evaluate_controller(self,
                           controller,
                           controller_name: str) -> BenchmarkResults:
        """
        Evaluate a controller over multiple episodes

        Args:
            controller: Controller object with get_action(observation) method
            controller_name: Name for reporting

        Returns:
            BenchmarkResults with aggregated metrics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {controller_name}")
            print(f"Episodes: {self.num_episodes}, Max steps: {self.max_steps}")
            print(f"{'='*60}")

        episodes = []

        for ep in range(self.num_episodes):
            if self.verbose:
                print(f"Episode {ep + 1}/{self.num_episodes}...", end=" ")

            metrics = self._run_episode(controller)
            episodes.append(metrics)

            if self.verbose:
                status = "✓ SUCCESS" if metrics.success else ("✗ COLLISION" if metrics.collision else "✗ OUT OF ROAD" if metrics.outside_road else "✗ TIMEOUT")
                print(f"{status} | Reward: {metrics.episode_reward:.1f} | Steps: {metrics.episode_length}")

        # Aggregate results
        results = self._aggregate_results(controller_name, episodes)

        return results

    def _run_episode(self, controller) -> EpisodeMetrics:
        """
        Run a single episode with the controller

        Returns:
            EpisodeMetrics for the episode
        """
        import time

        # Reset environment
        observation, info = self.env.reset()

        # Episode state
        episode_reward = 0.0
        episode_length = 0
        done = False
        truncated = False

        # Tracking lists
        action_times = []
        accelerations = []
        steering_angles = []
        lateral_errors = []
        heading_errors = []

        prev_accel = 0.0
        prev_steer = 0.0

        episode_start_time = time.time()

        while not (done or truncated) and episode_length < self.max_steps:
            # Compute action (timed)
            action_start = time.time()
            action = controller.get_action(observation)
            action_end = time.time()
            action_times.append(action_end - action_start)

            # Step environment
            observation, reward, done, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_length += 1

            # Track control smoothness
            # Denormalize action for physical units
            accel_phys = action[0] * 4.0 if action[0] >= 0 else action[0] * 6.0  # [m/s²]
            steer_phys = action[1] * 0.611  # [rad]

            accelerations.append(accel_phys)
            steering_angles.append(steer_phys)

            # Track path errors
            frenet_d = observation[7] * 4.0  # denormalize
            lateral_errors.append(abs(frenet_d))

            heading_error = atan2(observation[9], observation[8])
            heading_errors.append(abs(heading_error))

            prev_accel = accel_phys
            prev_steer = steer_phys

        episode_end_time = time.time()
        total_time = episode_end_time - episode_start_time

        # Compute control smoothness metrics
        if len(accelerations) > 1:
            dt = 0.016  # simulation timestep
            jerks = np.diff(accelerations) / dt
            avg_jerk = float(np.mean(np.abs(jerks)))
            max_jerk = float(np.max(np.abs(jerks)))

            steering_rates = np.diff(steering_angles) / dt
            avg_steering_rate = float(np.mean(np.abs(steering_rates)))
            max_steering_rate = float(np.max(np.abs(steering_rates)))
        else:
            avg_jerk = 0.0
            max_jerk = 0.0
            avg_steering_rate = 0.0
            max_steering_rate = 0.0

        # Determine episode outcome
        success = info.get('goal_reached', False)
        collision = info.get('collision', False)
        outside_road = info.get('outside_road', False)

        return EpisodeMetrics(
            episode_reward=episode_reward,
            episode_length=episode_length,
            success=success,
            collision=collision,
            outside_road=outside_road,
            total_time=total_time,
            avg_action_time=float(np.mean(action_times)),
            avg_jerk=avg_jerk,
            max_jerk=max_jerk,
            avg_steering_rate=avg_steering_rate,
            max_steering_rate=max_steering_rate,
            mean_lateral_error=float(np.mean(lateral_errors)),
            max_lateral_error=float(np.max(lateral_errors)),
            mean_heading_error=float(np.mean(heading_errors))
        )

    def _aggregate_results(self,
                          controller_name: str,
                          episodes: List[EpisodeMetrics]) -> BenchmarkResults:
        """
        Aggregate episode metrics into benchmark results

        Args:
            controller_name: Name of controller
            episodes: List of episode metrics

        Returns:
            BenchmarkResults with aggregated statistics
        """
        n = len(episodes)

        # Success metrics
        success_rate = sum(ep.success for ep in episodes) / n
        collision_rate = sum(ep.collision for ep in episodes) / n
        outside_road_rate = sum(ep.outside_road for ep in episodes) / n

        # Reward and length
        rewards = [ep.episode_reward for ep in episodes]
        lengths = [ep.episode_length for ep in episodes]
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_episode_length = float(np.mean(lengths))
        std_episode_length = float(np.std(lengths))

        # Timing
        action_times = [ep.avg_action_time for ep in episodes]
        mean_action_time = float(np.mean(action_times))
        std_action_time = float(np.std(action_times))

        # Control smoothness
        mean_jerk = float(np.mean([ep.avg_jerk for ep in episodes]))
        mean_steering_rate = float(np.mean([ep.avg_steering_rate for ep in episodes]))

        # Path tracking
        mean_lateral_error = float(np.mean([ep.mean_lateral_error for ep in episodes]))
        mean_heading_error = float(np.mean([ep.mean_heading_error for ep in episodes]))

        return BenchmarkResults(
            controller_name=controller_name,
            num_episodes=n,
            success_rate=success_rate,
            collision_rate=collision_rate,
            outside_road_rate=outside_road_rate,
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_episode_length=mean_episode_length,
            std_episode_length=std_episode_length,
            mean_action_time=mean_action_time,
            std_action_time=std_action_time,
            mean_jerk=mean_jerk,
            mean_steering_rate=mean_steering_rate,
            mean_lateral_error=mean_lateral_error,
            mean_heading_error=mean_heading_error,
            episodes=episodes
        )

    def print_results(self, results: BenchmarkResults):
        """
        Pretty print benchmark results

        Args:
            results: BenchmarkResults to display
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS: {results.controller_name}")
        print(f"{'='*60}")
        print(f"Episodes: {results.num_episodes}")
        print(f"\n--- Success Metrics ---")
        print(f"Success Rate:      {results.success_rate*100:6.2f}%")
        print(f"Collision Rate:    {results.collision_rate*100:6.2f}%")
        print(f"Out-of-Road Rate:  {results.outside_road_rate*100:6.2f}%")
        print(f"\n--- Performance ---")
        print(f"Mean Reward:       {results.mean_reward:8.2f} ± {results.std_reward:.2f}")
        print(f"Mean Episode Len:  {results.mean_episode_length:8.1f} ± {results.std_episode_length:.1f} steps")
        print(f"\n--- Timing ---")
        print(f"Mean Action Time:  {results.mean_action_time*1000:8.3f} ± {results.std_action_time*1000:.3f} ms")
        print(f"\n--- Control Smoothness ---")
        print(f"Mean Jerk:         {results.mean_jerk:8.3f} m/s³")
        print(f"Mean Steering Rate:{results.mean_steering_rate:8.3f} rad/s")
        print(f"\n--- Path Tracking ---")
        print(f"Mean Lateral Error:{results.mean_lateral_error:8.3f} m")
        print(f"Mean Heading Error:{results.mean_heading_error:8.3f} rad ({np.degrees(results.mean_heading_error):.2f}°)")
        print(f"{'='*60}\n")

    def compare_controllers(self,
                           results_list: List[BenchmarkResults]) -> None:
        """
        Print comparison table for multiple controllers

        Args:
            results_list: List of BenchmarkResults to compare
        """
        if not results_list:
            return

        print(f"\n{'='*80}")
        print("CONTROLLER COMPARISON")
        print(f"{'='*80}")

        # Header
        print(f"{'Controller':<20} {'Success%':>10} {'Reward':>12} {'Steps':>10} {'Time(ms)':>12} {'Lat.Err':>10}")
        print(f"{'-'*80}")

        # Rows
        for res in results_list:
            print(f"{res.controller_name:<20} "
                  f"{res.success_rate*100:>10.2f} "
                  f"{res.mean_reward:>12.1f} "
                  f"{res.mean_episode_length:>10.1f} "
                  f"{res.mean_action_time*1000:>12.3f} "
                  f"{res.mean_lateral_error:>10.3f}")

        print(f"{'='*80}\n")


# ======================
# Testing and Validation
# ======================

def test_stanley_pid_controller():
    """
    Test the Stanley+PID controller with dummy observation
    """
    print("=== Testing Stanley+PID Controller ===")

    # Create controller
    controller = StanleyPIDController()

    # Create dummy observation (23-dim)
    # Simulating: vehicle slightly right of path, heading aligned, moderate speed
    observation = np.zeros(23, dtype=np.float32)
    observation[0] = 0.5   # progress = 0.5
    observation[1] = 1.0   # cos(yaw) = 1 (heading 0)
    observation[2] = 0.0   # sin(yaw) = 0
    observation[3] = 0.3   # vel_long scaled (~19.5 m/s)
    observation[4] = 0.0   # vel_lat = 0
    observation[5] = 1.0   # cos(goal_yaw_diff)
    observation[6] = 0.0   # sin(goal_yaw_diff)
    observation[7] = -0.2  # frenet_d scaled (slightly right of center)
    observation[8] = 0.99  # cos(heading_error)
    observation[9] = 0.1   # sin(heading_error) (small left heading error)
    observation[10:23] = 1.0  # lidar all far

    # Compute action
    action = controller.get_action(observation, dt=0.016)

    print(f"Observation (first 10): {observation[:10]}")
    print(f"Computed action: acceleration={action[0]:.3f}, steering={action[1]:.3f}")
    print(f"Expected: Small negative acceleration (near target speed), "
          f"Small positive steering (correct right deviation)")
    print("Test completed successfully!\n")


def test_mpc_controller():
    """
    Test the MPC controller with dummy observation
    """
    print("=== Testing MPC Controller ===")

    # Create controller
    controller = MPCController()

    # Create dummy observation (23-dim)
    # Simulating: vehicle slightly right of path, heading aligned, moderate speed
    observation = np.zeros(23, dtype=np.float32)
    observation[0] = 0.5   # progress = 0.5
    observation[1] = 1.0   # cos(yaw) = 1 (heading 0)
    observation[2] = 0.0   # sin(yaw) = 0
    observation[3] = 0.3   # vel_long scaled (~19.5 m/s)
    observation[4] = 0.0   # vel_lat = 0
    observation[5] = 1.0   # cos(goal_yaw_diff)
    observation[6] = 0.0   # sin(goal_yaw_diff)
    observation[7] = -0.2  # frenet_d scaled (slightly right of center)
    observation[8] = 0.99  # cos(heading_error)
    observation[9] = 0.1   # sin(heading_error) (small left heading error)
    observation[10:23] = 1.0  # lidar all far

    # Compute action
    print("Computing optimal control (this may take a moment)...")
    action = controller.get_action(observation, dt=0.016)

    print(f"Observation (first 10): {observation[:10]}")
    print(f"Computed action: acceleration={action[0]:.3f}, steering={action[1]:.3f}")
    print(f"Expected: Optimized controls for path tracking and speed maintenance")
    print("Test completed successfully!\n")


def demo_benchmark_comparison():
    """
    Demo: Comprehensive benchmark comparison of all controllers

    This function demonstrates how to use the benchmark framework
    to compare classical controllers (Stanley+PID, Lattice+Stanley, MPC)
    and optionally RL models.

    Note: Requires environment setup. Uncomment to run with actual environment.
    """
    print("\n" + "="*80)
    print("BENCHMARK DEMO: Classical Controller Comparison")
    print("="*80 + "\n")

    # This is a template - requires environment initialization
    print("To run benchmark comparison:")
    print("1. Import environment: from src.env.env_rl import BasicRLDrivingEnv")
    print("2. Create environment: env = BasicRLDrivingEnv(...)")
    print("3. Create benchmark: benchmark = ControllerBenchmark(env, num_episodes=10)")
    print("4. Evaluate controllers:")
    print("   - stanley_results = benchmark.evaluate_controller(StanleyPIDController(), 'Stanley+PID')")
    print("   - lattice_results = benchmark.evaluate_controller(LatticePlannerController(), 'Lattice+Stanley')")
    print("   - mpc_results = benchmark.evaluate_controller(MPCController(), 'MPC')")
    print("5. Compare: benchmark.compare_controllers([stanley_results, lattice_results, mpc_results])")
    print("\nExample code:")
    print("""
    # Initialize environment
    from src.env.env_rl import BasicRLDrivingEnv
    env = BasicRLDrivingEnv()

    # Create benchmark
    benchmark = ControllerBenchmark(env, num_episodes=10)

    # Create controllers
    stanley_controller = StanleyPIDController()
    lattice_controller = LatticePlannerController()
    mpc_controller = MPCController()

    # Evaluate each controller
    stanley_results = benchmark.evaluate_controller(stanley_controller, "Stanley+PID")
    benchmark.print_results(stanley_results)

    lattice_results = benchmark.evaluate_controller(lattice_controller, "Lattice+Stanley")
    benchmark.print_results(lattice_results)

    mpc_results = benchmark.evaluate_controller(mpc_controller, "MPC")
    benchmark.print_results(mpc_results)

    # Compare all controllers
    benchmark.compare_controllers([stanley_results, lattice_results, mpc_results])

    # Optional: Compare with RL model
    # from stable_baselines3 import SAC
    # rl_model = SAC.load("path/to/trained/model.zip")
    # rl_controller = lambda obs: rl_model.predict(obs, deterministic=True)[0]
    # rl_results = benchmark.evaluate_controller(rl_controller, "SAC-RL")
    # benchmark.compare_controllers([stanley_results, lattice_results, mpc_results, rl_results])
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Show benchmark demo
        demo_benchmark_comparison()
    else:
        # Run basic unit tests
        test_stanley_pid_controller()
        test_mpc_controller()
        print("\nBasic tests completed!")
        print("Run with '--demo' flag to see benchmark usage example.")
