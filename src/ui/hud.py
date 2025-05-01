# -*- coding: utf-8 -*-
import pygame
import numpy as np

class HUD:
    """Head-Up Display 관리 클래스"""

    def __init__(self, config):
        """
        HUD 초기화

        Args:
            config: 시뮬레이터 설정 정보
        """
        self.config = config
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)  # 큰 글꼴 추가

        # G-force 표시기 초기화
        self.g_force_surf = pygame.Surface((100, 100), pygame.SRCALPHA)

    def draw_hud(self, screen, vehicle, active_vehicle_idx, num_vehicles, performance_metrics, goal_manager):
        """
        차량 상태 HUD 표시

        Args:
            screen: pygame 화면 객체
            vehicle: 현재 활성 차량
            active_vehicle_idx: 활성 차량 인덱스
            num_vehicles: 총 차량 수
            performance_metrics: 성능 측정 정보
            goal_manager: 목적지 관리자
        """
        state = vehicle.state
        config = vehicle.vehicle_config

        # 아커만 조향각 계산 (좌/우 앞바퀴)
        wheelbase = config['wheelbase']
        track = config['track']
        steer = state.steer

        if abs(steer) > 0.001:
            R = wheelbase / np.tan(abs(steer))
            if steer < 0:  # 좌회전
                left_steer = -np.arctan(wheelbase / (R - track/2))
                right_steer = -np.arctan(wheelbase / (R + track/2))
            else:  # 우회전
                left_steer = np.arctan(wheelbase / (R + track/2))
                right_steer = np.arctan(wheelbase / (R - track/2))
        else:
            left_steer = right_steer = 0.0

        # 기본 정보
        hud = [
            f"Vehicle {active_vehicle_idx + 1}/{num_vehicles}",
            f"Position: ({state.x:.1f}, {state.y:.1f})",
            f"Heading: {np.degrees(state.yaw):.1f}°",
            f"Velocity: {state.vel_long:.2f} m/s ({state.vel_long * 3.6:.1f} km/h)",
            f"FPS: {performance_metrics['fps']:.1f}",
        ]

        # 목적지 정보 추가
        goal = goal_manager.get_vehicle_goal(vehicle.get_id())
        if goal:
            distance = state.distance_to_target
            yaw_diff = np.degrees(state.yaw_diff_to_target)
            hud.extend([
                f"Target: ({state.target_x:.1f}, {state.target_y:.1f})",
                f"Distance to target: {distance:.2f}m",
                f"Heading diff: {yaw_diff:.1f}°"
            ])

        # 디버그 정보 추가
        if self.config['visualization']['debug_mode']:
            hud.extend([
                f"Throttling: {state.throttle:.2f} m/s²",
                f"Longitude Accel: {state.acc_long:.2f} m/s²",
                f"Steering: {np.degrees(state.steer):.1f}°",
                f"Left wheel: {np.degrees(left_steer):.1f}°, Right wheel: {np.degrees(right_steer):.1f}°",
                f"Lateral Vel: {state.vel_lat:.2f} m/s",
                f"Lateral Accel: {state.acc_lat:.2f} m/s²",
                f"G-Forces: Long {state.g_forces[0]:.2f}g, Lat {state.g_forces[1]:.2f}g"
            ])

        # HUD 배경
        hud_width = 280
        hud_height = len(hud) * 25 + 10
        hud_surf = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        hud_surf.fill(self.config['visualization']['hud_bg_color'])
        screen.blit(hud_surf, (10, 10))

        # HUD 텍스트
        y_pos = 15
        for line in hud:
            text = self.font.render(line, True, self.config['visualization']['hud_fg_color'])
            screen.blit(text, (20, y_pos))
            y_pos += 25

        # G-force 미터
        self._draw_g_force_meter(screen, vehicle, self.config['visualization']['window_width'] - 90, 20)

        # 큰 속도계
        speed_kmh = state.vel_long * 3.6
        speed_text = self.large_font.render(f"{speed_kmh:.0f}", True, (255, 255, 255))
        kmh_text = self.font.render("km/h", True, (200, 200, 200))

        # 중앙 상단에 위치
        speed_rect = speed_text.get_rect(center=(self.config['visualization']['window_width'] // 2, 40))
        kmh_rect = kmh_text.get_rect(center=(self.config['visualization']['window_width'] // 2, 65))

        screen.blit(speed_text, speed_rect)
        screen.blit(kmh_text, kmh_rect)

    def _draw_g_force_meter(self, screen, vehicle, x, y):
        """
        G-force 미터 그리기

        Args:
            screen: pygame 화면 객체
            vehicle: 차량 객체
            x, y: 미터 위치 좌표
        """
        meter_size = 80
        center = (x + meter_size // 2, y + meter_size // 2)
        radius = meter_size // 2 - 5

        # 배경
        bg_surf = pygame.Surface((meter_size, meter_size), pygame.SRCALPHA)
        pygame.draw.circle(bg_surf, (50, 50, 50, 150), (meter_size//2, meter_size//2), radius)

        # 눈금 그리기
        pygame.draw.circle(bg_surf, (100, 100, 100), (meter_size//2, meter_size//2), radius, 1)
        pygame.draw.circle(bg_surf, (100, 100, 100), (meter_size//2, meter_size//2), radius//2, 1)

        # 십자선
        pygame.draw.line(bg_surf, (100, 100, 100), (meter_size//2, 0), (meter_size//2, meter_size), 1)
        pygame.draw.line(bg_surf, (100, 100, 100), (0, meter_size//2), (meter_size, meter_size//2), 1)

        screen.blit(bg_surf, (x, y))

        # G-force 포인터 그리기
        g_lateral = vehicle.state.g_forces[1]
        g_longitudinal = vehicle.state.g_forces[0]

        # G-force를 픽셀로 변환 (1G = 반경의 1/3)
        scale_factor = radius / 3
        pointer_x = center[0] + g_lateral * scale_factor
        pointer_y = center[1] - g_longitudinal * scale_factor  # Y축 반전

        # G-force 범위 제한
        max_distance = radius - 2
        dx = pointer_x - center[0]
        dy = pointer_y - center[1]
        distance = (dx**2 + dy**2)**0.5

        if distance > max_distance:
            scale = max_distance / distance
            pointer_x = center[0] + dx * scale
            pointer_y = center[1] + dy * scale

        # 포인터 그리기
        pygame.draw.circle(screen, (255, 50, 50), (int(pointer_x), int(pointer_y)), 5)
        pygame.draw.line(screen, (200, 200, 200), center, (int(pointer_x), int(pointer_y)), 2)

        # G-force 값 표시
        g_text = self.font.render(f"{(g_longitudinal**2 + g_lateral**2)**0.5:.1f}G", True, (255, 255, 0))
        screen.blit(g_text, (x + 28, y + meter_size + 5))

    def draw_target_direction_arrows(self, screen, vehicles, active_vehicle_idx, world_to_screen, goal_manager):
        """
        각 차량의 목표 방향 화살표 그리기

        Args:
            screen: pygame 화면 객체
            vehicles: 차량 리스트
            active_vehicle_idx: 활성 차량 인덱스
            world_to_screen: 월드 좌표를 화면 좌표로 변환하는 함수
            goal_manager: 목적지 관리자
        """
        if not vehicles:
            return

        # 활성 차량 ID 가져오기
        active_vehicle_id = vehicles[active_vehicle_idx].id if active_vehicle_idx < len(vehicles) else None

        for vehicle in vehicles:
            goal = goal_manager.get_vehicle_goal(vehicle.id)
            if goal:
                # 차량 위치와 목표 위치를 화면 좌표로 변환
                vehicle_pos = world_to_screen(vehicle.state.x, vehicle.state.y)
                target_pos = world_to_screen(vehicle.state.target_x, vehicle.state.target_y)

                # 화살표 그리기 (차량에서 목표로)
                # 화살표 색상은 활성 차량이면 밝은 노란색, 아니면 어두운 노란색
                arrow_color = (255, 255, 0) if vehicle.id == active_vehicle_id else (180, 180, 0)
                pygame.draw.line(screen, arrow_color, vehicle_pos, target_pos, 2)

                # 화살표 머리 그리기
                # 화살표 방향 계산
                dx = target_pos[0] - vehicle_pos[0]
                dy = target_pos[1] - vehicle_pos[1]
                length = np.sqrt(dx*dx + dy*dy)

                if length > 10:  # 일정 거리 이상일 때만 화살표 머리 그리기
                    # 단위 벡터
                    dx, dy = dx/length, dy/length

                    # 화살표 머리 크기
                    head_length = 10

                    # 화살표 머리 끝점 좌표
                    arrow_head1_x = target_pos[0] - head_length * (dx*0.866 + dy*0.5)
                    arrow_head1_y = target_pos[1] - head_length * (-dx*0.5 + dy*0.866)
                    arrow_head2_x = target_pos[0] - head_length * (dx*0.866 - dy*0.5)
                    arrow_head2_y = target_pos[1] - head_length * (dx*0.5 + dy*0.866)

                    # 화살표 머리 그리기
                    pygame.draw.line(screen, arrow_color, target_pos, (int(arrow_head1_x), int(arrow_head1_y)), 2)
                    pygame.draw.line(screen, arrow_color, target_pos, (int(arrow_head2_x), int(arrow_head2_y)), 2)
