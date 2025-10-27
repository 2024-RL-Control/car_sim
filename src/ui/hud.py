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

    def draw_hud(self, screen, vehicle, active_vehicle_idx, num_vehicles, performance_metrics):
        """
        차량 상태 HUD 표시

        Args:
            screen: pygame 화면 객체
            vehicle: 현재 활성 차량
            active_vehicle_idx: 활성 차량 인덱스
            num_vehicles: 총 차량 수
            performance_metrics: 성능 측정 정보
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
        goal = vehicle.get_current_goal()
        if goal:
            distance = state.curr_distance_to_target
            frenet_d = state.frenet_d if state.frenet_d is not None else -float('inf')
            target_vel_long = state.target_vel_long if state.target_vel_long is not None else -float('inf')
            hud.extend([
                f"Target: ({state.target_x:.1f}, {state.target_y:.1f})",
                f"Progress: {state.get_progress():.2f}%",
                f"Distance to target: {distance:.2f}m",
                f"Target Velocity: {target_vel_long:.2f}",
                f"Smoothed Target Velocity: {state.smoothed_target_vel_long:.2f}",
                f"Goal Heading Error: {np.degrees(state.error_to_target):.1f}°",
                f"Goal Heading Angle: {np.degrees(state.angle_to_target):.1f}°",
                f"Frenet D: {frenet_d:.2f}",
                f"Frenet Heading Error: {np.degrees(state.error_to_ref):.2f}°",
                f"Frenet Heading Angle: {np.degrees(state.angle_to_ref):.2f}°",
            ])

        # 디버그 정보 추가
        if self.config['visualization']['debug_mode']:
            hud.extend([
                f"Throttling Engine: {state.throttle_engine:.2f}",
                f"Throttling Brake: {state.throttle_brake:.2f}",
                f"Steering: {np.degrees(state.steer):.1f}°",
                f"Left wheel: {np.degrees(left_steer):.1f}°, Right wheel: {np.degrees(right_steer):.1f}°",
                f"Longitude Accel: {state.acc_long:.2f} m/s²",
                f"Lateral Vel: {state.vel_lat:.2f} m/s",
                f"Lateral Accel: {state.acc_lat:.2f} m/s²",
                f"Render Time: {performance_metrics['render_time']:.5f} ms",
                f"Physics Time: {performance_metrics['physics_time']:.5f} ms"
            ])

        # HUD 배경
        hud_width = 300
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

        # 큰 속도계
        speed_kmh = state.vel_long * 3.6
        speed_text = self.large_font.render(f"{speed_kmh:.0f}", True, (255, 255, 255))
        kmh_text = self.font.render("km/h", True, (200, 200, 200))

        # 중앙 상단에 위치
        speed_rect = speed_text.get_rect(center=(self.config['visualization']['window_width'] // 2, 40))
        kmh_rect = kmh_text.get_rect(center=(self.config['visualization']['window_width'] // 2, 65))

        screen.blit(speed_text, speed_rect)
        screen.blit(kmh_text, kmh_rect)

    def draw_target_direction_arrows(self, screen, vehicles, active_vehicle_idx, world_to_screen_func):
        """
        각 차량의 목표 방향 화살표 그리기

        Args:
            screen: pygame 화면 객체
            vehicles: 차량 리스트
            active_vehicle_idx: 활성 차량 인덱스
            world_to_screen_func: 월드 좌표를 화면 좌표로 변환하는 함수
        """
        if not vehicles:
            return

        # 활성 차량 ID 가져오기
        active_vehicle_id = vehicles[active_vehicle_idx].id if active_vehicle_idx < len(vehicles) else None

        for vehicle in vehicles:
            goal = vehicle.get_current_goal()
            if goal:
                # 차량 위치와 목표 위치를 화면 좌표로 변환
                vehicle_pos = world_to_screen_func((vehicle.state.x, vehicle.state.y))
                target_pos = world_to_screen_func((vehicle.state.target_x, vehicle.state.target_y))

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
