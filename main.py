# -*- coding: utf-8 -*-
import pygame
import numpy as np
from src.env import CarSimulatorEnv

# ==============
# Manual Control
# ==============
def manual_control():
    """키보드로 차량 직접 제어하는 함수"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print("=== Vehicle Simulator ===")
    print("Controls:")
    print("  Arrow Keys / WASD: Drive the vehicle")
    print("  Space: Handbrake")
    print("  C: Toggle camera follow")
    print("  H: Toggle debug info")
    print("  R: Reset camera view")
    print("  +/-: Zoom in/out")
    print("  I/J/K/L: Pan camera")
    print("  F5: Save state")
    print("  F9: Load state")
    print("  ESC: Quit")

    running = True
    while running:
        # 기본 액션: 정지, 직진
        action = np.zeros(2)

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 환경 스텝
        _, _, done, _ = env.step(action)

        # 렌더링
        env.render()

        # 종료 조건
        if done:
            break

    # 환경 종료
    env.close()

# ==============
# Multi-Vehicle Test
# ==============
def multi_vehicle_test():
    """다중 차량 테스트 모드"""
    # 두 대의 차량으로 환경 초기화
    env = CarSimulatorEnv(multi_vehicle=True, num_vehicles=2)
    env.reset()

    # 안내 메시지 출력
    print("=== Multi-Vehicle Test ===")
    print("Controls:")
    print("  Arrow Keys / WASD: Drive the active vehicle")
    print("  Tab: Switch between vehicles")
    print("  Space: Handbrake")
    print("  C: Toggle camera follow")
    print("  H: Toggle debug info")
    print("  R: Reset camera view")
    print("  +/-: Zoom in/out")
    print("  I/J/K/L: Pan camera")
    print("  F5: Save state")
    print("  F9: Load state")
    print("  ESC: Quit")

    # 초기 목적지 생성 (각 차량마다)
    for i in range(env.num_vehicles):
        create_random_goal(env, i)

    running = True
    while running:
        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 환경 스텝
        _, _, done, info = env.step(action)

        # 목적지 도달 확인 및 새 목적지 생성
        for i in range(env.num_vehicles):
            if info['reached_targets'].get(i, False):
                # 목적지 도달 시 새 목적지 생성
                create_random_goal(env, i)

        # 렌더링
        env.render()

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 충돌로 인한 종료 시 잠시 대기 후 재시작
        if done:
            if info.get('collision', False):
                pygame.time.wait(1000)  # 1초 대기
                env.reset()
                # 초기 목적지 다시 생성
                for i in range(env.num_vehicles):
                    create_random_goal(env, i)

    # 환경 종료
    env.close()

def create_random_goal(env, vehicle_id, min_distance=15.0, max_distance=30.0):
    """
    지정된 차량에 대한 랜덤 목적지 생성

    Args:
        env: CarSimulatorEnv 객체
        vehicle_id: 차량 ID
        min_distance: 현재 위치로부터 최소 거리 (m)
        max_distance: 현재 위치로부터 최대 거리 (m)
    """
    vehicle = env.vehicles[vehicle_id]
    goal_manager = env.get_goal_manager()

    # 현재 차량 위치
    current_x = vehicle.state.x
    current_y = vehicle.state.y

    # 현재 목적지 정보 (있는 경우)
    current_goal_id = goal_manager.get_vehicle_goal_id(vehicle_id)
    if current_goal_id is not None:
        goal_manager.remove_goal(current_goal_id)

    # 적절한 거리에 새 목적지 생성
    while True:
        # 랜덤 방향 (0-2π)
        angle = np.random.random() * 2 * np.pi
        # 랜덤 거리 (min_distance-max_distance)
        distance = min_distance + np.random.random() * (max_distance - min_distance)

        # 새 위치 계산
        new_x = current_x + distance * np.cos(angle)
        new_y = current_y + distance * np.sin(angle)

        # 목표 방향 (차량이 도착 시 가질 방향) - 무작위 설정
        target_yaw = np.random.random() * 2 * np.pi

        # 다른 차량의 목적지와 일정 거리 이상 떨어지게 (충돌 방지)
        valid_position = True
        for other_id in range(env.num_vehicles):
            if other_id == vehicle_id:
                continue

            other_goal = goal_manager.get_vehicle_goal(other_id)
            if other_goal:
                dist_to_other_goal = np.sqrt((new_x - other_goal.x)**2 + (new_y - other_goal.y)**2)
                if dist_to_other_goal < 10.0:  # 최소 10m 떨어지게
                    valid_position = False
                    break

        if valid_position:
            # 적절한 위치를 찾았으면 목적지 생성
            goal_color = (0, 255, 0) if vehicle_id == 0 else (255, 255, 0)  # 첫 번째 차량은 녹색, 두 번째는 노란색
            env.add_goal_for_vehicle(vehicle_id, new_x, new_y, target_yaw, 2.0, goal_color)
            break

# ==============
# Waypoint Navigation Test
# ==============
def waypoint_navigation_test():
    """웨이포인트 주행 테스트 모드"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print("=== Waypoint Navigation Test ===")
    print("Controls:")
    print("  Arrow Keys / WASD: Drive the vehicle")
    print("  Space: Handbrake")
    print("  N: Generate new waypoint sequence")
    print("  C: Toggle camera follow")
    print("  H: Toggle debug info")
    print("  R: Reset camera view")
    print("  +/-: Zoom in/out")
    print("  I/J/K/L: Pan camera")
    print("  F5: Save state")
    print("  F9: Load state")
    print("  ESC: Quit")

    # 웨이포인트 시퀀스 상태 변수들
    waypoints = []
    current_waypoint_index = 0

    # 초기 웨이포인트 시퀀스 생성
    waypoints = create_waypoint_sequence(env)
    set_next_waypoint(env, waypoints, current_waypoint_index)

    running = True
    generate_new_waypoints = False

    while running:
        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    # 새 웨이포인트 시퀀스 생성
                    generate_new_waypoints = True

        # 환경 스텝
        _, _, done, info = env.step(action)

        # 목적지 도달 확인
        if info['reached_targets'].get(env.vehicle.id, False):
            # 다음 웨이포인트로 이동
            current_waypoint_index += 1

            # 모든 웨이포인트 완료 시 새 시퀀스 생성
            if current_waypoint_index >= len(waypoints):
                generate_new_waypoints = True
            else:
                set_next_waypoint(env, waypoints, current_waypoint_index)

        # 새 웨이포인트 시퀀스 생성이 필요한 경우
        if generate_new_waypoints:
            # 기존 목적지 모두 제거
            env.get_goal_manager().clear_goals()

            # 새 웨이포인트 시퀀스 생성
            waypoints = create_waypoint_sequence(env)
            current_waypoint_index = 0
            set_next_waypoint(env, waypoints, current_waypoint_index)

            generate_new_waypoints = False

        # 렌더링
        env.render()

        # 충돌로 인한 종료 시 잠시 대기 후 재시작
        if done:
            if info.get('collision', False):
                pygame.time.wait(1000)  # 1초 대기
                env.reset()

                # 새 웨이포인트 시퀀스 생성
                waypoints = create_waypoint_sequence(env)
                current_waypoint_index = 0
                set_next_waypoint(env, waypoints, current_waypoint_index)

    # 환경 종료
    env.close()

def create_waypoint_sequence(env, num_waypoints=5, min_distance=10.0, max_distance=25.0):
    """
    웨이포인트 시퀀스 생성

    Args:
        env: CarSimulatorEnv 객체
        num_waypoints: 생성할 웨이포인트 수
        min_distance: 웨이포인트 간 최소 거리 (m)
        max_distance: 웨이포인트 간 최대 거리 (m)

    Returns:
        웨이포인트 리스트: [(x, y, yaw), ...]
    """
    vehicle = env.vehicle
    waypoints = []

    # 시작점 (현재 차량 위치)
    current_x = vehicle.state.x
    current_y = vehicle.state.y

    for i in range(num_waypoints):
        # 랜덤 방향 (이전 방향에서 ±90도 이내로 제한하여 자연스러운 경로 생성)
        if i == 0:
            # 첫 웨이포인트는 현재 차량 방향 기준 ±45도 내에서 생성
            base_angle = vehicle.state.yaw
            angle = base_angle + (np.random.random() - 0.5) * np.pi/2
        else:
            # 이전 웨이포인트 방향 기준 ±90도 내에서 생성
            dx = current_x - waypoints[-1][0]
            dy = current_y - waypoints[-1][1]
            base_angle = np.arctan2(dy, dx)
            angle = base_angle + (np.random.random() - 0.5) * np.pi

        # 랜덤 거리
        distance = min_distance + np.random.random() * (max_distance - min_distance)

        # 새 위치 계산
        new_x = current_x + distance * np.cos(angle)
        new_y = current_y + distance * np.sin(angle)

        # 도착 방향 (다음 웨이포인트를 향하도록, 마지막은 랜덤)
        if i < num_waypoints - 1:
            target_yaw = angle
        else:
            target_yaw = np.random.random() * 2 * np.pi

        # 웨이포인트 추가
        waypoints.append((new_x, new_y, target_yaw))

        # 현재 위치 업데이트
        current_x, current_y = new_x, new_y

    # 모든 웨이포인트를 미리 생성하여 표시 (현재 활성 목표만 활성 색상)
    goal_manager = env.get_goal_manager()

    for i, (x, y, yaw) in enumerate(waypoints):
        # 웨이포인트 색상: 비활성 웨이포인트는 회색으로 표시
        if i > 0:  # 첫 번째 웨이포인트는 set_next_waypoint에서 추가
            color = (150, 150, 150)  # 회색
            radius = 1.5  # 기본 반경
            goal_id = goal_manager.add_goal(x, y, yaw, radius, color)

    return waypoints

def set_next_waypoint(env, waypoints, index):
    """
    다음 웨이포인트를 활성 목표로 설정

    Args:
        env: CarSimulatorEnv 객체
        waypoints: 웨이포인트 리스트
        index: 활성화할 웨이포인트 인덱스
    """
    if index >= len(waypoints):
        return

    goal_manager = env.get_goal_manager()
    vehicle_id = env.vehicle.id

    # 기존 목표가 있으면 제거
    current_goal_id = goal_manager.get_vehicle_goal_id(vehicle_id)
    if current_goal_id is not None:
        goal_manager.remove_goal(current_goal_id)

    # 새 목표 생성 및 활성화
    x, y, yaw = waypoints[index]
    new_goal_id = env.add_goal_for_vehicle(vehicle_id, x, y, yaw, 2.0, (0, 255, 0))

# ==============
# Course Builder Helper Functions
# ==============
def create_test_course(obstacle_manager, course_type="basic"):
    """
    테스트 코스 생성 함수

    Args:
        obstacle_manager: 장애물 관리자 객체
        course_type: 코스 유형 ("basic", "slalom")
    """
    # 모든 장애물 제거
    obstacle_manager.clear_obstacles()

    if course_type == "basic":
        # 기본 테스트 코스 - 단순한 장애물 배치
        obstacle_manager.add_circle_obstacle(10, 10, 2.0)
        obstacle_manager.add_circle_obstacle(-10, -10, 3.0)
        obstacle_manager.add_square_obstacle(15, -15, 3.0)
        obstacle_manager.add_rectangle_obstacle(-15, 15, 5.0, 2.0, yaw=np.radians(30))

        # 원형 트랙 형태로 장애물 배치
        radius = 25.0
        num_barriers = 8
        for i in range(num_barriers):
            angle = 2 * np.pi * i / num_barriers
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # 트랙 내부 경계
            inner_radius = 1.5
            if i % 2 == 0:
                obstacle_manager.add_circle_obstacle(x*0.6, y*0.6, inner_radius, color=(150, 150, 150))
            else:
                obstacle_manager.add_square_obstacle(x*0.6, y*0.6, inner_radius*2, color=(150, 150, 150))

            # 트랙 외부 경계
            outer_radius = 2.0
            if i % 2 == 0:
                obstacle_manager.add_rectangle_obstacle(x*1.2, y*1.2, outer_radius*3, outer_radius, yaw=angle, color=(100, 100, 100))
            else:
                obstacle_manager.add_circle_obstacle(x*1.2, y*1.2, outer_radius, color=(100, 100, 100))

    elif course_type == "slalom":
        # 슬라럼 코스 - 지그재그 장애물 배치
        num_obstacles = 10
        spacing = 10.0

        for i in range(num_obstacles):
            x = i * spacing + 20
            y = 8.0 if i % 2 == 0 else -8.0
            obstacle_manager.add_circle_obstacle(x, y, 2.0, color=(200, 100, 100))

        # 코스 경계 추가
        for i in range(num_obstacles + 1):
            x = i * spacing - spacing/2 + 20
            obstacle_manager.add_rectangle_obstacle(x, 35, spacing, 1.0, color=(100, 100, 100))
            obstacle_manager.add_rectangle_obstacle(x, -35, spacing, 1.0, color=(100, 100, 100))

# ==============
# Manual Control With Obstacles
# ==============
def manual_control_with_obstacles():
    """장애물이있는 환경에서 차량 수동 제어"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 기본 코스 생성
    current_course = "basic"
    create_test_course(env.get_obstacle_manager(), current_course)

    # 안내 메시지 출력
    print("=== Vehicle Simulator with Obstacles ===")
    print("Controls:")
    print("  Arrow Keys / WASD: Drive the vehicle")
    print("  Space: Handbrake")
    print("  C: Toggle camera follow")
    print("  H: Toggle debug info")
    print("  R: Reset camera view")
    print("  +/-: Zoom in/out")
    print("  I/J/K/L: Pan camera")
    print("  1: Load Basic Course")
    print("  2: Load Slalom Course")
    print("  F5: Save state")
    print("  F9: Load state")
    print("  ESC: Quit")

    running = True
    while running:
        # 기본 액션: 정지, 직진
        action = np.zeros(2)

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    # 기본 코스 로드
                    current_course = "basic"
                    env.reset()
                    create_test_course(env.get_obstacle_manager(), current_course)
                elif event.key == pygame.K_2:
                    # 슬라럼 코스 로드
                    current_course = "slalom"
                    env.reset()
                    create_test_course(env.get_obstacle_manager(), current_course)

        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 환경 스텝
        _, reward, done, info = env.step(action)

        # 렌더링
        env.render()

        # 충돌로 인한 종료 시 잠시 대기 후 재시작
        if done:
            if info.get('collision', False):
                pygame.time.wait(1000)  # 1초 대기
                env.reset()
                create_test_course(env.obstacle_manager, current_course)

    # 환경 종료
    env.close()

# ==============
# Main Menu
# ==============
def show_main_menu():
    """메인 메뉴 표시"""
    # Pygame 초기화
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Vehicle Simulator")

    # 폰트 설정
    title_font = pygame.font.Font(None, 64)
    menu_font = pygame.font.Font(None, 36)

    # 메뉴 항목
    menu_items = [
        "1. Manual Driving",
        "2. Obstacle Course Driving",
        "3. Multi-Vehicle Test",
        "4. Waypoint Navigation Test",
        "5. Quit"
    ]

    running = True
    while running:
        # 화면 지우기
        screen.fill((0, 0, 0))

        # 타이틀 표시
        title_text = title_font.render("Vehicle Simulator", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(400, 100))
        screen.blit(title_text, title_rect)

        # 메뉴 항목 표시
        for i, item in enumerate(menu_items):
            item_text = menu_font.render(item, True, (200, 200, 200))
            item_rect = item_text.get_rect(center=(400, 250 + i * 50))
            screen.blit(item_text, item_rect)

        # 화면 업데이트
        pygame.display.flip()

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    # 수동 운전 모드
                    pygame.quit()
                    manual_control()
                    return
                elif event.key == pygame.K_2:
                    # 장애물 코스 운전 모드
                    pygame.quit()
                    manual_control_with_obstacles()
                    return
                elif event.key == pygame.K_3:
                    # 다중 차량 테스트 모드
                    pygame.quit()
                    multi_vehicle_test()
                    return
                elif event.key == pygame.K_4:
                    # 웨이포인트 주행 테스트 모드
                    pygame.quit()
                    waypoint_navigation_test()
                    return
                elif event.key == pygame.K_5 or event.key == pygame.K_ESCAPE:
                    # 종료
                    running = False

    pygame.quit()

# ==============
# Main Function
# ==============
if __name__ == "__main__":
    # 메인 메뉴 표시
    show_main_menu()
