import pygame
import numpy as np
from src.env.env import CarSimulatorEnv

def print_basic_controls(menu_name):
    print(f"=== Vehicle Simulator_{menu_name} ===")
    print("Controls:")
    print("  Arrow Keys / WASD: Drive the vehicle")
    print("  Space: Handbrake")
    print("  C: Toggle camera follow")
    print("  R: Reset camera view")
    print("  +/-: Zoom in/out")
    print("  I/J/K/L: Pan camera")
    print("  F1: Toggle Training Mode")
    print("  F2: Toggle Visualization")
    print("  F3: Toggle HUD")
    print("  F4: Toggle debug mode")
    print("  F5: Save state")
    print("  F9: Load state")
    print("  Tab: Switch between vehicles")
    print("  ESC: Quit")

# ==============
# Manual Control
# ==============
def manual_control():
    """키보드로 차량 직접 제어하는 함수"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print_basic_controls("Manual Control")

    running = True
    while running:
        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 환경 스텝
        _, _, done, _ = env.step(action)

        # 렌더링
        env.render()

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

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
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print_basic_controls("Multi-Vehicle Test")
    print("  Tab: Switch between vehicles")

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
            # 차량별 다른 색상 사용
            colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
            goal_color = colors[vehicle_id % len(colors)]
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
    print_basic_controls("Waypoint Navigation Test")
    print("  N: Generate new waypoints")
    print("  Tab: Switch between vehicles")

    # 웨이포인트 시퀀스 상태 변수들
    vehicle_waypoints = {}
    vehicle_current_index = {}

    # 모든 차량에 대한 초기 웨이포인트 시퀀스 생성
    for vehicle_id in range(env.num_vehicles):
        vehicle_waypoints[vehicle_id] = create_waypoint_sequence(env, vehicle_id)
        vehicle_current_index[vehicle_id] = 0
        set_next_waypoint(env, vehicle_waypoints[vehicle_id], vehicle_current_index[vehicle_id], vehicle_id)

    running = True
    generate_new_waypoints = False
    target_vehicle_id = None

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
                    target_vehicle_id = env.active_vehicle_idx

        # 환경 스텝
        _, _, done, info = env.step(action)

        # 모든 차량에 대해 목적지 도달 확인
        for vehicle_id in range(env.num_vehicles):
            if info['reached_targets'].get(vehicle_id, False):
                # 해당 차량의 웨이포인트 인덱스 증가
                if vehicle_id in vehicle_current_index:
                    vehicle_current_index[vehicle_id] += 1

                    # 모든 웨이포인트 완료 시 새 시퀀스 생성
                    if vehicle_current_index[vehicle_id] >= len(vehicle_waypoints[vehicle_id]):
                        # 해당 차량에 대한 새 웨이포인트 시퀀스 생성
                        vehicle_waypoints[vehicle_id] = create_waypoint_sequence(env, vehicle_id)
                        vehicle_current_index[vehicle_id] = 0
                        set_next_waypoint(env, vehicle_waypoints[vehicle_id], vehicle_current_index[vehicle_id], vehicle_id)
                    else:
                        # 다음 웨이포인트로 이동
                        set_next_waypoint(env, vehicle_waypoints[vehicle_id], vehicle_current_index[vehicle_id], vehicle_id)

        # 특정 차량에 대한 새 웨이포인트 시퀀스 생성이 필요한 경우
        if generate_new_waypoints and target_vehicle_id is not None:
            # 해당 차량의 현재 목적지 제거
            goal_manager = env.get_goal_manager()
            current_goal_id = goal_manager.get_vehicle_goal_id(target_vehicle_id)
            if current_goal_id is not None:
                goal_manager.remove_goal(current_goal_id)

            # 해당 차량에 대한 새 웨이포인트 시퀀스 생성
            vehicle_waypoints[target_vehicle_id] = create_waypoint_sequence(env, target_vehicle_id)
            vehicle_current_index[target_vehicle_id] = 0
            set_next_waypoint(env, vehicle_waypoints[target_vehicle_id], vehicle_current_index[target_vehicle_id], target_vehicle_id)

            generate_new_waypoints = False
            target_vehicle_id = None

        # 렌더링
        env.render()

        # 충돌로 인한 종료 시 잠시 대기 후 재시작
        if done:
            if info.get('collision', False):
                pygame.time.wait(1000)  # 1초 대기
                env.reset()

                # 모든 차량에 대한 새 웨이포인트 시퀀스 생성
                for vehicle_id in range(env.num_vehicles):
                    vehicle_waypoints[vehicle_id] = create_waypoint_sequence(env, vehicle_id)
                    vehicle_current_index[vehicle_id] = 0
                    set_next_waypoint(env, vehicle_waypoints[vehicle_id], vehicle_current_index[vehicle_id], vehicle_id)

    # 환경 종료
    env.close()

def create_waypoint_sequence(env, vehicle_id=0, num_waypoints=5, min_distance=10.0, max_distance=25.0):
    """
    웨이포인트 시퀀스 생성

    Args:
        env: CarSimulatorEnv 객체
        vehicle_id: 차량 ID
        num_waypoints: 생성할 웨이포인트 수
        min_distance: 웨이포인트 간 최소 거리 (m)
        max_distance: 웨이포인트 간 최대 거리 (m)

    Returns:
        웨이포인트 리스트: [(x, y, yaw), ...]
    """
    vehicle = env.vehicles[vehicle_id]
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

    # 차량별 색상 설정
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
    base_color = colors[vehicle_id % len(colors)]

    # 비활성 웨이포인트는 더 투명하게 표시
    inactive_color = (base_color[0]//3, base_color[1]//3, base_color[2]//3)

    for i, (x, y, yaw) in enumerate(waypoints):
        # 웨이포인트 색상: 비활성 웨이포인트는 옅은 색으로 표시
        if i > 0:  # 첫 번째 웨이포인트는 set_next_waypoint에서 추가
            radius = 1.5  # 기본 반경
            goal_id = goal_manager.add_goal(x, y, yaw, radius, inactive_color)

    return waypoints

def set_next_waypoint(env, waypoints, index, vehicle_id=0):
    """
    다음 웨이포인트를 활성 목표로 설정

    Args:
        env: CarSimulatorEnv 객체
        waypoints: 웨이포인트 리스트
        index: 활성화할 웨이포인트 인덱스
        vehicle_id: 차량 ID
    """
    if index >= len(waypoints):
        return

    goal_manager = env.get_goal_manager()

    # 기존 목표가 있으면 제거
    current_goal_id = goal_manager.get_vehicle_goal_id(vehicle_id)
    if current_goal_id is not None:
        goal_manager.remove_goal(current_goal_id)

    # 차량별 색상 설정
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
    goal_color = colors[vehicle_id % len(colors)]

    # 새 목표 생성 및 활성화
    x, y, yaw = waypoints[index]
    new_goal_id = env.add_goal_for_vehicle(vehicle_id, x, y, yaw, 2.0, goal_color)

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
        obstacle_manager.add_circle_obstacle(None, 10, 10, 0.0, 0.0, 0.0, 2.0)
        obstacle_manager.add_circle_obstacle(None, -10, -10, 0.0, 0.0, 0.0, 3.0)
        obstacle_manager.add_square_obstacle(None, 15, -15, 0.0, 0.0, 0.0, 3.0)
        obstacle_manager.add_rectangle_obstacle(None, -15, 15, np.radians(30), 0.0, 0.0, 5.0, 2.0)

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
                obstacle_manager.add_circle_obstacle(None, x*0.6, y*0.6, 0.0, 0.0, 0.0, inner_radius, (150, 150, 150))
            else:
                obstacle_manager.add_square_obstacle(None, x*0.6, y*0.6, 0.0, 0.0, 0.0, inner_radius*2, (150, 150, 150))

            # 트랙 외부 경계
            outer_radius = 2.0
            if i % 2 == 0:
                obstacle_manager.add_rectangle_obstacle(None, x*1.2, y*1.2, angle, 0.0, 0.0, outer_radius*3, outer_radius, (100, 100, 100))
            else:
                obstacle_manager.add_circle_obstacle(None, x*1.2, y*1.2, 0.0, 0.0, 0.0, outer_radius, (100, 100, 100))

    elif course_type == "slalom":
        # 슬라럼 코스 - 지그재그 장애물 배치
        num_obstacles = 10
        spacing = 10.0

        for i in range(num_obstacles):
            x = i * spacing + 20
            y = 8.0 if i % 2 == 0 else -8.0
            obstacle_manager.add_circle_obstacle(None, x, y, 0.0, 0.0, 0.0, 2.0, (200, 100, 100))

        # 코스 경계 추가
        for i in range(num_obstacles + 1):
            x = i * spacing - spacing/2 + 20
            obstacle_manager.add_rectangle_obstacle(None, x, 35, 0.0, 0.0, 0.0, spacing, 1.0, (100, 100, 100))
            obstacle_manager.add_rectangle_obstacle(None, x, -35, 0.0, 0.0, 0.0, spacing, 1.0, (100, 100, 100))

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
    print_basic_controls("Manual Control With Obstacles")
    print("  1: Load Basic Course")
    print("  2: Load Slalom Course")
    print("  Tab: Switch between vehicles")

    running = True
    while running:
        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 환경 스텝
        _, reward, done, info = env.step(action)

        # 렌더링
        env.render()

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

        # 충돌로 인한 종료 시 잠시 대기 후 재시작
        if done:
            if info.get('collision', False):
                pygame.time.wait(1000)  # 1초 대기
                env.reset()
                create_test_course(env.get_obstacle_manager(), current_course)

    # 환경 종료
    env.close()

# ==============
# Dynamic Obstacles Test
# ==============
def dynamic_obstacles_test():
    """동적 장애물이 있는 환경에서 차량 주행 테스트"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 동적 장애물 코스 생성
    create_dynamic_obstacle_course(env.get_obstacle_manager())

    # 안내 메시지 출력
    print_basic_controls("Dynamic Obstacles Test")
    print("  1: Reset with circular moving obstacles")
    print("  2: Reset with oscillating obstacles")
    print("  3: Reset with random movement obstacles")
    print("  Tab: Switch between vehicles")

    # 모든 차량에 대한 초기 목적지 생성
    for i in range(env.num_vehicles):
        create_random_goal(env, i)

    current_course = "circular"
    running = True
    while running:
        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 환경 스텝
        _, reward, done, info = env.step(action)

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
                elif event.key == pygame.K_1:
                    # 원형 이동 장애물 코스
                    current_course = "circular"
                    env.reset()
                    create_dynamic_obstacle_course(env.get_obstacle_manager(), current_course)
                    # 목적지 재생성
                    for i in range(env.num_vehicles):
                        create_random_goal(env, i)
                elif event.key == pygame.K_2:
                    # 진동하는 장애물 코스
                    current_course = "oscillating"
                    env.reset()
                    create_dynamic_obstacle_course(env.get_obstacle_manager(), current_course)
                    # 목적지 재생성
                    for i in range(env.num_vehicles):
                        create_random_goal(env, i)
                elif event.key == pygame.K_3:
                    # 무작위 움직임 장애물 코스
                    current_course = "random"
                    env.reset()
                    create_dynamic_obstacle_course(env.get_obstacle_manager(), current_course)
                    # 목적지 재생성
                    for i in range(env.num_vehicles):
                        create_random_goal(env, i)

        # 충돌로 인한 종료 시 잠시 대기 후 재시작
        if done:
            if info.get('collision', False):
                pygame.time.wait(1000)  # 1초 대기
                env.reset()
                create_dynamic_obstacle_course(env.get_obstacle_manager(), current_course)
                # 목적지 재생성
                for i in range(env.num_vehicles):
                    create_random_goal(env, i)

    # 환경 종료
    env.close()

def create_dynamic_obstacle_course(obstacle_manager, course_type="circular"):
    """
    동적 장애물이 있는 코스 생성 함수

    Args:
        obstacle_manager: 장애물 관리자 객체
        course_type: 장애물 움직임 유형 ("circular", "oscillating", "random")
    """
    # 모든 장애물 제거
    obstacle_manager.clear_obstacles()

    if course_type == "circular":
        # 원형 경로를 따라 이동하는 장애물들
        radius = 20.0
        num_obstacles = 6
        for i in range(num_obstacles):
            angle = 2 * np.pi * i / num_obstacles
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # 각각 다른 속도와 크기로 설정
            speed = 2.0 + i * 0.5  # 속도 차이
            size = 1.0 + (i % 3) * 0.5  # 크기 차이

            # 장애물 유형을 번갈아가며 생성
            if i % 3 == 0:
                # 원형 장애물
                obstacle_manager.add_circle_obstacle(
                    None, x, y, angle + np.pi/2, 0.5, speed, size, (255, 100, 100)
                )
            elif i % 3 == 1:
                # 정사각형 장애물
                obstacle_manager.add_square_obstacle(
                    None, x, y, angle + np.pi/2, 0.3, speed, size * 1.5, (100, 255, 100)
                )
            else:
                # 직사각형 장애물
                obstacle_manager.add_rectangle_obstacle(
                    None, x, y, angle + np.pi/2, 0.2, speed, size * 3, size, (100, 100, 255)
                )

        # 몇 개의 벽 추가
        wall_dist = 35
        wall_len = 15
        obstacle_manager.add_rectangle_obstacle(None, -wall_dist, 0, 0, 0, 0, 1.0, wall_len, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, wall_dist, 0, 0, 0, 0, 1.0, wall_len, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, 0, -wall_dist, np.pi/2, 0, 0, 1.0, wall_len, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, 0, wall_dist, np.pi/2, 0, 0, 1.0, wall_len, (100, 100, 100))

    elif course_type == "oscillating":
        # 양쪽으로 진동하는 장애물
        spacing = 8.0
        num_rows = 5
        for i in range(num_rows):
            y = -20 + i * 10
            # 좌우 스윙 장애물 (x축)
            for j in range(3):
                x = -15 + j * 15
                if(x == 0 or y == 0):
                    continue
                # 속도값이 음수일 때 역방향으로 움직임, yaw_rate는 0
                # 위치 + (속도 * 시간)이므로, 속도를 바꿔주면 방향 전환됨
                # 각각 다른 속도로 이동하는 장애물 설정
                speed = 3.0 - j * 0.5
                size = 1.5
                # 짝수 행은 왼쪽에서 오른쪽, 홀수 행은 오른쪽에서 왼쪽으로 시작
                direction = 0.0 if i % 2 == 0 else np.pi
                obstacle_manager.add_circle_obstacle(
                    None, x, y, direction, 0.0, speed, size, (255, 150, 50)
                )

        # 위아래로 진동하는 장애물 추가 (y축)
        for i in range(3):
            x = -25 + i * 25
            for j in range(2):
                y = -5 + j * 10
                if(x == 0 or y == 0):
                    continue
                speed = 2.0 + i * 0.3
                size = 1.2
                # 위아래 이동 설정 (yaw가 π/2이면 위쪽, -π/2이면 아래쪽)
                direction = np.pi/2 if j % 2 == 0 else -np.pi/2
                obstacle_manager.add_square_obstacle(
                    None, x, y, direction, 0.0, speed, size * 1.5, (50, 255, 150)
                )

        # 경계 장애물
        border_size = 40
        thickness = 1.0
        obstacle_manager.add_rectangle_obstacle(None, -border_size, 0, 0, 0, 0, thickness, border_size*2, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, border_size, 0, 0, 0, 0, thickness, border_size*2, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, 0, -border_size, np.pi/2, 0, 0, thickness, border_size*2, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, 0, border_size, np.pi/2, 0, 0, thickness, border_size*2, (100, 100, 100))

    elif course_type == "random":
        # 무작위 이동 및 회전하는 장애물
        num_obstacles = 15
        area_size = 30

        for i in range(num_obstacles):
            # 무작위 위치
            x = np.random.uniform(-area_size, area_size)
            y = np.random.uniform(-area_size, area_size)

            # 무작위 방향, 회전 속도, 이동 속도
            yaw = np.random.uniform(0, 2*np.pi)
            yaw_rate = np.random.uniform(-0.5, 0.5)
            speed = np.random.uniform(1.0, 3.0)

            # 장애물 유형 무작위 선택
            obstacle_type = np.random.randint(0, 3)

            if obstacle_type == 0:
                # 원형 장애물
                size = np.random.uniform(1.0, 2.0)
                color = (np.random.randint(150, 255), np.random.randint(50, 150), np.random.randint(50, 150))
                obstacle_manager.add_circle_obstacle(None, x, y, yaw, yaw_rate, speed, size, color)
            elif obstacle_type == 1:
                # 정사각형 장애물
                size = np.random.uniform(1.5, 2.5)
                color = (np.random.randint(50, 150), np.random.randint(150, 255), np.random.randint(50, 150))
                obstacle_manager.add_square_obstacle(None, x, y, yaw, yaw_rate, speed, size, color)
            else:
                # 직사각형 장애물
                width = np.random.uniform(2.0, 4.0)
                height = np.random.uniform(1.0, 2.0)
                color = (np.random.randint(50, 150), np.random.randint(50, 150), np.random.randint(150, 255))
                obstacle_manager.add_rectangle_obstacle(None, x, y, yaw, yaw_rate, speed, width, height, color)

        # 영역 경계 추가
        border_size = 45
        obstacle_manager.add_rectangle_obstacle(None, -border_size, 0, 0, 0, 0, 1.0, border_size*2, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, border_size, 0, 0, 0, 0, 1.0, border_size*2, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, 0, -border_size, np.pi/2, 0, 0, 1.0, border_size*2, (100, 100, 100))
        obstacle_manager.add_rectangle_obstacle(None, 0, border_size, np.pi/2, 0, 0, 1.0, border_size*2, (100, 100, 100))