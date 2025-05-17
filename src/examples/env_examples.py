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

def create_random_goal(env, vehicle_id, min_distance=15.0, max_distance=30.0):
    """
    지정된 차량에 대한 랜덤 목적지 생성

    Args:
        env: CarSimulatorEnv 객체
        vehicle_id: 차량 ID
        min_distance: 현재 위치로부터 최소 거리 (m)
        max_distance: 현재 위치로부터 최대 거리 (m)
    """
    vehicle = env.vehicle_manager.get_vehicle_by_id(vehicle_id)
    vehicle.clear_goals()

    # 현재 차량 위치
    current_x = vehicle.state.x
    current_y = vehicle.state.y

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
        valid = True
        for other in env.vehicle_manager.get_all_vehicles():
            if other.id == vehicle_id:
                continue
            g = other.get_current_goal()
            if g and np.hypot(new_x - g.x, new_y - g.y) < 10.0:
                valid = False
                break

        if valid:
            # 적절한 위치를 찾았으면 목적지 생성
            # 차량별 다른 색상 사용
            colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255)]
            goal_color = colors[vehicle_id % len(colors)]
            env.add_goal_for_vehicle(vehicle_id, new_x, new_y, target_yaw, 2.0, goal_color)
            break

# ==============
# Manual Control With Goal
# ==============
def manual_control_with_goal():
    """목표가 있는 차량 수동 제어"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print_basic_controls("Manual Control With Goal")

    # 초기 목적지 생성 (각 차량마다)
    for i in range(env.num_vehicles):
        create_random_goal(env, i, min_distance=50.0, max_distance=100.0)

    running = True
    while running:
        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 환경 스텝
        _, _, done, info = env.step(action)

        # 렌더링
        env.render()

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 종료 시 잠시 대기 후 재시작
        if done:
            if info.get('all_collision', False) or info.get('all_reached_target', False):
                pygame.time.wait(1000)  # 1초 대기
                env.reset()
                # 초기 목적지 다시 생성
                for i in range(env.num_vehicles):
                    create_random_goal(env, i, min_distance=50.0, max_distance=100.0)

    # 환경 종료
    env.close()

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
            if info.get('all_collision', False):
                pygame.time.wait(1000)  # 1초 대기
                env.reset()
                create_test_course(env.get_obstacle_manager(), current_course)

    # 환경 종료
    env.close()
