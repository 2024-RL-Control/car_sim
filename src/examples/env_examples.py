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
# Manual Control With Goal
# ==============
def create_random_goal(env, vehicle_id, min_distance=25.0, max_distance=50.0):
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

def manual_control_with_goal():
    """목표가 있는 차량 수동 제어"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print_basic_controls("Manual Control With Goal")

    # 초기 목적지 생성 (각 차량마다)
    for i in range(env.num_vehicles):
        create_random_goal(env, i)

    running = True
    while running:
        # 키보드 입력 처리 및 액션 생성
        action = env.handle_keyboard_input()

        # 환경 스텝
        _, _, done, _, info = env.step(action)

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
            pygame.time.wait(1000)  # 1초 대기
            env.reset()
            # 초기 목적지 다시 생성
            for i in range(env.num_vehicles):
                create_random_goal(env, i)

    # 환경 종료
    env.close()
