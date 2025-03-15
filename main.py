# -*- coding: utf-8 -*-
import pygame
import numpy as np
from env import CarSimulatorEnv

# ==============
# Reinforcement Learning (SB3)
# ==============
def rl_learn():
    """강화 학습 수행"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print("=== Vehicle Simulator(RL Learning) ===")
    print("Controls:")
    print("  F1: Toggle Environment Rendering")
    print("  F5: Save state")
    print("  F9: Load state")
    print("  ESC: Quit")

# ==============
# Reinforcement Basic Inference (SB3)
# ==============
def rl_test1():
    """강화 학습 추론(기본)"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print("=== Vehicle Simulator(RL Inference; Basic) ===")
    print("Controls:")
    print("  F1: Toggle Environment Rendering")
    print("  F5: Save state")
    print("  F9: Load state")
    print("  ESC: Quit")

# ==============
# Reinforcement Maze Agent Inference (SB3)
# ==============
def rl_test2():
    """강화 학습 추론(미로 환경)"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print("=== Vehicle Simulator(RL Inference; Maze) ===")
    print("Controls:")
    print("  F1: Toggle Environment Rendering")
    print("  F5: Save state")
    print("  F9: Load state")
    print("  ESC: Quit")

# ==============
# Reinforcement Multi Inference (SB3)
# ==============
def rl_test3():
    """강화 학습 추론(다중 에이전트)"""
    # 환경 초기화
    env = CarSimulatorEnv()
    env.reset()

    # 안내 메시지 출력
    print("=== Vehicle Simulator(RL Inference; Multi Agent) ===")
    print("Controls:")
    print("  F1: Toggle Environment Rendering")
    print("  F5: Save state")
    print("  F9: Load state")
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
    print("=== Vehicle Simulator ===")
    print("Controls:")
    print("  Arrow Keys / WASD: Drive the vehicle")
    print("  Space: Handbrake")
    print("  T: Toggle tire marks")
    print("  C: Toggle camera follow")
    print("  H: Toggle debug info")
    print("  F: Clear all tire marks")
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
    print("  T: Toggle tire marks")
    print("  C: Toggle camera follow")
    print("  H: Toggle debug info")
    print("  F: Clear all tire marks")
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
        "4. Quit"
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
                elif event.key == pygame.K_4 or event.key == pygame.K_ESCAPE:
                    # 종료
                    running = False

    pygame.quit()

# ==============
# Main Function
# ==============
if __name__ == "__main__":
    # 메인 메뉴 표시
    show_main_menu()