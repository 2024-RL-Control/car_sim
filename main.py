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

    print("=== Vehicle Simulator ===")
    print("📌 장애물 포함 글로벌 경로 테스트 실행")
    
    # 장애물 리스트 추가
    obstacles = [[200, 200, 50], [400, 300, 50], [600, 150, 70],[100,100,30]] #장애물 3개 통과과

    # ✅ 차량 현재 위치를 start_pos로 설정
    start_pos = (env.state.x, env.state.y)
    goal_pos = (1000, 100)
    env.set_global_path(start_pos, goal_pos, obstacles)

    running = True
    while running:
        action = np.zeros(2)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        action = env._handle_keyboard_input()

        # 환경 스텝 실행
        _, _, done, _ = env.step(action)

        # 환경 렌더링
        env.render()

        if done:
            break

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