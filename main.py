# -*- coding: utf-8 -*-
import pygame
from src.examples.env_examples import manual_control, manual_control_with_goal, manual_control_with_obstacles, dynamic_obstacles_test
from src.ui.menu import MainMenu

# ==============
# Main Function
# ==============
if __name__ == "__main__":
    running = True

    while running:
        # 메인 메뉴 생성 및 표시
        menu = MainMenu()
        selected_option = menu.show_main_menu()

        # 선택된 옵션에 따라 실행
        if selected_option == 'manual':
            manual_control()
        elif selected_option == 'obstacles':
            manual_control_with_obstacles()
        elif selected_option == 'goal':
            manual_control_with_goal()
        elif selected_option == 'dynamic_obstacles':
            dynamic_obstacles_test()
        elif selected_option == 'quit':
            running = False
        # 메뉴로 돌아가서 반복

    # 프로그램 종료 시 pygame 정리
    pygame.quit()
