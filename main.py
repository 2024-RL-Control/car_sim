# -*- coding: utf-8 -*-
import pygame
from src.examples.env_examples import manual_control, manual_control_with_goal, manual_control_with_obstacles
from src.examples.env_rl import BasicRLDrivingEnv
from src.ui.menu import MainMenu
import cProfile
import pstats
import io

# ==============
# Main Function
# ==============
if __name__ == "__main__":
    running = True
    is_profiling = False

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
        elif selected_option == 'basic_rl_training(sac)':
            env = BasicRLDrivingEnv()

            if is_profiling:
                profiler = cProfile.Profile()
                profiler.enable()

            env.learn(algorithm='sac')

            if is_profiling:
                profiler.disable()

            if is_profiling:
                s = io.StringIO()
                sortby = pstats.SortKey.CUMULATIVE # 'tottime', 'calls', 'cumulative'
                ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
                ps.print_stats(30)
                print("\n=== Profiling Results (Top 30) ===")
                print(s.getvalue())
                print("=== End of Profiling ===\n")
        elif selected_option == 'basic_rl_training(ppo)':
            env = BasicRLDrivingEnv()
            env.learn(algorithm='ppo')
        elif selected_option == 'basic_rl_testing':
            env = BasicRLDrivingEnv()
            env.test(algorithm='sac')

        elif selected_option == 'quit':
            running = False
        # 메뉴로 돌아가서 반복

    # 프로그램 종료 시 pygame 정리
    pygame.quit()
