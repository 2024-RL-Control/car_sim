# -*- coding: utf-8 -*-
import pygame

class MainMenu:
    """차량 시뮬레이터 메인 메뉴 클래스"""

    def __init__(self):
        """메인 메뉴 초기화"""
        # Pygame 초기화
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Vehicle Simulator")

        # 폰트 설정
        self.title_font = pygame.font.Font(None, 64)
        self.menu_font = pygame.font.Font(None, 36)

        # 메뉴 항목
        self.menu_items = [
            "1. Goal Driving",
            "2. Basic RL Driving Training(sac)",
            "3. Basic RL Driving Training(ppo)",
            "4. Basic RL Driving Testing",
            "5. Classic Control Testing",
            "6. Basic RL vs Classic Control",
            "7. Quit"
        ]

    def show_main_menu(self):
        """
        메인 메뉴 표시 및 선택 처리

        Returns:
            str: 선택한 메뉴 항목 ('manual', 'obstacles', 'goal', 'basic_rl_training', 'quit')
        """
        running = True
        while running:
            # 화면 지우기
            self.screen.fill((0, 0, 0))

            # 타이틀 표시
            title_text = self.title_font.render("Vehicle Simulator", True, (255, 255, 255))
            title_rect = title_text.get_rect(center=(400, 100))
            self.screen.blit(title_text, title_rect)

            # 메뉴 항목 표시
            for i, item in enumerate(self.menu_items):
                item_text = self.menu_font.render(item, True, (200, 200, 200))
                item_rect = item_text.get_rect(center=(400, 200 + i * 50))
                self.screen.blit(item_text, item_rect)

            # 화면 업데이트
            pygame.display.flip()

            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return 'quit'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        # 목표 주행 모드
                        pygame.quit()
                        return 'goal'
                    elif event.key == pygame.K_2:
                        # 자율주행 학습 모드
                        pygame.quit()
                        return 'basic_rl_training(sac)'
                    elif event.key == pygame.K_3:
                        # 자율주행 학습 모드
                        pygame.quit()
                        return 'basic_rl_training(ppo)'
                    elif event.key == pygame.K_4:
                        # 자율주행 테스트 모드
                        pygame.quit()
                        return 'basic_rl_testing'
                    elif event.key == pygame.K_5:
                        # 클래식 제어 테스트 모드
                        pygame.quit()
                        return 'classic_control'
                    elif event.key == pygame.K_6:
                        # 자율주행 vs 클래식 제어 비교 모드
                        pygame.quit()
                        return 'basic_rl_vs_classic'
                    elif event.key == pygame.K_7 or event.key == pygame.K_ESCAPE:
                        # 종료
                        running = False
                        return 'quit'

        pygame.quit()
        return 'quit'
