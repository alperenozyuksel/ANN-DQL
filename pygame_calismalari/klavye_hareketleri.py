import pygame

pygame.init()

WIDTH = 750
HEIGHT = 680
BLACK = (0, 0, 0)
window = pygame.display.set_mode((WIDTH, HEIGHT))

SPEED = 10

monster = pygame.image.load('monster.png')
monster_rect = monster.get_rect()
monster_rect.topleft = (WIDTH/2 , HEIGHT/2)
durum = True

while durum:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            durum = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                monster_rect.x -= SPEED

            if event.key == pygame.K_RIGHT:
                monster_rect.x += SPEED

            if event.key == pygame.K_UP:
                monster_rect.y -= SPEED

            if event.key == pygame.K_DOWN:
                monster_rect.y += SPEED


    window.fill(BLACK)
    window.blit(monster, monster_rect)
    pygame.display.update()




pygame.quit()