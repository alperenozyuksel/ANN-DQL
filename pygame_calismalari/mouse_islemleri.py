import pygame
import random
import numpy as np

WIDTH = 1920
HEIGHT = 1080

BLACK = (0, 0, 0)
SPEED = 5
FPS = 60
WHITE = (255, 255, 255)
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
monster = pygame.image.load('png/monster.png')
monster_rect = monster.get_rect()

coin = pygame.image.load('png/coin.png')
coin_rect = coin.get_rect()

is_jumping = False

monster_rect.topleft = (WIDTH / 2, 1060)
durum = True

while durum:

    pygame.init()

    is_jumping = False

    tus = pygame.key.get_pressed()



    if tus[pygame.K_LEFT]:
        monster_rect.x -= SPEED

    if tus[pygame.K_RIGHT]:
        monster_rect.x += SPEED

    if tus[pygame.K_SPACE]:
        monster_rect.y -= 25
        is_jumping = True

    if is_jumping == False:
        monster_rect.y += SPEED

    if monster_rect.bottom > HEIGHT:
        monster_rect.bottom = HEIGHT

    if monster_rect.right > WIDTH:
        monster_rect.right = WIDTH

    if monster_rect.left < 0:
        monster_rect.left = 0

    if monster_rect.top < 0:
        monster_rect.top = 0

    if monster_rect.colliderect(coin_rect):
        print("COIN")
        coin_rect.x = random.randint(0, WIDTH-100)
        coin_rect.y = random.randint(0, HEIGHT-100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            durum = False



    WINDOW.fill(BLACK)
    WINDOW.blit(monster, monster_rect)
    WINDOW.blit(coin, coin_rect)

    pygame.display.update()
    clock.tick(FPS)
