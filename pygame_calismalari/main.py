import pygame
import random


pygame.init()

WIDTH = 720
HEIGHT = 720
FPS = 60
CLOCK = pygame.time.Clock()
BLACK = (0,0,0)
WINDOW =pygame.display.set_mode((WIDTH,HEIGHT))

class Player(pygame.sprite.Sprite):

    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('png/ufoo.png')
        self.rect = self.image.get_rect()
        self.rect.x = WIDTH / 2
        self.rect.y = 620
        self.speed = 5

    def update(self):

        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            self.rect.x -= self.speed

        if key[pygame.K_RIGHT]:
            self.rect.x += self.speed

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH

        if self.rect.left < 0:
            self.rect.left = 0

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('png/monster.png')
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, 720)  # Ekran genişliğine göre
        self.rect.y = random.randint(0, 500)  # Ekran yüksekliğine göre

    def update(self):
        pass


class Bullet(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load('png/bullet.png')
        self.rect = self.image.get_rect()

player_group = pygame.sprite.Group()
enemy_group = pygame.sprite.Group()

player = Player()
enemy = Enemy()

player_group.add(player)
enemy_group.add(enemy)

durum = True

while durum:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            durum = False


    player_group.update()
    enemy_group.update()

    WINDOW.fill(BLACK)

    player_group.draw(WINDOW)
    enemy_group.draw(WINDOW)

    pygame.display.update()
    CLOCK.tick(FPS)




pygame.quit()



