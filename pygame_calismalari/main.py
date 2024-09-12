import pygame
import random

pygame.init()

WIDTH = 720
HEIGHT = 720
FPS = 60
CLOCK = pygame.time.Clock()
BLACK = (0, 0, 0)
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
RED = (255, 0, 0)

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
        self.rect.x = random.randint(0, WIDTH - self.rect.width)
        self.rect.y = random.randint(0, 500)

    def update(self):
        pass


class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('png/bullet.png')
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 5



    def update(self):
        self.rect.y -= self.speed
        if self.rect.y < 0:
            self.kill()  # Mermi ekranın dışına çıkarsa yok edilir


# Gruplar
player_group = pygame.sprite.Group()
enemy_group = pygame.sprite.Group()
bullet_group = pygame.sprite.Group()

player = Player()
enemy = Enemy()
player_group.add(player)
enemy_group.add(enemy)

fire = False
last_shot_time = 0  # Son ateşleme zamanı

# Oyun döngüsü
durum = True

while durum:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            durum = False

    # Klavye girişini kontrol et
    key = pygame.key.get_pressed()

    # Mevcut zamanı al
    current_time = pygame.time.get_ticks()

    # 0.2 saniyede bir mermi atılmasına izin ver
    if key[pygame.K_SPACE] and current_time - last_shot_time > 200:
        # Mermi oyuncunun pozisyonuna göre yaratılır
        bullet = Bullet(player.rect.centerx - 12, player.rect.y - 20)
        bullet_group.add(bullet)
        last_shot_time = current_time  # Son ateşleme zamanı güncellenir



    for bullet in bullet_group:
        if pygame.sprite.collide_rect(bullet, enemy):
            bullet.kill()  # Çarpışma olduğunda mermiyi yok et
            enemy.kill()  # Düşmanı yok et
            enemy = Enemy()
            enemy_group.add(enemy)



    # Güncellemeler
    player_group.update()
    enemy_group.update()
    bullet_group.update()

    # Çizim
    WINDOW.fill(BLACK)
    player_group.draw(WINDOW)
    enemy_group.draw(WINDOW)
    bullet_group.draw(WINDOW)

    pygame.draw.rect(WINDOW, RED, player.rect, 2)  # Player hitbox
    pygame.draw.rect(WINDOW, RED, enemy.rect, 2)  # Enemy hitbox
    for bullet in bullet_group:
        pygame.draw.rect(WINDOW, RED, bullet.rect, 2)  # Bullet hitbox

    pygame.display.update()
    CLOCK.tick(FPS)

pygame.quit()
