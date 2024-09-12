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
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('png/monster.png')
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 2
        self.hareket = True

    def update(self):
        if self.hareket:
            self.rect.x += self.speed
        if self.rect.right > WIDTH:
            self.hareket = False
        if not self.hareket:
            self.rect.x -= self.speed
        if self.rect.left < 0:
            self.rect.x += self.speed
            self.hareket = True

        # Rastgele düşman mermisi oluşturma
        if random.randint(1, 1000) <= 1:  # Yaklaşık %2 olasılıkla mermi at
            enemy_bullet = EnemyBullet(self.rect.centerx, self.rect.bottom)
            enemy_bullet_group.add(enemy_bullet)


class EnemyBullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('png/bullet_3.png')
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 5

    def update(self):
        self.rect.y += self.speed
        if self.rect.y > HEIGHT:  # Ekranın dışına çıkarsa yok edilir
            self.kill()


class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('png/bullet_3.png')
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 5

    def update(self):
        self.rect.y -= self.speed
        if self.rect.y < 0:  # Mermi ekranın dışına çıkarsa yok edilir
            self.kill()


# Gruplar
player_group = pygame.sprite.Group()
enemy_group = pygame.sprite.Group()
bullet_group = pygame.sprite.Group()
enemy_bullet_group = pygame.sprite.Group()

player = Player()
player_group.add(player)



for _ in range(10):
    enemy = Enemy(_ * 100, 100)
    enemy_group.add(enemy)
    enemy1 = Enemy(_ * 100, 150)
    enemy_group.add(enemy1)





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
        bullet = Bullet(player.rect.centerx - 12, player.rect.y - 20)
        bullet_group.add(bullet)
        last_shot_time = current_time  # Son ateşleme zamanı güncellenir

    # Çarpışma kontrolü
    for bullet in bullet_group:
        if pygame.sprite.spritecollide(bullet, enemy_group, True):  # True parametresi çarpıştığında enemy'yi siler
            bullet.kill()  # Çarpışma olduğunda mermiyi yok et

    for enemy_bullet in enemy_bullet_group:
        if pygame.sprite.spritecollide(enemy_bullet, player_group, True):  # Oyuncuya çarptığında
            enemy_bullet.kill()  # Çarpışma olduğunda düşman mermisini yok et

    

    # Güncellemeler
    player_group.update()
    enemy_group.update()
    bullet_group.update()
    enemy_bullet_group.update()

    # Çizim
    WINDOW.fill(BLACK)
    player_group.draw(WINDOW)
    enemy_group.draw(WINDOW)
    bullet_group.draw(WINDOW)
    enemy_bullet_group.draw(WINDOW)

    pygame.display.update()
    CLOCK.tick(FPS)

pygame.quit()
