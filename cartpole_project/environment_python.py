import pygame
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np



width = 360
height = 360

fps = 30

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)





class Player(pygame.sprite.Sprite):

    # sprite for the player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20)) # 20,20 bir nesne yarattık
        self.image.fill(blue) # mavi ile doldurduk
        self.rect = self.image.get_rect() # kare içine aldık
        self.rect.centerx = (width/2) # ekranın ortasına koyduk
        self.rect.bottom = height - 1
        self.speed_x = 0 # y eksenindeki hızı
        self.speed_y = 0



    def update(self):
        keystate = pygame.key.get_pressed()

        if keystate[pygame.K_LEFT]:
            self.speed_x = -4

        elif keystate[pygame.K_RIGHT]:
            self.speed_x = 4

        else :
            self.speed_x = 0

        self.rect.x += self.speed_x

        if self.rect.right > width:
            self.rect.right = width

        elif self.rect.left < 0:
            self.rect.left = 0

    def getCoordinates(self):
        return self.rect.x, self.rect.y



class Enemy(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(red)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0,width - self.rect.width)
        self.rect.y = random.randrange(2,6)
        self.speedx = 0
        self.speedy = 3


    def update(self):

        self.rect.x += self.speedx
        self.rect.y += self.speedy

        if self.rect.top > height + 10:
            self.rect.x = random.randrange(0, width - self.rect.width)
            self.rect.y = random.randrange(2, 6)


class DQLAgent:
    def __init__(self, env):
        # Hiperparametreler
        self.state_size = 4
        self.action_size = 3
        self.gamma = 0.95  # Gelecek ödüllerin indirim oranı
        self.learning_rate = 0.001  # Öğrenme hızı
        self.epsilon = 1.0  # Keşif oranı
        self.epsilon_decay = 0.995  # Keşif oranının zamanla azalması
        self.epsilon_min = 0.01  # Minimum keşif oranı
        self.memory = deque(maxlen=1000)  # Ajanın deneyimlerinin saklandığı bellek
        self.model = self.build_model()  # Q-learning modelini oluştur

    def build_model(self):
        """Derin Q-öğrenme için yapay sinir ağı modeli oluşturur."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Deneyimi belleğe kaydeder."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Ajanın, keşif mi yoksa sömürü mü yapacağına karar verir."""
        pass

    def replay(self, batch_size):
        """Modeli bellekteki deneyimlerle yeniden eğitir."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)

    def update_epsilon(self):
        """Keşif oranını (epsilon) zamanla azaltır."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



# initiliaze pygame and create window

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('RL Game')
clock = pygame.time.Clock()

# sprite
all_sprite = pygame.sprite.Group() # sprite grubu oluşturduk
enemy = pygame.sprite.Group()
player = Player() # player objesini yarattık
m1 = Enemy()
m2 = Enemy()

all_sprite.add(player) # içerisine ekliyorum
all_sprite.add(m1)
all_sprite.add(m2)
enemy.add(m1)
enemy.add(m2)


# game loop
running = True
while running:
    # keep loop running at the right speed
    clock.tick(fps)

    # procces input
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False




    # update
    all_sprite.update()

    hits = pygame.sprite.spritecollide(player, enemy, False,pygame.sprite.collide_rect)
    if hits:
        running = False


    # draw / render (show)
    screen.fill(white)
    all_sprite.draw(screen)
    # after drawing flip display
    pygame.display.flip()


pygame.quit()
