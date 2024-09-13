import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import cv2

# window size
WIDTH = 360
HEIGHT = 360
FPS = 30

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20, 20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 1
        self.speedx = 0

    def update(self, action):
        self.speedx = 0
        if action == 0:
            self.speedx = -7
        elif action == 1:
            self.speedx = 7

        self.rect.x += self.speedx

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0

    def getCoordinates(self):
        """Oyuncunun mevcut x ve y koordinatlarını döndürür."""
        return (self.rect.x, self.rect.y)

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(2, 6)
        self.speedy = 6

    def update(self):
        self.rect.y += self.speedy
        if self.rect.top > HEIGHT + 10:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(2, 6)

    def getCoordinates(self):
        """Düşmanın mevcut x ve y koordinatlarını döndürür."""
        return (self.rect.x, self.rect.y)

class DQLAgent:
    def __init__(self):
        self.state_size = 2
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state).reshape(1, -1)  # State'i (1, 2) şeklinde yeniden şekillendiriyoruz
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).reshape(1, -1)  # State ve next_state biçimlendirilmesi
            next_state = np.array(next_state).reshape(1, -1)  # Biçimlendirilmiş
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, verbose=0)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Env:
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy()
        self.all_sprite.add(self.m1)
        self.enemy.add(self.m1)

        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()

        # OpenCV penceresini başlat
        self.img_height, self.img_width = 600, 800
        self.img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def findDistance(self, a, b):
        return a - b

    def step(self, action):
        state_list = []
        self.player.update(action)
        self.enemy.update()

        player_state = self.player.getCoordinates()
        m1_state = self.m1.getCoordinates()

        state_list.append(self.findDistance(player_state[0], m1_state[0]))
        state_list.append(self.findDistance(player_state[1], m1_state[1]))

        return [state_list]

    def initialStates(self):
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy()
        self.all_sprite.add(self.m1)
        self.enemy.add(self.m1)

        self.reward = 0
        self.total_reward = 0
        self.done = False

        state_list = []
        player_state = self.player.getCoordinates()
        m1_state = self.m1.getCoordinates()

        state_list.append(self.findDistance(player_state[0], m1_state[0]))
        state_list.append(self.findDistance(player_state[1], m1_state[1]))

        return [state_list]

    def run(self):
        state = self.initialStates()
        running = True
        batch_size = 100000
        while running:
            self.reward = 1
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward

            hits = pygame.sprite.spritecollide(self.player, self.enemy, False, pygame.sprite.collide_circle)

            if hits:
                self.reward = 10
                self.total_reward += self.reward
                self.m1.kill()
                self.m1 = Enemy()
                self.all_sprite.add(self.m1)
                self.enemy.add(self.m1)
                print("Total reward: ", self.total_reward)

            elif self.m1.rect.top > HEIGHT:
                self.reward = -50
                self.total_reward += self.reward
                print(f"Missed the box! Total reward: {self.total_reward}")

            self.agent.remember(state, action, self.reward, next_state, self.done)
            state = next_state
            self.agent.replay(batch_size)
            self.agent.adaptiveEGreedy()

            # PyGame ekranı güncelleme
            screen.fill(BLACK)
            self.all_sprite.draw(screen)
            pygame.display.flip()

            # OpenCV ile ağırlık görselleştirme
            self.visualize_weights_with_opencv(action)

        pygame.quit()

    def visualize_weights_with_opencv(self, current_action):
        # Ağın katmanlarındaki ağırlıkları çek
        weights = self.agent.model.get_weights()

        # Görselleştirme için boş bir resim oluştur
        self.img_height, self.img_width = 800, 1200  # Pencereyi genişlettik
        self.img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # Katmanlar arası boşluk hesaplaması
        num_layers = len(weights) // 2
        layer_spacing = self.img_width // (num_layers + 2)
        neuron_spacing = self.img_height // (max([w.shape[0] for w in weights[::2]]) + 1)

        # Katmanların sayısı ve her katmandaki nöronların sayısı
        layer_sizes = [weights[i].shape[0] for i in range(0, len(weights), 2)]
        layer_sizes.append(weights[-2].shape[1])  # Son katmanın çıkış boyutu

        # Katmanları çiz
        for i, size in enumerate(layer_sizes[:-1]):
            next_size = layer_sizes[i + 1]

            # Katmandaki her nöron için
            for j in range(size):
                x1, y1 = (i + 1) * layer_spacing, (
                            self.img_height - (size - 1) * neuron_spacing) // 2 + j * neuron_spacing
                cv2.circle(self.img, (x1, y1), 15, (255, 255, 255), -1)  # Nöronları beyaz yuvarlakla göster

                # Bir sonraki katmandaki her nöron için bağlantı çiz
                for k in range(next_size):
                    x2, y2 = (i + 2) * layer_spacing, (
                                self.img_height - (next_size - 1) * neuron_spacing) // 2 + k * neuron_spacing

                    # Ağırlık değeri
                    weight = weights[2 * i][j, k]  # Ağırlıkları çekiyoruz

                    # Ağırlık kalınlık ve renk (negatif mavi, pozitif kırmızı)
                    thickness = max(1, int(2 * np.abs(weight)))  # Kalınlık için minimum 1
                    color = (0, 0, 255) if weight > 0 else (255, 0, 0)  # Kırmızı pozitif, mavi negatif

                    cv2.line(self.img, (x1, y1), (x2, y2), color, thickness)

        # Çıkış nöronlarını çiz
        action_names = ["Sol", "Sag", "Aksiyon Yok"]
        output_x = (len(layer_sizes)) * layer_spacing
        for j in range(layer_sizes[-1]):
            y1 = (self.img_height - (layer_sizes[-1] - 1) * neuron_spacing) // 2 + j * neuron_spacing
            # Aktif eylemi gösteren yeşil renk
            color = (0, 255, 0) if j == current_action else (128, 128, 128)  # Aktif eylem yeşil, diğerleri gri
            cv2.circle(self.img, (output_x, y1), 15, color, -1)  # Çıkış nöronlarını renklendir

            # Nöron isimlerini ekle (beyaz renkte) ve biraz daha sağa kaydır
            cv2.putText(self.img, action_names[j], (output_x + 15, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1,
                        cv2.LINE_AA)

        # Ağırlık çizimlerini göster
        cv2.imshow('Neural Network Weights', self.img)
        cv2.waitKey(1)  # 1 milisaniye bekler, bu sayede oyun döngüsünü engellemez


if __name__ == "__main__":
    env = Env()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Game with Live Neural Network Visualization")
    clock = pygame.time.Clock()

    env.run()