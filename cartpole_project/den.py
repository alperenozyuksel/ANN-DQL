import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import networkx as nx

# window size
WIDTH = 360
HEIGHT = 360
FPS = 30  # how fast game is

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # RGB
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20, 20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, WHITE, self.rect.center, self.radius)
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
        return (self.rect.x, self.rect.y)

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image, WHITE, self.rect.center, self.radius)
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(2, 6)
        self.speedy = 6

    def update(self):
        self.rect.y += self.speedy
        if self.rect.top > HEIGHT + 10:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(2, 6)
            self.speedy = 3

    def getCoordinates(self):
        return (self.rect.x, self.rect.y)

class DQLAgent:
    def __init__(self):
        self.state_size = 2
        self.action_size = 3
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=1000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, verbose=0)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def visualize_weights(self, chosen_action=None):
        # Create a graph for visualization of the network
        layers = [layer.units for layer in self.model.layers if hasattr(layer, 'units')]
        G = nx.DiGraph()

        fig, ax = plt.subplots(figsize=(10, 10))

        # Aksiyonlar için isimler (çıktı katmanı için)
        action_names = ["Left", "Right", "No Move"]  # Bu aksiyonlar, çıkış katmanı nöronlarına karşılık gelir.

        # Pozisyonları hesapla (yani her katman ve düğüm için koordinatlar ayarla)
        pos = {}
        layer_width = 1.0 / (len(layers) - 1)  # Katmanlar arasında yatay mesafe
        for i, layer_size in enumerate(layers):
            for j in range(layer_size):
                pos[f'L{i}N{j}'] = (i * layer_width, j / layer_size)  # Her düğümü yatayda ve dikeyde pozisyonlandır

        # Iterate over layers and neurons to visualize weights
        for i in range(len(layers) - 1):
            for j in range(layers[i]):
                for k in range(layers[i + 1]):
                    weight = np.random.randn()  # Placeholder for real weights (Bunu gerçek ağırlıklarla değiştir)
                    color = 'red' if weight > 0 else 'blue'
                    G.add_edge(f'L{i}N{j}', f'L{i + 1}N{k}', weight=abs(weight), color=color)

        edges = G.edges(data=True)
        colors = [edge[2]['color'] for edge in edges]
        weights = [edge[2]['weight'] for edge in edges]

        # Çıktı katmanını aksiyon isimleriyle etiketle
        output_layer_idx = len(layers) - 1
        for j, action_name in enumerate(action_names):
            node_label = f'L{output_layer_idx}N{j}'
            pos[node_label] = (output_layer_idx * layer_width, j / len(action_names))
            if chosen_action == j:
                nx.draw_networkx_nodes(G, pos, nodelist=[node_label], node_color="green",
                                       node_size=700)  # Seçilen aksiyonu yeşil yap
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=[node_label], node_color="lightblue",
                                       node_size=500)  # Diğerlerini mavi yap

            # Aksiyon isimlerini yazdır
            ax.text(pos[node_label][0], pos[node_label][1] + 0.05, action_name, horizontalalignment='center',
                    fontsize=12)

        # Diğer düğümleri çiz
        nx.draw(G, pos, with_labels=False, node_size=500, edge_color=colors, width=weights, ax=ax)

        plt.show()


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
        batch_size = 2000
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

            chosen_action = self.agent.act(state)  # Hangi aksiyon seçildi
            self.agent.visualize_weights(chosen_action)  # Seçilen aksiyonu görselleştir

            screen.fill(BLACK)
            self.all_sprite.draw(screen)
            pygame.display.flip()

            # Visualization of weights
            self.agent.visualize_weights()




        pygame.quit()

if __name__ == "__main__":
    env = Env()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RL Game with Live Neural Network Visualization")
    clock = pygame.time.Clock()

    env.run()