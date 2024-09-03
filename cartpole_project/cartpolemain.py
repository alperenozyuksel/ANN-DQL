import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    def __init__(self, env):
        # Hiperparametreler
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
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
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()  # Keşif: Rastgele bir eylem seç
        act_values = self.model.predict(state)  # Sömürü: Öğrenilen modele göre eylem seç
        return np.argmax(act_values[0])

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

if __name__ == '__main__':
    # Ortamı ve ajanı başlat
    env = gym.make('CartPole-v1', render_mode='human')  # 'render_mode' parametresi eklendi
    agent = DQLAgent(env)
    episodes = 1000

    for episode in range(episodes):
        # Ortamı başlat ve başlangıç durumunu al
        state, _ = env.reset()  # Ortamın yeniden başlatılması
        state = np.reshape(state, [1, agent.state_size])
        time = 0
        batch_size = 8

        while True:
            # Ortamı görselleştirmek için render et
            env.render()

            # Ajan bir eylem gerçekleştirir
            action = agent.act(state)

            # Ortam eylemi işler ve sonraki durum ve ödülü döndürür
            next_state, reward, done, truncated, _ = env.step(action)  # Tüm 5 değeri aç
            next_state = np.reshape(next_state, [1, agent.state_size])

            # Deneyimi belleğe kaydet
            agent.remember(state, action, reward, next_state, done)

            # Durumu güncelle
            state = next_state

            # Ajanı replay ile eğit
            agent.replay(batch_size)

            # Keşif oranını azalt
            agent.update_epsilon()

            time += 1
            if done or truncated:  # Eğer ortam bitmiş veya kesilmişse
                print(f"Episode: {episode+1}/{episodes}, Time: {time}, Epsilon: {agent.epsilon:.2f}")
                break

    # İşlem bitince ortamı kapat
    env.close()
