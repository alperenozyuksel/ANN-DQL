import gym

# Ortamı oluştur ve render_mode parametresini belirt
env = gym.make("Taxi-v3", render_mode="ansi")

# Ortamı başlat
env.reset()

# Ortamı render et
print(env.render())

# Ortamı kapat
env.close()
