import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, verbose=1)

print("Training started...")
model.learn(total_timesteps=1_000_000)
model.save("ppo_lunar_lander_model")
print("Training complete! Model saved.")