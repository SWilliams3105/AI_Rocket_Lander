import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_lunar_lander_model")
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

for _ in range(2000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
