#train_airplane.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from airplane_landing_env import AirplaneLandingEnv


def make_env():
    # wird von make_vec_env benutzt
    return AirplaneLandingEnv(render_mode="none")


def train():
    # Vektorisierte Envs für schnelleres Training
    env = make_vec_env(make_env, n_envs=4)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_airplane_tensorboard/")
    model.learn(total_timesteps=910_000)
    model.save("ppo_airplane_landing")


def evaluate():
    model = PPO.load("ppo_airplane_landing")
    env = AirplaneLandingEnv(render_mode="human")

    obs, info = env.reset()
    done = False

    while not done:
        # greedy Policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


def main():
    # Schritt 1: train() ausführen, damit das Modell gelernt wird
    # Schritt 2: train() auskommentieren und evaluate() laufen lassen
    train()
    evaluate()


if __name__ == "__main__":
    main()