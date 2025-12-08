import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def train():
    env = make_vec_env("LunarLander-v3", n_envs=4)                                          # n_envs= Anzahl parallele Environments
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_lander_tensorboard/")
    model.learn(total_timesteps=910_000)
    model.save("ppo_lander_v3")

def evaluate():
    model = PPO.load("ppo_lander_v3")
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()

def main():
    #train()
    evaluate()

if __name__ == "__main__":
    main()