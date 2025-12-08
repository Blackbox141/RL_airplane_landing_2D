import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import pygame
import numpy as np

# Importiere dein Environment
from airplane_alt_env import AirplaneAltitudeEnv

# --- KONFIGURATION ---
LOG_DIR = "./ppo_airplane_alt_tensorboard/"
MODEL_DIR = "./saved_models/"


def get_next_run_id(log_dir):
    max_id = 0
    if os.path.exists(log_dir):
        for sub_dir in os.listdir(log_dir):
            if sub_dir.startswith("PPO_"):
                try:
                    run_id = int(sub_dir.split("_")[1])
                    if run_id > max_id:
                        max_id = run_id
                except ValueError:
                    pass
    return max_id + 1


def make_env():
    env = AirplaneAltitudeEnv(render_mode="none")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1200)
    return env


def draw_evaluation_overlay(screen, env, obs, model_name="Unknown"):
    if not screen: return

    if not pygame.font.get_init(): pygame.font.init()
    font = pygame.font.SysFont("Consolas", 18, bold=True)

    # --- FIX: Echte Werte aus dem ENV holen, nicht aus OBS ---
    # obs[4] ist normalisiert (z.B. 0.1), env.vx ist echt (z.B. 75.0 m/s)
    speed_kmh = env.vx * 3.6

    throttle_pct = env.throttle_actual * 100.0
    pitch_deg = np.rad2deg(env.pitch)

    # Farben
    c_bg = (0, 0, 0, 180)
    c_txt = (255, 255, 255)
    c_green = (0, 255, 0)
    c_blue = (100, 200, 255)

    w, h = screen.get_size()
    panel_w = 300
    panel_h = 170
    panel_x = w - panel_w - 10
    panel_y = 10

    s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    s.fill(c_bg)
    screen.blit(s, (panel_x, panel_y))
    pygame.draw.rect(screen, c_txt, (panel_x, panel_y, panel_w, panel_h), 2)

    lines = [
        ("EVALUATION MODE", c_green),
        (f"Model: {model_name}", c_blue),
        (f"Altitude:  {env.altitude:.1f} m", c_txt),
        (f"Target:    {env.target_altitude:.1f} m", c_txt),
        (f"Speed:     {speed_kmh:.0f} km/h", c_txt),
        (f"Throttle:  {throttle_pct:.1f} %", c_txt),
        (f"Pitch:     {pitch_deg:.1f} deg", c_txt),
    ]

    for i, (text, col) in enumerate(lines):
        img = font.render(text, True, col)
        screen.blit(img, (panel_x + 15, panel_y + 15 + i * 20))


def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    run_id = get_next_run_id(LOG_DIR)
    print(f"--- STARTE TRAINING RUN #{run_id} ---")

    env = make_vec_env(make_env, n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR,
    )

    print("Starte Training (5 Mio Steps)...")
    model.learn(total_timesteps=500_000, tb_log_name="PPO")

    save_name = f"model_{run_id}"
    save_path = os.path.join(MODEL_DIR, save_name)

    print(f"Speichere Modell als: {save_path}.zip")
    model.save(save_path)
    model.save(os.path.join(MODEL_DIR, "model_latest"))
    print("Training abgeschlossen!")


def evaluate():
    model_path = os.path.join(MODEL_DIR, "model_latest.zip")

    if not os.path.exists(model_path):
        print(f"FEHLER: Datei {model_path} nicht gefunden!")
        print("Bitte führe erst 'train()' aus.")
        return

    print("Lade 'model_latest.zip' und starte Evaluation...")
    model = PPO.load(model_path)

    env = AirplaneAltitudeEnv(render_mode="human")
    obs, info = env.reset()

    env.render(flip=False)
    pygame.display.flip()

    step = 0
    clock = pygame.time.Clock()

    print("Drücke ESC im Fenster oder CTRL+C in der Konsole zum Abbrechen.")

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running: break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                print(f"Crash bei Step {step}! Resetting...")
                obs, info = env.reset()

            step += 1

            env.render(flip=False)
            if env.screen:
                draw_evaluation_overlay(env.screen, env, obs, model_name="LATEST")
            pygame.display.flip()

            clock.tick(60)

    except KeyboardInterrupt:
        print("\nAbbruch durch Benutzer.")
    finally:
        env.close()


def main():
    # 1. Training (Kommentieren wenn nicht benötigt)
    train()

    # 2. Evaluation
    evaluate()


if __name__ == "__main__":
    main()