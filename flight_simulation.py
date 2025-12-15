# flight_simulation.py
import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import glob

from airplane_alt_env import AirplaneAltitudeEnv

BASE_DIR = "./airplane_project_data"
MODEL_DIR = os.path.join(BASE_DIR, "models")
VECNORM_DIR = os.path.join(BASE_DIR, "vecnormalize")


def get_latest_run_dir():
    if not os.path.exists(MODEL_DIR):
        return None
    run_paths = glob.glob(os.path.join(MODEL_DIR, "run_*"))
    if not run_paths:
        return None
    run_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]), reverse=True)
    return run_paths[0]


def get_latest_model_and_vecnorm():
    run_dir = get_latest_run_dir()
    if not run_dir:
        return None, None

    zips = glob.glob(os.path.join(run_dir, "*.zip"))
    if not zips:
        return None, None
    model_path = max(zips, key=os.path.getctime)

    run_name = os.path.basename(run_dir)
    vecnorm_path = os.path.join(VECNORM_DIR, f"{run_name}_vecnorm.pkl")
    if not os.path.exists(vecnorm_path):
        vecnorm_path = None

    return model_path, vecnorm_path


def draw_overlay(screen, env, autopilot, manual_thr):
    if not screen:
        return
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont("Consolas", 18, bold=True)

    w, h = screen.get_size()
    bg = pygame.Surface((300, 180))
    bg.set_alpha(180)
    bg.fill((0, 0, 0))
    screen.blit(bg, (w - 310, 10))
    pygame.draw.rect(screen, (255, 255, 255), (w - 310, 10, 300, 180), 2)

    c_auto = (0, 255, 0) if autopilot else (255, 100, 100)
    lines = [
        (f"MODE: {'AUTOPILOT' if autopilot else 'MANUAL'}", c_auto),
        (f"Alt: {env.altitude:.1f} m", (255, 255, 255)),
        (f"Tgt: {env.target_altitude:.0f} m", (255, 255, 0)),
        (f"Spd: {env.speed * 3.6:.0f} km/h", (255, 255, 255)),
        (f"Vz:  {env.vertical_speed:.1f} m/s", (255, 255, 255)),
        (f"Thr: {manual_thr:.0f}%", (255, 255, 255)),
        (f"Pit: {np.rad2deg(env.pitch):.1f}", (255, 255, 255)),
    ]

    for i, (txt, col) in enumerate(lines):
        img = font.render(txt, True, col)
        screen.blit(img, (w - 295, 20 + i * 22))


def run():
    model_path, vecnorm_path = get_latest_model_and_vecnorm()
    print(f"Lade: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")

    if not model_path or not vecnorm_path:
        raise FileNotFoundError(
            "Model oder VecNormalize nicht gefunden. "
            "Stelle sicher, dass du nach dem Training sowohl model_final.zip als auch run_X_vecnorm.pkl gespeichert hast."
        )

    # Model laden
    model = PPO.load(model_path)

    # VecEnv + VecNormalize exakt wie im Training (n_envs=1)
    def make_env():
        return AirplaneAltitudeEnv(render_mode="human", enable_scenarios=True, max_episode_steps=3000)

    venv = DummyVecEnv([make_env])

    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False          # keine Running-Stats weiter updaten
    venv.norm_reward = False       # Rewards fürs Render egal; und stabiler

    # Zugriff auf echte Env für Rendering/Overlay/Target-Änderungen
    base_env = venv.venv.envs[0]


    # Reset (liefert normierte Obs in Batch-Form)
    obs = venv.reset()

    # --- FIX: pygame initialisieren ---
    pygame.init()
    base_env.render(flip=True)

    running = True
    autopilot = False
    manual_thr = 50.0
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    running = True
    autopilot = False
    manual_thr = 50.0
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    autopilot = not autopilot
                if event.key == pygame.K_r:
                    obs = venv.reset()
                if event.key == pygame.K_w:
                    base_env.target_altitude += 100
                if event.key == pygame.K_s:
                    base_env.target_altitude -= 100

        keys = pygame.key.get_pressed()
        pitch_in = 0.0
        if keys[pygame.K_UP]:
            pitch_in = 1.0
        if keys[pygame.K_DOWN]:
            pitch_in = -1.0
        if keys[pygame.K_LSHIFT]:
            manual_thr = min(100, manual_thr + 1)
        if keys[pygame.K_LCTRL]:
            manual_thr = max(0, manual_thr - 1)

        if autopilot:
            action, _ = model.predict(obs, deterministic=True)  # action ist (1,2) oder (2,)
            action = np.array(action, dtype=np.float32).reshape(1, -1)
            manual_thr = ((action[0, 1] + 1) / 2) * 100
        else:
            action = np.array([[pitch_in, (manual_thr / 50.0) - 1.0]], dtype=np.float32)

        obs, rewards, dones, infos = venv.step(action)

        if bool(dones[0]):
            obs = venv.reset()

        # Render / Overlay wie zuvor (Darstellung bleibt gleich)
        base_env.render(flip=False)
        draw_overlay(base_env.screen, base_env, autopilot, manual_thr)
        pygame.display.flip()
        clock.tick(60)

    base_env.close()
    pygame.quit()


if __name__ == "__main__":
    run()