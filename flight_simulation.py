import pygame
import numpy as np
from stable_baselines3 import PPO
import os

from airplane_alt_env import AirplaneAltitudeEnv

# ==========================================
#  EINSTELLUNGEN
# ==========================================

# Welches Training willst du laden? (0 = "model_latest.zip")
RUN_ID = 103

# Basispfad
BASE_PATH = "/Users/dennis/PycharmProjects/RL Lander V2"
MODEL_FOLDER = os.path.join(BASE_PATH, "saved_models")

# Automatische Pfad-Ermittlung
if RUN_ID == 0:
    MODEL_NAME = "model_latest"
else:
    MODEL_NAME = f"model_{RUN_ID}"

MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME + ".zip")


# ==========================================
#  UI FUNKTIONEN
# ==========================================
def draw_overlay(env, autopilot, sensitivity, manual_thr, manual_pitch, speed_kmh):
    screen = pygame.display.get_surface()
    if not screen: return

    if not pygame.font.get_init(): pygame.font.init()
    font = pygame.font.SysFont("Consolas", 18, bold=True)

    c_bg = (0, 0, 0, 180)
    c_txt = (255, 255, 255)
    c_acc = (0, 255, 0) if autopilot else (255, 100, 100)

    w, h = screen.get_size()
    panel_w = 320
    panel_h = 180
    panel_x = w - panel_w - 10
    panel_y = 10

    s = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    s.fill(c_bg)
    screen.blit(s, (panel_x, panel_y))
    pygame.draw.rect(screen, c_txt, (panel_x, panel_y, panel_w, panel_h), 2)

    # Modell Name anzeigen
    model_disp = f"Run #{RUN_ID}" if RUN_ID > 0 else "LATEST"

    lines = [
        f"MODEL:     {model_disp}",
        f"MODE:      {'AUTOPILOT' if autopilot else 'MANUAL'}",
        f"Sensitiv:  {sensitivity:.1f}",
        f"Throttle:  {manual_thr:.0f}%",
        f"----------------",
        f"Alt:       {env.altitude:.1f} m",
        f"Speed:     {speed_kmh:.0f} km/h",
        f"V-Speed:   {env.vz:.1f} m/s",
    ]

    for i, line in enumerate(lines):
        col = c_acc if "MODE" in line else c_txt
        if "MODEL" in line: col = (100, 200, 255)

        img = font.render(line, True, col)
        screen.blit(img, (panel_x + 15, panel_y + 15 + i * 20))

    if env.altitude <= 0:
        big_font = pygame.font.SysFont("Arial", 50, bold=True)
        txt = big_font.render("CRASH! 'R' to Reset", True, (255, 0, 0))
        rect = txt.get_rect(center=(w // 2, h // 2))
        screen.blit(txt, rect)


# ==========================================
#  MAIN LOOP
# ==========================================
def run():
    pygame.init()

    print(f"Suche Modell: {MODEL_PATH}")

    model = None
    if os.path.exists(MODEL_PATH):
        print(f">>> ERFOLG: Modell '{MODEL_NAME}' geladen.")
        model = PPO.load(MODEL_PATH)
    else:
        print(f">>> WARNUNG: Datei nicht gefunden!")
        print(f"    Erwartet in: {MODEL_FOLDER}")
        print(f"    Bitte stelle sicher, dass 'train_alt.py' ausgeführt wurde")
        print(f"    oder benenne dein altes Modell um in 'model_{RUN_ID}.zip'.")

    # Env Setup
    env = AirplaneAltitudeEnv(render_mode="human")
    obs, _ = env.reset()
    env.render(flip=False)
    pygame.display.flip()

    running = True
    autopilot = False
    sensitivity = 1.0
    manual_thr = 50.0
    crashed = False

    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_a: autopilot = not autopilot
                if event.key == pygame.K_r and crashed:
                    obs, _ = env.reset()
                    crashed = False
                    env.altitude = 300
                if event.key == pygame.K_PERIOD: sensitivity += 0.1
                if event.key == pygame.K_COMMA: sensitivity = max(0.1, sensitivity - 0.1)

        keys = pygame.key.get_pressed()
        pitch_cmd = 0.0
        if keys[pygame.K_UP]: pitch_cmd = 1.0
        if keys[pygame.K_DOWN]: pitch_cmd = -1.0
        if keys[pygame.K_t]: manual_thr = min(100, manual_thr + 1.0)
        if keys[pygame.K_g]: manual_thr = max(0, manual_thr - 1.0)

        if not crashed:
            if autopilot and model:
                action, _ = model.predict(obs, deterministic=True)
                ki_thr_pct = ((action[1] + 1) / 2) * 100
                manual_thr = ki_thr_pct
            else:
                thr_action = (manual_thr / 50.0) - 1.0
                action = np.array([pitch_cmd * sensitivity, thr_action], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                crashed = True
                print("Absturz!")

        # Render Loop
        env.render(flip=False)
        vx_kmh = env.vx * 3.6
        draw_overlay(env, autopilot, sensitivity, manual_thr, 0, vx_kmh)
        pygame.display.flip()

        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    run()