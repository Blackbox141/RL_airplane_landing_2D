import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

from airplane_alt_env import AirplaneAltitudeEnv

BASE_PATH = "altitude_agent"
SELECTED_RUN = "run_50"

RUN_DIR = os.path.join(BASE_PATH, SELECTED_RUN)
MODEL_PATH = os.path.join(RUN_DIR, "best_model.zip")
VECNORM_PATH = os.path.join(RUN_DIR, "vecnorm.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")


def draw_overlay(screen, env, autopilot, manual_thr, scenario, pitch_cmd, thr_cmd, stall_blink_counter):
    font = pygame.font.SysFont("Consolas", 16, bold=True)
    width, height = screen.get_size()

    # LINKES PANEL: Controls
    left_w, left_h = 430, 150
    left_x, left_y = 10, 10
    left_surface = pygame.Surface((left_w, left_h), pygame.SRCALPHA)
    left_surface.fill((0, 0, 0, 190))
    screen.blit(left_surface, (left_x, left_y))

    controls_lines = [
        "Controls:",
        "  W / S  : Pitch hoch / runter",
        "  D / A  : Throttle + / -",
        "  P      : Autopilot an/aus",
        "  R      : Episode zurücksetzen",
        "  ESC    : Simulation beenden",
    ]

    for i, text in enumerate(controls_lines):
        color = (220, 220, 220) if i == 0 else (200, 200, 200)
        img = font.render(text, True, color)
        screen.blit(img, (left_x + 10, left_y + 10 + i * 20))

    # STALL-WARNUNG unterhalb der Controls, blinkend
    if getattr(env, "is_stalled_state", False):

        if (stall_blink_counter // 18) % 2 == 0:
            stall_text = "!! STALL !!"
            stall_img = font.render(stall_text, True, (255, 0, 0))
            stall_rect = stall_img.get_rect()
            stall_rect.topleft = (left_x + 20, left_y + left_h + 10)

            box_rect = stall_rect.inflate(20, 10)
            stall_surface = pygame.Surface(box_rect.size, pygame.SRCALPHA)
            stall_surface.fill((0, 0, 0, 220))
            pygame.draw.rect(stall_surface, (255, 0, 0), stall_surface.get_rect(), 2)

            screen.blit(stall_surface, box_rect.topleft)
            screen.blit(stall_img, stall_rect.topleft)

    # --- RECHTES PANEL: Flugparameter & Inputs ---
    right_w, right_h = 430, 220
    right_x, right_y = width - right_w - 10, 10
    right_surface = pygame.Surface((right_w, right_h), pygame.SRCALPHA)
    right_surface.fill((0, 0, 0, 190))
    screen.blit(right_surface, (right_x, right_y))

    stats = [
        (f"RUN: {SELECTED_RUN}", (200, 200, 200)),
        (f"MODE: {'AUTOPILOT' if autopilot else 'MANUAL'}",
         (0, 255, 0) if autopilot else (255, 255, 255)),
        (f"Scenario: {scenario}", (0, 200, 255)),
        (f"Altitude: {env.altitude:.0f}m / Tgt: {env.target_altitude:.0f}m", (255, 255, 0)),
        (f"Speed: {env.speed * 3.6:.0f} km/h", (255, 255, 0)),
        (f"Pitch: {np.rad2deg(env.pitch):.1f}°", (255, 255, 0)),
        (f"Throttle: {manual_thr:.0f}%", (255, 255, 0)),
        ("", (0, 0, 0)),
        (f"Input Pitch cmd: {pitch_cmd:+.2f}", (255, 165, 0)),
        (f"Input Throttle cmd: {thr_cmd:+.2f}", (255, 165, 0)),
    ]

    for i, (text, color) in enumerate(stats):
        if text == "":
            continue
        img = font.render(text, True, color)
        screen.blit(img, (right_x + 10, right_y + 10 + i * 20))


def run():
    model = PPO.load(MODEL_PATH)

    def make_env():
        return AirplaneAltitudeEnv(render_mode="human")

    venv = DummyVecEnv([make_env])

    if os.path.exists(VECNORM_PATH):
        venv = VecNormalize.load(VECNORM_PATH, venv)
        venv.training = False
        venv.norm_reward = False

    base_env = venv.venv.envs[0]
    pygame.init()
    clock = pygame.time.Clock()

    # Kurze Einleitung zur Steuerung auf der Konsole
    print("=== Flight Simulation gestartet ===")
    print(f"Geladener Run: {SELECTED_RUN}")
    print("Steuerung:")
    print("  W / S       : Pitch hoch / runter")
    print("  D / A       : Throttle erhöhen / verringern")
    print("  P           : Autopilot an/aus")
    print("  R           : Episode zurücksetzen")
    print("  ESC / Fenster schliessen: Simulation beenden")
    print("Das aktuelle Szenario und Stall-Zustände werden im HUD angezeigt.\n")

    obs = venv.reset()
    autopilot = False
    manual_thr = 50.0
    stall_blink_counter = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    # P wie "Autopilot"
                    autopilot = not autopilot
                if event.key == pygame.K_r:
                    obs = venv.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()

        # Pitch-Steuerung: W / S
        pitch_in = 0.0
        if keys[pygame.K_w]:
            pitch_in = 1.0
        if keys[pygame.K_s]:
            pitch_in = -1.0

        # Throttle-Steuerung: D (mehr Schub) / A (weniger Schub)
        if keys[pygame.K_d]:
            manual_thr = min(100, manual_thr + 1)
        if keys[pygame.K_a]:
            manual_thr = max(0, manual_thr - 1)

        if autopilot:
            # Autopilot: Action kommt vom Modell
            action, _ = model.predict(obs, deterministic=True)
            # Throttle-Prozent für Anzeige aus dem Modell ableiten
            manual_thr = ((action[0][1] + 1) / 2) * 100
        else:
            # Manuelle Steuerung: Pitch und Throttle-Normierung für das Environment
            thr_norm = (manual_thr / 50.0) - 1.0
            action = np.array([[pitch_in, thr_norm]], dtype=np.float32)

        # Aktueller Steuer-Input (Action) für Overlay-Anzeige
        current_pitch_cmd = float(action[0][0])
        current_thr_cmd = float(action[0][1])

        obs, _, dones, _ = venv.step(action)
        if dones[0]: obs = venv.reset()

        base_env.render(flip=False)
        draw_overlay(
            base_env.screen,
            base_env,
            autopilot,
            manual_thr,
            getattr(base_env, "current_scenario", "N/A"),
            current_pitch_cmd,
            current_thr_cmd,
            stall_blink_counter
        )
        pygame.display.flip()
        stall_blink_counter += 1
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    run()