import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# Das korrekte Env importieren
from airplane_alt_env import AirplaneAltitudeEnv

# --- DEINE STRUKTUR LAUT SCREENSHOT ---
BASE_PATH = "altitude_agent"
SELECTED_RUN = "run_5"  # Diesen Wert ändern, um andere Runs zu laden

# Pfade exakt nach deinem Screenshot:
RUN_DIR = os.path.join(BASE_PATH, SELECTED_RUN)
MODEL_PATH = os.path.join(RUN_DIR, "best_model.zip")
VECNORM_PATH = os.path.join(RUN_DIR, "vecnorm.pkl")

# --- PRÜFUNG ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")


class ScenarioMenu:
    def __init__(self):
        self.items = ["NORMAL", "stall_recovery"]
        self.selected = None
        self.visible = True
        self.font = None
        self.big_font = None
        self.btn_rects = []

    def _ensure_fonts(self):
        if not pygame.font.get_init(): pygame.font.init()
        if self.font is None: self.font = pygame.font.SysFont("Consolas", 18, bold=True)
        if self.big_font is None: self.big_font = pygame.font.SysFont("Consolas", 22, bold=True)

    def draw(self, screen):
        if not self.visible: return
        self._ensure_fonts()
        w, h = screen.get_size()
        panel_w, panel_h = 400, 200
        px, py = (w - panel_w) // 2, (h - panel_h) // 2

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        pygame.draw.rect(screen, (30, 30, 30), (px, py, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(screen, (255, 255, 255), (px, py, panel_w, panel_h), 2, border_radius=10)

        title = self.big_font.render(f"Geladener Run: {SELECTED_RUN}", True, (255, 255, 0))
        screen.blit(title, (px + 20, py + 20))

        self.btn_rects = []
        for i, item in enumerate(self.items):
            rect = pygame.Rect(px + 50, py + 70 + i * 60, panel_w - 100, 40)
            pygame.draw.rect(screen, (60, 60, 60), rect, border_radius=5)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1, border_radius=5)
            txt = self.font.render(item, True, (255, 255, 255))
            screen.blit(txt, txt.get_rect(center=rect.center))
            self.btn_rects.append((item, rect))

    def handle_event(self, event):
        if not self.visible: return None
        if event.type == pygame.MOUSEBUTTONDOWN:
            for label, rect in self.btn_rects:
                if rect.collidepoint(event.pos):
                    self.visible = False
                    return label
        return None


def draw_overlay(screen, env, autopilot, manual_thr, scenario):
    font = pygame.font.SysFont("Consolas", 16, bold=True)
    stats = [
        f"RUN: {SELECTED_RUN}",
        f"MODE: {'AUTOPILOT' if autopilot else 'MANUAL'}",
        f"Scenario: {scenario}",
        f"Altitude: {env.altitude:.0f}m / Tgt: {env.target_altitude:.0f}m",
        f"Speed: {env.speed * 3.6:.0f} km/h",
        f"Pitch: {np.rad2deg(env.pitch):.1f}°",
        f"Throttle: {manual_thr:.0f}%",
        f"[A] Autopilot ON/OFF | [M] Menu"
    ]
    for i, s in enumerate(stats):
        img = font.render(s, True, (0, 255, 0) if autopilot else (255, 255, 255))
        screen.blit(img, (20, 20 + i * 20))


def run():
    model = PPO.load(MODEL_PATH)

    def make_env():
        return AirplaneAltitudeEnv(render_mode="human")

    venv = DummyVecEnv([make_env])

    # Normalisierungs-Datei laden
    if os.path.exists(VECNORM_PATH):
        venv = VecNormalize.load(VECNORM_PATH, venv)
        venv.training = False
        venv.norm_reward = False

    base_env = venv.venv.envs[0]
    pygame.init()
    clock = pygame.time.Clock()
    menu = ScenarioMenu()

    obs = venv.reset()
    selected_scenario = "NORMAL"
    autopilot = False
    manual_thr = 50.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m: menu.visible = True

            picked = menu.handle_event(event)
            if picked:
                selected_scenario = picked
                obs = venv.reset()
                autopilot = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a: autopilot = not autopilot
                if event.key == pygame.K_r: obs = venv.reset()

        if menu.visible:
            base_env.render(flip=False)
            menu.draw(base_env.screen)
            pygame.display.flip()
            continue

        keys = pygame.key.get_pressed()
        pitch_in = 0.0
        if keys[pygame.K_UP]: pitch_in = 1.0
        if keys[pygame.K_DOWN]: pitch_in = -1.0

        if keys[pygame.K_LSHIFT]: manual_thr = min(100, manual_thr + 1)
        if keys[pygame.K_LCTRL]: manual_thr = max(0, manual_thr - 1)

        if autopilot:
            action, _ = model.predict(obs, deterministic=True)
            manual_thr = ((action[0][1] + 1) / 2) * 100
        else:
            thr_norm = (manual_thr / 50.0) - 1.0
            action = np.array([[pitch_in, thr_norm]], dtype=np.float32)

        obs, _, dones, _ = venv.step(action)
        if dones[0]: obs = venv.reset()

        base_env.render(flip=False)
        draw_overlay(base_env.screen, base_env, autopilot, manual_thr, selected_scenario)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    run()