import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import os


class AirplaneAltitudeEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self, render_mode: str = "none"):
        super().__init__()
        self.render_mode = render_mode

        # --- DATEIPFAD ZUM BILD ---
        # HIER DEIN PFAD:
        self.image_path = r"/Users/dennis/PycharmProjects/RL Lander V2/Flugzeug.svg"

        # --- PHYSIK ---
        self.dt = 0.05  # 20 Hz
        self.g = 9.81
        self.rho = 1.225
        self.mass = 60000.0
        self.wing_area = 122.6
        self.max_thrust = 240000.0

        # Aerodynamik (Verbessert für Loopings)
        self.cd0 = 0.0267
        self.k = 0.0387
        self.cl_alpha = 5.5
        self.alpha_crit = np.deg2rad(15.0)

        # Ziele
        self.target_altitude = 300.0
        self.cruise_speed = 70.0
        self.max_steps = 1000000000

        # Action Space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Obs Space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # State
        self.altitude = 300.0
        self.vx = 70.0
        self.vz = 0.0
        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.throttle_actual = 0.5
        self.ground_offset_x = 0.0
        self.last_action = np.zeros(2)

        # Grafik
        self.screen = None
        self.clock = None
        self.font = None
        self.plane_img_original = None  # Zum Speichern des geladenen Bildes

        self.scenery = []
        for i in range(0, 1000, 50):
            obj_type = self.np_random.integers(0, 3)
            pos_x = i + self.np_random.integers(-10, 10)
            self.scenery.append((pos_x, obj_type))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Entscheidung: Normaler Start oder EXTREM-Szenario?
        # 30% Chance auf Chaos, damit er Loopings/Stalls abfangen lernt
        scenario = self.np_random.choice(["normal", "extreme"], p=[0.7, 0.3])

        if scenario == "normal":
            self.altitude = self.np_random.uniform(100.0, 500.0)
            self.vx = self.np_random.uniform(60.0, 80.0)
            self.vz = self.np_random.uniform(-2.0, 2.0)
            self.pitch = np.deg2rad(self.np_random.uniform(-5.0, 5.0))
        else:
            # EXTREM: Nase steil hoch/runter oder fast Strömungsabriss
            mode = self.np_random.choice(["nose_high", "nose_low", "low_speed"])

            if mode == "nose_high":
                self.altitude = self.np_random.uniform(200.0, 400.0)
                self.vx = self.np_random.uniform(60.0, 90.0)
                # Bis zu 45 Grad Nase hoch!
                self.pitch = np.deg2rad(self.np_random.uniform(20.0, 45.0))
                self.vz = 10.0  # Steigt bereits

            elif mode == "nose_low":
                self.altitude = self.np_random.uniform(400.0, 600.0)
                self.vx = self.np_random.uniform(80.0, 110.0)  # Schnell
                # Bis zu 30 Grad Sturzflug
                self.pitch = np.deg2rad(self.np_random.uniform(-30.0, -10.0))
                self.vz = -10.0

            elif mode == "low_speed":
                self.altitude = 400.0
                # Gefährlich langsam -> kurz vor Stall
                self.vx = self.np_random.uniform(50.0, 58.0)
                self.pitch = np.deg2rad(5.0)
                self.vz = -2.0

        self.pitch_rate = 0.0
        self.throttle_actual = 0.5
        self.last_action = np.zeros(2)
        self.ground_offset_x = 0.0

        return self._get_obs(np.zeros(2)), {}

    def _get_obs(self, action):
        return np.array([
            (self.altitude - self.target_altitude) / 100.0,
            self.vz / 20.0,
            self.pitch,
            self.throttle_actual,
            (self.vx - self.cruise_speed) / 50.0,
            action[0],
            action[1]
        ], dtype=np.float32)

    def step(self, action):
        stick_pitch = float(np.clip(action[0], -1.0, 1.0))
        stick_throttle = float(np.clip(action[1], -1.0, 1.0))
        target_thr = (stick_throttle + 1.0) / 2.0

        # Engine Lag
        if self.throttle_actual < target_thr:
            self.throttle_actual += 0.5 * self.dt
        else:
            self.throttle_actual -= 0.5 * self.dt
        self.throttle_actual = np.clip(self.throttle_actual, 0.0, 1.0)

        thrust = self.throttle_actual * self.max_thrust
        v = np.sqrt(self.vx ** 2 + self.vz ** 2)
        if v < 0.1: v = 0.1  # Schutz vor 0

        # Flugbahnwinkel
        gamma = np.arctan2(self.vz, self.vx)

        # --- VERBESSERTE PHYSIK (AoA Normalisierung) ---
        # Anstellwinkel (Alpha) ist die Differenz zwischen Nase und Flugbahn.
        # Wir nutzen atan2 von sin/cos, um den Winkel sauber zwischen -PI und PI zu halten.
        # Das verhindert seltsame Sprünge beim Überschlag.
        alpha = math.atan2(math.sin(self.pitch - gamma), math.cos(self.pitch - gamma))

        # Aero Koeffizienten
        # Wir nutzen eine sigmoid-artige Funktion für den Stall statt harter Kante,
        # damit die Physik nicht "springt".

        # Linearer Lift Bereich
        cl_linear = 0.2 + self.cl_alpha * alpha

        # Stall Dämpfung: Wenn Alpha > 15 Grad, sinkt der Lift drastisch
        stall_start = self.alpha_crit
        if abs(alpha) < stall_start:
            cl = cl_linear
            cd = self.cd0 + self.k * (cl ** 2)
            is_stalled = False
        else:
            # Post-Stall Physik (Vereinfacht: Lift weg, Drag hoch)
            # Wir behalten Vorzeichen bei, reduzieren aber den Wert
            sign = np.sign(alpha)
            cl = cl_linear * 0.3  # Auftrieb bricht ein

            # Drag explodiert bei 90 Grad AoA (Wand)
            # Bei 90 Grad trifft der Wind die volle Fläche von unten/oben
            drag_factor = 1.0 + 10.0 * (abs(alpha) - stall_start)
            cd = (self.cd0 + self.k * (cl_linear ** 2)) * drag_factor
            is_stalled = True

        q = 0.5 * self.rho * (v ** 2)
        lift = q * self.wing_area * cl
        drag = q * self.wing_area * cd

        # Kräfte berechnen
        fx = thrust * np.cos(self.pitch) - drag * np.cos(gamma) - lift * np.sin(gamma)
        fz = thrust * np.sin(self.pitch) - drag * np.sin(gamma) + lift * np.cos(gamma) - self.mass * self.g

        self.vx += (fx / self.mass) * self.dt
        self.vz += (fz / self.mass) * self.dt
        self.altitude += self.vz * self.dt

        # Pitch Rotation
        target_rate = stick_pitch * np.deg2rad(20.0)  # Etwas agiler (20 deg/s)
        self.pitch_rate += (target_rate - self.pitch_rate) * 5.0 * self.dt
        self.pitch += self.pitch_rate * self.dt

        # Normalisieren von Pitch zwischen -PI und PI (damit Winkel nicht unendlich wachsen)
        self.pitch = math.atan2(math.sin(self.pitch), math.cos(self.pitch))

        self.ground_offset_x += self.vx * self.dt
        self.ground_offset_x %= 1000.0

        # --- REWARD ---
        terminated = False
        reward = 0.0

        # Altitude
        dist = abs(self.altitude - self.target_altitude)
        reward += np.exp(-dist / 30.0) * 1.5

        # Speed (Ziel 70 m/s)
        speed_diff = abs(self.vx - self.cruise_speed)
        reward += np.exp(-speed_diff / 10.0) * 0.5

        # Angst vor Langsamflug (Stallgefahr)
        if self.vx < 60.0: reward -= 0.1

        # Smoothness
        action_diff = np.sum(np.abs(action - self.last_action))
        reward -= action_diff * 0.01
        self.last_action = action

        # Crash
        if self.altitude <= 0:
            terminated = True
            reward = -100.0

        # Kein harter Abbruch mehr bei Loopings, nur Physik und Strafe
        if is_stalled:
            reward -= 0.5  # Bestrafung für unsauberen Flug

        return self._get_obs(action), reward, terminated, False, {}

    def render(self, flip=True):
        if self.render_mode == "human":
            self._draw(flip)

    def _draw(self, do_flip):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 600))
            pygame.display.set_caption("Flight Sim RL")
            self.font = pygame.font.SysFont("Arial", 16)
            self.clock = pygame.time.Clock()

            # --- BILD LADEN ---
            try:
                # Versuche das Bild zu laden
                raw_img = pygame.image.load(self.image_path)
                # Skalieren auf eine vernünftige Größe (z.B. 80px breit)
                scale_ratio = 80.0 / raw_img.get_width()
                new_h = int(raw_img.get_height() * scale_ratio)
                self.plane_img_original = pygame.transform.smoothscale(raw_img, (80, new_h))
                print(f"Bild erfolgreich geladen: {self.image_path}")
            except Exception as e:
                print(f"FEHLER: Konnte Bild nicht laden: {e}")
                print("Benutze Standard-Polygon.")
                self.plane_img_original = None

        self.screen.fill((135, 206, 235))

        screen_h = 600
        plane_y_screen = screen_h / 2
        scale = 1.5
        ground_y_screen = plane_y_screen + (self.altitude * scale)

        pygame.draw.rect(self.screen, (34, 139, 34), (0, ground_y_screen, 1000, 2000))
        pygame.draw.line(self.screen, (0, 100, 0), (0, ground_y_screen), (1000, ground_y_screen), 2)

        for pos_meter, type_id in self.scenery:
            draw_x = (pos_meter - self.ground_offset_x) * scale
            if draw_x < -50: draw_x += 1000 * scale
            draw_x += 500
            draw_x %= (1000 * scale)

            rel_x = (pos_meter - self.ground_offset_x)
            if rel_x < -500: rel_x += 1000
            if rel_x > 500: rel_x -= 1000
            final_x = 500 + (rel_x * scale)
            final_y = ground_y_screen

            if -50 < final_x < 1050 and -50 < final_y < 700:
                if type_id == 0:
                    pygame.draw.rect(self.screen, (100, 50, 0), (final_x, final_y - 15, 6, 15))
                    pygame.draw.circle(self.screen, (0, 100, 0), (int(final_x + 3), int(final_y - 15)), 12)
                elif type_id == 1:
                    pygame.draw.rect(self.screen, (200, 200, 200), (final_x, final_y - 20, 30, 20))
                    pygame.draw.polygon(self.screen, (150, 0, 0),
                                        [(final_x, final_y - 20), (final_x + 15, final_y - 35),
                                         (final_x + 30, final_y - 20)])
                elif type_id == 2:
                    pygame.draw.circle(self.screen, (0, 150, 0), (int(final_x), int(final_y)), 8)

        target_y = plane_y_screen + (self.altitude - self.target_altitude) * scale
        pygame.draw.line(self.screen, (0, 255, 255), (0, target_y), (1000, target_y), 1)

        # --- FLUGZEUG ZEICHNEN ---
        # Rotation umrechnen: Pygame dreht gegen Uhrzeigersinn (positiv),
        # aber unsere Pitch-Logik ist oft andersrum definiert.
        # Wir testen: np.rad2deg(self.pitch). Ggf. muss hier ein Minus davor.
        deg = np.rad2deg(self.pitch)

        if self.plane_img_original:
            # Bild rotieren
            rotated_img = pygame.transform.rotate(self.plane_img_original, deg)
            rect = rotated_img.get_rect(center=(500, plane_y_screen))
            self.screen.blit(rotated_img, rect)
        else:
            # Fallback Polygon
            plane_surf = pygame.Surface((60, 20), pygame.SRCALPHA)
            pygame.draw.polygon(plane_surf, (220, 220, 220), [(0, 10), (60, 10), (10, 0)])
            pygame.draw.polygon(plane_surf, (255, 0, 0), [(0, 10), (15, 10), (5, 0)])
            pygame.draw.line(plane_surf, (50, 50, 50), (30, 10), (40, 20), 3)
            rotated = pygame.transform.rotate(plane_surf, deg)
            rect = rotated.get_rect(center=(500, plane_y_screen))
            self.screen.blit(rotated, rect)

        if do_flip:
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen: pygame.quit()