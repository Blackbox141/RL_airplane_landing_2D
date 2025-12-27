import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import pygame
import os
import warnings

warnings.filterwarnings("ignore")


class AirplaneAltitudeEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(
            self,
            render_mode="none",
            enable_scenarios=True,
            max_episode_steps=3000,
    ):
        super(AirplaneAltitudeEnv, self).__init__()

        self.render_mode = render_mode
        self.enable_scenarios = enable_scenarios
        self.max_episode_steps = int(max_episode_steps)

        self.screen = None
        self.clock = None
        self.width = 1000
        self.height = 600

        # --- OPTIK ---
        self.image_path = r"/Users/dennis/PycharmProjects/RL Lander V2/Flugzeug.svg"
        self.plane_img_original = None
        self.font = None

        self.scenery = []
        rng = np.random.default_rng()
        for i in range(0, 5000, 50):
            obj_type = rng.integers(0, 3)
            pos_x = i + rng.integers(-10, 10)
            self.scenery.append((pos_x, obj_type))
        self.ground_offset_x = 0.0

        self.clouds = []
        for i in range(0, 12000, 300):
            cx = i + rng.integers(-80, 80)
            cy = rng.integers(20, 220)
            size = rng.integers(30, 90)
            self.clouds.append((float(cx), int(cy), int(size)))

        # Variable für sanften Kamera-Zoom der Umgebung
        self.render_view_height = 500.0

        # --- PHYSIK & STEUERUNG ---
        self.dt = 0.05
        self.gravity = 9.81
        self.rho = 1.225
        self.mass = 60000.0
        self.wing_area = 122.6
        self.max_thrust = 240000.0

        self.cd0 = 0.0267
        self.k = 0.0387
        self.cl_alpha = 5.5

        self.alpha_crit = np.deg2rad(15.0)
        self.alpha_stall_on = np.deg2rad(18.0)
        self.alpha_stall_off = np.deg2rad(14.0)

        self.min_speed = 40.0
        self.max_speed = 350.0
        self.safe_speed = 80.0
        self.safe_speed_hard = 55.0

        self.low_speed_counter = 0
        self.low_speed_counter_limit = 400

        self.max_structural_speed = 333.0
        self.min_altitude = 0.0
        self.max_altitude = 80000.0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        low_obs = np.array([-np.pi, 0, 0, -300, -5000], dtype=np.float32)
        high_obs = np.array([np.pi, self.max_speed, self.max_altitude, 300, 5000], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.pitch = 0.0
        self.speed = 200.0
        self.altitude = 1000.0
        self.vertical_speed = 0.0
        self.target_altitude = 1000.0
        self.current_scenario = "normal"
        self.step_counter = 0

        self.vx = 200.0
        self.vz = 0.0
        self.pitch_rate = 0.0
        self.throttle_actual = 0.5
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.is_stalled_state = False
        self.last_pitch_input = 0.0
        self.last_throttle = 0.5
        self.on_target_streak = 0
        self.prev_abs_error = None

    def _get_obs(self):
        return np.array(
            [
                self.pitch, #Ausrichtung der "Nase" des Flugzeugs
                self.speed, #Geschwindigkeit, welche den Auftrieb beeinflusst
                self.altitude, #Höhe über dem Boden
                self.vertical_speed, #Vertikalgeschwindigkeit - zeigt, ob das Flugzeug steigt oder sinkt
                self.target_altitude - self.altitude, #Differenz zwischen Zielhöhe und aktueller Höhe
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.step_counter += 1
        self.prev_action = self.last_action.copy()

        stick_pitch = float(np.clip(action[0], -1.0, 1.0))
        stick_throttle = float(np.clip(action[1], -1.0, 1.0))
        self.last_pitch_input = stick_pitch
        self.last_action = np.array([stick_pitch, stick_throttle], dtype=np.float32)

        # --- PHYSIK ---
        target_thr = (stick_throttle + 1.0) / 2.0
        self.throttle_actual += (target_thr - self.throttle_actual) * 0.4 * self.dt
        self.throttle_actual = np.clip(self.throttle_actual, 0.0, 1.0)
        self.last_throttle = self.throttle_actual

        thrust = self.throttle_actual * self.max_thrust
        v = np.sqrt(self.vx ** 2 + self.vz ** 2)
        if v < 0.1:
            v = 0.1

        gamma = np.arctan2(self.vz, self.vx)
        alpha = math.atan2(math.sin(self.pitch - gamma), math.cos(self.pitch - gamma))
        cl_linear = 0.2 + self.cl_alpha * alpha

        # Stall-Logik
        if (not self.is_stalled_state) and (abs(alpha) >= self.alpha_stall_on):
            self.is_stalled_state = True
        elif self.is_stalled_state and (abs(alpha) <= self.alpha_stall_off):
            self.is_stalled_state = False

        if not self.is_stalled_state:
            cl = cl_linear
            cd = self.cd0 + self.k * (cl ** 2)
            q = 0.5 * self.rho * (v ** 2)
            lift = q * self.wing_area * cl
            drag = q * self.wing_area * cd
            fx = thrust * np.cos(self.pitch) - drag * np.cos(gamma) - lift * np.sin(gamma)
            fz = thrust * np.sin(self.pitch) - drag * np.sin(gamma) + lift * np.cos(gamma) - self.mass * self.gravity
            target_rate = stick_pitch * np.deg2rad(22.0)
            self.pitch_rate += (target_rate - self.pitch_rate) * 2.6 * self.dt
        else:
            lift = 0.0
            cd_stall = 1.2
            q = 0.5 * self.rho * (v ** 2)
            drag = q * self.wing_area * cd_stall
            fx = thrust * np.cos(self.pitch) - drag * np.cos(gamma)
            fz = thrust * np.sin(self.pitch) - drag * np.sin(gamma) - self.mass * self.gravity
            target_rate = stick_pitch * np.deg2rad(5.0)
            self.pitch_rate += (target_rate - self.pitch_rate) * 4.0 * self.dt

        self.vx += (fx / self.mass) * self.dt
        self.vz += (fz / self.mass) * self.dt
        self.altitude += self.vz * self.dt

        self.pitch += self.pitch_rate * self.dt
        self.pitch = math.atan2(math.sin(self.pitch), math.cos(self.pitch))

        self.speed = np.sqrt(self.vx ** 2 + self.vz ** 2)
        self.vertical_speed = self.vz
        self.ground_offset_x += self.vx * self.dt

        # --- TERMINATION ---
        terminated = False
        truncated = False
        crashed = False
        term_reason = "none"

        if self.altitude <= 0:
            self.altitude = 0
            terminated = True
            crashed = True
            term_reason = "CRASH_GROUND"

        if self.altitude >= self.max_altitude:
            terminated = True
            term_reason = "LIMIT_ALTITUDE"

        if self.speed > self.max_structural_speed:
            terminated = True
            crashed = True
            term_reason = "CRASH_OVERSPEED"

        if self.step_counter >= self.max_episode_steps:
            truncated = True
            term_reason = "TIMEOUT_STEPS"

        if self.speed < self.safe_speed_hard or self.is_stalled_state:
            self.low_speed_counter += 1
        else:
            self.low_speed_counter = 0

        # --- REWARD ---
        reward = 0.0
        alt_error = (self.target_altitude - self.altitude)
        abs_error = abs(alt_error)
        vz = self.vertical_speed

        approach = float(np.sign(alt_error) * vz)
        if abs_error > 80.0:
            reward += 0.25 * float(np.clip(approach, -35.0, 35.0))

        reward -= 0.18

        if self.prev_abs_error is not None:
            progress = (self.prev_abs_error - abs_error)
            reward += 3.5 * (progress / 100.0)

        reward -= (abs_error / 80.0)

        scale_r = float(np.clip(abs_error / 400.0, 0.0, 1.0))
        vz_coef = (0.002 * scale_r) + (0.012 * (1.0 - scale_r))
        reward -= vz_coef * (vz ** 2)

        tau = 1.5
        pred_alt = self.altitude + vz * tau
        pred_error = self.target_altitude - pred_alt
        lookahead_w = (1.0 * scale_r) + (3.0 * (1.0 - scale_r))
        reward -= lookahead_w * (abs(pred_error) / 200.0)

        da = self.last_action - self.prev_action
        reward -= 0.06 * float(np.dot(da, da))

        v_ref = 110.0
        reward -= abs(self.speed - v_ref) / 250.0

        if self.speed < self.safe_speed:
            dv = (self.safe_speed - self.speed)
            reward -= 0.04 * (dv ** 2)

        if self.is_stalled_state:
            reward -= 12.0

        in_zone = abs_error < 15.0
        is_stable = (abs(vz) < 3.0) and (abs(self.pitch_rate) < np.deg2rad(3.0))

        if in_zone and is_stable:
            self.on_target_streak += 1
            streak_bonus = 1.0 + (self.on_target_streak * 0.08)
            streak_bonus = min(streak_bonus, 6.0)
            reward += streak_bonus
        else:
            self.on_target_streak = 0

        if crashed:
            reward = -1000.0

        self.prev_abs_error = abs_error

        obs = self._get_obs()
        info = {
            "height_error": abs_error,
            "scenario": self.current_scenario,
            "streak": self.on_target_streak,
            "vz": float(vz),
            "truncated": truncated,
            "speed": float(self.speed),
            "stall": bool(self.is_stalled_state),
            "termination_reason": term_reason
        }
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.is_stalled_state = False
        self.throttle_actual = 0.5
        self.pitch_rate = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_pitch_input = 0.0
        self.last_throttle = self.throttle_actual
        self.vz = 0.0
        self.vertical_speed = 0.0
        self.on_target_streak = 0
        self.low_speed_counter = 0

        if self.enable_scenarios and random.random() < 0.8:
            modes = ["height_change", "on_target", "stall_recovery_too_high", "stall_recovery_too_low"]
            scenario = random.choices(modes, weights=[0.5, 0.3, 0.1, 0.1])[0]
            self.current_scenario = scenario

            if scenario == "height_change":
                self.target_altitude = np.random.uniform(1000, 4000)
                offset = 500 if random.random() < 0.5 else -500
                self.altitude = self.target_altitude + offset
                self.pitch = np.random.uniform(-0.1, 0.1)
                self.speed = 100.0
                self.vx = self.speed * math.cos(self.pitch)
                self.vz = self.speed * math.sin(self.pitch)

            elif scenario == "on_target":
                self.target_altitude = np.random.uniform(1000, 4000)
                self.altitude = self.target_altitude
                self.speed = 90.0
                self.pitch = 0.0
                self.vx = self.speed
                self.vz = 0.0

            elif scenario == "stall_recovery_too_high":
                self.target_altitude = 5000.0
                self.altitude = 8500.0
                self.speed = 35.0
                self.pitch = 0.30
                self.vx = self.speed * math.cos(self.pitch)
                self.vz = self.speed * math.sin(self.pitch)
                self.is_stalled_state = True
                self.throttle_actual = 0.65
                self.last_throttle = self.throttle_actual

            elif scenario == "stall_recovery_too_low":
                self.target_altitude = 20000.0
                self.altitude = 8500.0
                self.speed = 35.0
                self.pitch = 0.30
                self.vx = self.speed * math.cos(self.pitch)
                self.vz = self.speed * math.sin(self.pitch)
                self.is_stalled_state = True
                self.throttle_actual = 0.65
                self.last_throttle = self.throttle_actual
        else:
            self.current_scenario = "normal"
            self.target_altitude = np.random.uniform(1000, 4000)
            self.altitude = np.random.uniform(1000, 4000)
            self.speed = 100.0
            self.pitch = 0.0
            self.vx = 100.0
            self.vz = 0.0

        self.vertical_speed = self.vz
        self.prev_abs_error = abs(self.target_altitude - self.altitude)

        # Kamera Reset
        self.render_view_height = max(self.altitude * 1.5, 200.0)

        obs = self._get_obs()
        return obs, {"scenario": self.current_scenario}

    def render(self, flip=True):
        if self.render_mode == "human":
            self._draw(flip)

    def _draw(self, do_flip):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.SysFont("Arial", 16)
            self.clock = pygame.time.Clock()
            try:
                if os.path.exists(self.image_path):
                    raw_img = pygame.image.load(self.image_path)
                    scale_ratio = 80.0 / raw_img.get_width()
                    new_h = int(raw_img.get_height() * scale_ratio)
                    self.plane_img_original = pygame.transform.smoothscale(raw_img, (80, new_h))
            except:
                pass

        # Himmel
        top = np.array([110, 190, 255], dtype=np.int32)
        bot = np.array([180, 225, 255], dtype=np.int32)
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            col = (top * (1.0 - t) + bot * t).astype(np.int32)
            pygame.draw.line(self.screen, col.tolist(), (0, y), (self.width, y))

        # --- KAMERA LOGIK ---
        # 1. Glättung für View-Height (bestimmt Zoom für Boden/Hintergrund)
        target_view_height = max(self.altitude * 1.8, 200.0)
        self.render_view_height += (target_view_height - self.render_view_height) * 0.05

        screen_h_avail = self.height - 60
        pixels_per_meter = screen_h_avail / self.render_view_height

        ground_screen_y = self.height - 40
        # Flugzeug Y-Position relativ zum Boden (wichtig für Boden-Abstandswahrnehmung)
        plane_screen_y = int(ground_screen_y - (self.altitude * pixels_per_meter))

        # Scale für Deko-Objekte
        base_ppm = screen_h_avail / 200.0
        visual_scale = pixels_per_meter / base_ppm
        visual_scale = float(np.clip(visual_scale, 0.0001, 2.0))

        # --- WOLKEN ---
        for cx, cy, size in self.clouds:
            dx = (cx - self.ground_offset_x * 0.3) * 0.18
            dx %= (12000.0 * 0.18)
            x = int(dx - ((12000.0 * 0.18) / 2) + self.width / 2)
            if -200 < x < self.width + 200:
                s = int(size * (0.5 + 0.5 * visual_scale))
                pygame.draw.ellipse(self.screen, (255, 255, 255), (x, cy, s, int(s * 0.55)))

        # --- BODEN ---
        pygame.draw.rect(self.screen, (34, 139, 34), (0, ground_screen_y, self.width, self.height - ground_screen_y))
        pygame.draw.line(self.screen, (0, 100, 0), (0, ground_screen_y), (self.width, ground_screen_y), 3)

        for pos, type_id in self.scenery:
            scroll_width = 5000.0
            rel_x_m = (pos - self.ground_offset_x) % scroll_width
            if rel_x_m > scroll_width / 2: rel_x_m -= scroll_width

            x_ppm = max(pixels_per_meter, 0.05)
            obj_screen_x = int(self.width / 2 + rel_x_m * x_ppm * 0.8)

            if -100 < obj_screen_x < self.width + 100:
                base_y = ground_screen_y

                def get_dim(val_m):
                    px = int(val_m * 10.0 * visual_scale)
                    return max(2, px)

                if type_id == 0:
                    r = get_dim(1.4)
                    pygame.draw.circle(self.screen, (0, 110, 0), (obj_screen_x, base_y - r), r)
                    stem_h = max(2, int(r * 0.6))
                    pygame.draw.rect(self.screen, (90, 60, 30),
                                     (obj_screen_x - max(1, r // 4), base_y - stem_h, max(2, r // 2), stem_h))
                elif type_id == 1:
                    w = get_dim(3.0)
                    h = get_dim(2.2)
                    pygame.draw.rect(self.screen, (210, 210, 210), (obj_screen_x - w // 2, base_y - h, w, h))
                else:
                    w = get_dim(5.0)
                    h = get_dim(4.0)
                    pygame.draw.rect(self.screen, (170, 170, 185), (obj_screen_x - w // 2, base_y - h, w, h))

        # --- TARGET LINE ---
        tgt_screen_y = int(ground_screen_y - (self.target_altitude * pixels_per_meter))
        in_zone = self.on_target_streak > 0
        tgt_col = (255, 215, 0) if in_zone else (0, 255, 255)
        line_width = 4 if in_zone else 2

        if 0 < tgt_screen_y < self.height:
            pygame.draw.line(self.screen, tgt_col, (0, tgt_screen_y), (self.width, tgt_screen_y), line_width)
            if in_zone:
                self.screen.blit(self.font.render(f"STREAK: {self.on_target_streak}", True, (255, 215, 0)),
                                 (self.width / 2 + 20, tgt_screen_y - 20))
        else:
            txt = f"Target {'^' if tgt_screen_y < 0 else 'v'} {int(self.target_altitude)}m"
            py = 20 if tgt_screen_y < 0 else self.height - 30
            self.screen.blit(self.font.render(txt, True, tgt_col), (self.width / 2 - 40, py))

        # --- FLUGZEUG ---
        # 1. Größe NUR von Höhe abhängig (nicht Geschwindigkeit, nicht Kamera-Smoothness)
        # Kurve: 1.0 (Boden) -> 0.0 (Unendlich).
        # Halbwertshöhe z.B. 400m. D.h. bei 400m ist das Flugzeug halb so groß wie am Boden.
        current_alt = max(0.0, self.altitude)
        size_factor = 1.0 / (1.0 + current_alt / 400.0)

        # Max Größe am Boden: 95px, Min Größe in Stratosphäre: 25px
        max_px = 95
        min_px = 25
        plane_w = int(min_px + (max_px - min_px) * size_factor)

        deg = np.rad2deg(self.pitch)
        if self.plane_img_original:
            aspect = self.plane_img_original.get_height() / self.plane_img_original.get_width()
            h = int(plane_w * aspect)
            scaled = pygame.transform.smoothscale(self.plane_img_original, (plane_w, h))
            rot = pygame.transform.rotate(scaled, deg)
        else:
            surf = pygame.Surface((plane_w, int(plane_w * 0.25)), pygame.SRCALPHA)
            pygame.draw.ellipse(surf, (200, 200, 200), (0, 0, plane_w, int(plane_w * 0.25)))
            rot = pygame.transform.rotate(surf, deg)

        self.screen.blit(rot, rot.get_rect(center=(self.width // 2, plane_screen_y)))

        if self.is_stalled_state:
            warn = self.font.render("!! STALL !!", True, (255, 0, 0))
            pygame.draw.rect(self.screen, (0, 0, 0), (self.width / 2 - 40, self.height / 2 - 60, 100, 30))
            self.screen.blit(warn, (self.width / 2 - 30, self.height / 2 - 55))

        if do_flip:
            pygame.display.flip()
            self.clock.tick(60)

    def close(self):
        if self.screen:
            pygame.quit()