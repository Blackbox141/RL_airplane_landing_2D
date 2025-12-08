#airplane_landing_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import pygame


class AirplaneLandingEnv(gym.Env):
    """
    2D airplane landing environment, loosely based on a Boeing 737 on final.

    - World:
        * Runway is a straight line on the ground (y = 0).
        * Coordinate x is along the runway centerline.
        * The simulation starts ~3 NM before the intended touchdown point.

    - Touchdown:
        * Touchdown target is a single point on the runway (landing_spot_x).
        * Reward depends on:
            - distance from this point at touchdown,
            - vertical impact speed,
            - pitch attitude.

    - State (observation):
        [x, altitude, vx, vz, pitch, throttle]

        x        : horizontal position along the runway [m]
        altitude : height above the runway [m] (y)
        vx       : horizontal speed [m/s]
        vz       : vertical speed [m/s] (positive up)
        pitch    : airplane nose angle [rad], 0 = level flight, positive = nose up
        throttle : normalized engine throttle in [0, 1]

    - Action (continuous):
        [pitch_rate_cmd, throttle_rate_cmd] in [-1, 1]

        pitch_rate_cmd   : command to change pitch (nose up/down)
        throttle_rate_cmd: command to change throttle (more/less power)

    - Rough 737-like behaviour (highly simplified):
        * Typical final approach speed ~72 m/s (~140 kt).
        * Lift ~ k * vx^2 (so too slow => sink, too fast => climbs).
        * Stall if vx < stall_speed: extra sink rate and penalty.
        * Very hard touchdown (impact vertical speed too large) => "explosion" (crash).

    The goal is to touch down smoothly at the landing spot with:
        - small vertical speed,
        - small pitch angle,
        - touchdown close to the target point,
        - within runway bounds.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, render_mode: str = "none"):
        super().__init__()
        self.render_mode = render_mode

        # Basic units / distances
        self.nautical_mile_m = 1852.0
        self.approach_distance_nm = 3.0  # start ~3 NM before touchdown
        self.approach_distance_m = self.approach_distance_nm * self.nautical_mile_m

        # World / runway parameters (meters)
        self.runway_start_x = 0.0
        self.runway_length = 3000.0  # typical-ish runway length
        self.braking_zone_end = self.runway_start_x + self.runway_length

        # Single point landing spot (touchdown point) on the runway
        self.landing_spot_x = 1000.0  # target touchdown point
        self.landing_zone_half_width = 5.0  # for drawing only
        self.landing_zone_start = self.landing_spot_x - self.landing_zone_half_width
        self.landing_zone_end = self.landing_spot_x + self.landing_zone_half_width

        # Physics parameters
        self.dt = 0.1
        self.g = 9.81

        # Thrust model
        self.max_thrust = 40.0              # arbitrary force scale
        self.mass_factor = 0.05             # scales forces to acceleration

        # Pitch limits
        self.max_pitch = np.deg2rad(20.0)       # max pitch up/down
        self.max_pitch_rate = np.deg2rad(20.0)  # rad/s

        # Throttle limits
        self.max_throttle_change = 1.0      # per second

        # Rough "737-like" approach parameters
        self.target_approach_speed = 72.0   # m/s (~140 kt)
        self.stall_speed = 60.0             # m/s (~117 kt), rough, for behaviour only
        self.glide_slope_deg = 3.0
        self.target_descent_rate = -3.5     # m/s (~700 fpm)

        # Lift model (very simplified): lift_accel ~ lift_coefficient * vx^2 * cos(pitch)
        self.lift_coefficient = 0.02

        # Crash / hard-landing thresholds
        self.soft_landing_vspeed = 2.0      # m/s
        self.hard_landing_vspeed = 4.0      # m/s → above this is "crash"
        self.max_steps = 800

        # Observation space: [x, altitude, vx, vz, pitch, throttle]
        high = np.array([
            self.braking_zone_end + 2000.0,   # x
            800.0,                            # altitude
            150.0,                            # vx
            50.0,                             # vz
            np.pi / 2,                        # pitch
            1.0                               # throttle
        ], dtype=np.float32)

        low = np.array([
            -6000.0,                          # x (far before runway)
            0.0,                              # altitude
            0.0,                              # vx
            -50.0,                            # vz
            -np.pi / 2,                       # pitch
            0.0                               # throttle
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: [pitch_rate_cmd, throttle_rate_cmd] in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Internal state
        self.state: np.ndarray | None = None
        self.step_count = 0

        # Pygame / rendering
        self.screen = None
        self.clock = None
        self.window_width = 1200
        self.window_height = 600
        self.ground_y = 500  # pixel position of the ground line
        self.pixels_per_meter_y = 1.5

        # Airplane sprite (loaded lazily when rendering)
        self.plane_image = None
        self.plane_image_scale = 0.15  # scale factor for the loaded PNG

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start etwas weiter vor der Landebahn, damit das Flugzeug mehr „Anflugzeit“ hat
        # und nicht direkt in den Boden fällt. Wir beginnen ca. 1500 m vor dem Touchdown-Punkt.
        glideslope_rad = np.deg2rad(self.glide_slope_deg)

        start_distance_before_touchdown = 1500.0  # [m] vor dem Landepunkt
        base_x = self.landing_spot_x - start_distance_before_touchdown

        # Kleine Zufälligkeit in der Position entlang der Bahn (±200 m)
        x = base_x + self.np_random.uniform(-200.0, 200.0)

        horizontal_distance_to_touchdown = max(self.landing_spot_x - x, 300.0)
        altitude_nominal = np.tan(glideslope_rad) * horizontal_distance_to_touchdown

        # Add some noise to altitude, aber etwas höhere Mindesthöhe, damit er nicht gleich stallt
        altitude = altitude_nominal + self.np_random.uniform(-20.0, 20.0)
        altitude = float(np.clip(altitude, 80.0, 600.0))

        # Airspeed around typical 737 final approach speed
        vx = self.np_random.uniform(
            self.target_approach_speed - 5.0,
            self.target_approach_speed + 5.0,
        )

        # Vertical speed around target descent rate
        vz = self.np_random.uniform(
            self.target_descent_rate - 0.5,
            self.target_descent_rate + 0.5,
        )

        # Pitch near level / small nose-up
        pitch = self.np_random.uniform(
            -np.deg2rad(1.0),
            np.deg2rad(5.0),
        )

        throttle = 0.7

        self.state = np.array([x, altitude, vx, vz, pitch, throttle], dtype=np.float32)
        self.step_count = 0

        if self.render_mode == "human":
            self._render_frame()

        return self.state, {}

    def step(self, action):
        x, altitude, vx, vz, pitch, throttle = self.state
        self.step_count += 1

        info = {}

        # Ensure correct shape and type
        action = np.asarray(action, dtype=np.float32)
        pitch_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Apply commanded pitch / throttle rates
        pitch += pitch_cmd * self.max_pitch_rate * self.dt
        pitch = float(np.clip(pitch, -self.max_pitch, self.max_pitch))

        throttle += throttle_cmd * self.max_throttle_change * self.dt
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # Thrust force (along body axis)
        thrust = throttle * self.max_thrust

        # Simple lift model depending on airspeed
        # Lift grows with vx^2; if slower than stall_speed, effective lift is reduced.
        stalled = vx < self.stall_speed and altitude > 5.0
        if stalled:
            effective_lift_coeff = self.lift_coefficient * 0.3
        else:
            effective_lift_coeff = self.lift_coefficient

        lift = effective_lift_coeff * vx * vx

        # Accelerations
        ax = thrust * np.cos(pitch) * self.mass_factor
        az = (lift * np.cos(pitch) + thrust * np.sin(pitch)) * self.mass_factor - self.g

        # If stalled, add extra downward acceleration
        if stalled:
            az -= 2.0
            info["stalled"] = True

        # Integrate velocities
        vx = vx + ax * self.dt
        vz = vz + az * self.dt

        # Integrate positions
        x = x + vx * self.dt
        altitude = altitude + vz * self.dt

        # Base reward: small negative per step to encourage efficient landing
        reward = -0.1

        # Shaping rewards to encourage a shallow, stable approach
        # prefer small pitch, gentle descent, speed near target
        reward -= 0.02 * abs(np.rad2deg(pitch))  # keep nose near level
        reward -= 0.01 * abs(vz - self.target_descent_rate)

        # Penalise deviation from target approach speed (only while still in the air)
        speed_error = (vx - self.target_approach_speed) / max(self.target_approach_speed, 1.0)
        reward -= 0.5 * abs(speed_error)

        if stalled:
            reward -= 5.0  # stall is bad in general

        terminated = False
        truncated = False

        # Termination if too many steps
        if self.step_count >= self.max_steps:
            truncated = True

        # Handle ground contact / touchdown
        if altitude <= 0.0:
            altitude = 0.0
            terminated = True

            vertical_speed = abs(vz)
            pitch_deg = abs(np.rad2deg(pitch))
            distance_from_spot = abs(x - self.landing_spot_x)

            info["touchdown"] = True
            info["touchdown_x"] = x
            info["touchdown_vertical_speed"] = vertical_speed
            info["touchdown_pitch_deg"] = pitch_deg
            info["touchdown_distance_from_spot"] = distance_from_spot
            info["touchdown_speed"] = vx

            # Hard landing / crash logic
            if vertical_speed > self.hard_landing_vspeed:
                # "Explosion" / crash
                reward -= 500.0 + vertical_speed * 50.0
                info["crash"] = True
            else:
                # Base landing quality: combine speed, pitch, and distance to spot
                landing_quality = (
                    -vertical_speed * 20.0
                    - pitch_deg * 2.0
                    - distance_from_spot * 0.5
                )

                # Very good landing (close to spot, gentle, good attitude)
                if distance_from_spot < 5.0 and vertical_speed < self.soft_landing_vspeed and pitch_deg < 5.0:
                    reward += 500.0 + landing_quality
                # Acceptable landing near the spot
                elif distance_from_spot < 20.0:
                    reward += 200.0 + landing_quality
                # Elsewhere but still on runway
                elif self.runway_start_x <= x <= self.braking_zone_end:
                    reward += 50.0 + landing_quality
                else:
                    # Off the runway → heavy penalty
                    reward -= 300.0

        # Out-of-bounds / overshoot / fly-away
        elif altitude > 800.0 or x > self.braking_zone_end + 2000.0:
            truncated = True
            reward -= 100.0

        # Additional crash on stall very close to the ground with downward motion
        if not terminated and stalled and altitude < 20.0 and vz < 0.0:
            terminated = True
            reward -= 500.0
            info["stall_crash"] = True

        # Clip state to observation bounds
        x = float(np.clip(x, self.observation_space.low[0], self.observation_space.high[0]))
        altitude = float(np.clip(altitude, self.observation_space.low[1], self.observation_space.high[1]))
        vx = float(np.clip(vx, self.observation_space.low[2], self.observation_space.high[2]))
        vz = float(np.clip(vz, self.observation_space.low[3], self.observation_space.high[3]))

        self.state = np.array([x, altitude, vx, vz, pitch, throttle], dtype=np.float32)

        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------
    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Airplane Landing")
            self.clock = pygame.time.Clock()

        # Load airplane sprite lazily the first time we render.
        # Erwartet die Datei "airplane-svgrepo-com.jpeg" im Projektordner.
        if self.plane_image is None:
            try:
                raw_image = pygame.image.load("Flugzeug.svg").convert_alpha()
            except Exception as e:
                # Wenn das Laden fehlschlägt, keine Skalierung / Sprite (Fallback-Shape wird genutzt)
                raw_image = None

            if raw_image is not None:
                w, h = raw_image.get_size()
                scaled_size = (int(w * self.plane_image_scale), int(h * self.plane_image_scale))
                self.plane_image = pygame.transform.smoothscale(raw_image, scaled_size)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((255, 255, 255))

        # Current state for camera and HUD
        x, altitude, vx, vz, pitch, throttle = self.state

        # Camera / coordinates
        runway_y = self.ground_y
        # How many meters in x-direction are visible in the window
        meters_in_view = 2000.0
        scale_x = self.window_width / meters_in_view

        # Center the camera on the airplane's x-position
        camera_x = x

        def world_to_screen_x(wx: float) -> int:
            """
            Convert a world x-coordinate (in meters) to a screen x-coordinate (in pixels),
            with the camera centered on the airplane.
            """
            return int(self.window_width / 2 + (wx - camera_x) * scale_x)

        # Entire runway baseline (green)
        runway_start_px = world_to_screen_x(self.runway_start_x)
        runway_end_px = world_to_screen_x(self.braking_zone_end)
        pygame.draw.line(
            self.screen,
            (80, 180, 80),
            (runway_start_px, runway_y),
            (runway_end_px, runway_y),
            6,
        )

        # Landing spot as a small yellow/orange segment around the target point
        landing_start_px = world_to_screen_x(self.landing_zone_start)
        landing_end_px = world_to_screen_x(self.landing_zone_end)
        pygame.draw.line(
            self.screen,
            (255, 180, 0),
            (landing_start_px, runway_y),
            (landing_end_px, runway_y),
            10,
        )

        # Braking zone (red) from the landing spot to end of runway
        braking_start_px = landing_end_px
        braking_end_px = world_to_screen_x(self.braking_zone_end)
        pygame.draw.line(
            self.screen,
            (200, 50, 50),
            (braking_start_px, runway_y),
            (braking_end_px, runway_y),
            10,
        )

        # Text labels
        font = pygame.font.Font(None, 28)
        landing_text = font.render("Landing Spot", True, (0, 0, 0))
        braking_text = font.render("Braking Zone", True, (0, 0, 0))

        self.screen.blit(
            landing_text,
            (landing_start_px + (landing_end_px - landing_start_px) // 2 - landing_text.get_width() // 2,
             runway_y - 30),
        )
        self.screen.blit(
            braking_text,
            (braking_start_px + (braking_end_px - braking_start_px) // 2 - braking_text.get_width() // 2,
             runway_y - 30),
        )

        # Draw airplane
        plane_x_px = world_to_screen_x(x)
        plane_y_px = int(runway_y - altitude * self.pixels_per_meter_y)

        if self.plane_image is not None:
            # Rotate sprite by pitch
            plane_rotated = pygame.transform.rotate(self.plane_image, -np.rad2deg(pitch))
            rect = plane_rotated.get_rect(center=(plane_x_px, plane_y_px))
            self.screen.blit(plane_rotated, rect.topleft)
        else:
            # Fallback: simple ellipse + gear dot if sprite not available
            plane_width = 90
            plane_height = 25

            plane_surface = pygame.Surface((plane_width, plane_height), pygame.SRCALPHA)
            plane_surface.fill((0, 0, 0, 0))

            pygame.draw.ellipse(plane_surface, (230, 230, 230), (0, 0, plane_width, plane_height))
            gear_x = int(plane_width * 0.3)
            gear_y = plane_height - 3
            pygame.draw.circle(plane_surface, (220, 70, 50), (gear_x, gear_y), 5)

            plane_rotated = pygame.transform.rotate(plane_surface, -np.rad2deg(pitch))
            rect = plane_rotated.get_rect(center=(plane_x_px, plane_y_px))
            self.screen.blit(plane_rotated, rect.topleft)

        # HUD text
        hud_font = pygame.font.Font(None, 24)
        hud_text = hud_font.render(
            f"x={x:7.1f}m  h={altitude:6.1f}m  vx={vx:5.1f}m/s  vz={vz:5.2f}m/s  "
            f"pitch={np.rad2deg(pitch):5.1f}deg  thr={throttle:4.2f}",
            True,
            (0, 0, 0),
        )
        self.screen.blit(hud_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None