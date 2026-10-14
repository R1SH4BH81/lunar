"""NASA Apollo Mission Control & Lunar Module (LM) Renderer.

Implements an authentic, realistic visualization of the Apollo Lunar Module
touchdown at Tranquility Base with Apollo Guidance Computer (AGC) telemetry:
- Accurate Apollo LM geometry: Gold Kapton foil descent stage, faceted ascent stage,
  quad RCS thrusters, cantilevered landing struts, and surface contact probes.
- Vacuum hypergolic exhaust: Realistic faint translucent expansion plume.
- NASA Apollo Flight Telemetry HUD: Real aerospace engineering parameters
  (Altitude, H-DOT vertical descent rate, X-DOT forward drift, pitch attitude,
  and the iconic Apollo "LUNAR CONTACT" annunciator light).
No sci-fi neon, no arcade fluff, pure engineering realism.
"""

import math
import numpy as np
import pygame
from typing import List, Tuple, Dict, Any, Optional
from src.utils import save_frames_as_gif

# NASA Telemetry Palette (Apollo CRT Monochrome & Annunciators)
COLOR_SPACE_BLACK = (10, 10, 14)
COLOR_LUNAR_REGOLITH = (125, 125, 128)
COLOR_LUNAR_CRATER = (90, 90, 95)
COLOR_GOLD_FOIL = (212, 175, 55)
COLOR_GOLD_DARK = (165, 130, 30)
COLOR_CABIN_ALUMINUM = (200, 205, 210)
COLOR_CABIN_DARK = (45, 48, 52)
COLOR_STRUT_STEEL = (170, 170, 175)
COLOR_FOOTPAD = (140, 140, 145)
COLOR_HUD_GREEN = (40, 220, 120)
COLOR_HUD_DIM = (25, 120, 70)
COLOR_AMBER_WARN = (235, 160, 40)
COLOR_CONTACT_BLUE = (70, 190, 255)
COLOR_PLUME_AMBER = (240, 210, 140, 70)

class NASALunarRenderer:
    """Renders authentic Apollo Lunar Module flight and NASA mission telemetry."""

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        pygame.init()
        pygame.font.init()
        self.surface = pygame.Surface((width, height))
        # Use clean monospace font for NASA telemetry
        self.font = pygame.font.SysFont("Consolas", 14, bold=True)
        self.title_font = pygame.font.SysFont("Consolas", 16, bold=True)
        self.sub_font = pygame.font.SysFont("Consolas", 11)
        # Physics-based accelerometer tracking
        self.prev_vel: Optional[np.ndarray] = None
        self.filtered_g_axial: float = 0.0
        self.filtered_g_lateral: float = 0.0
 
    def reset(self):
        """Reset internal telemetry and filter states for a new flight."""
        self.prev_vel = None
        self.filtered_g_axial = 0.0
        self.filtered_g_lateral = 0.0

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert LunarLander coordinates (-1.0 to 1.0, 0.0 to 1.4) to screen pixels."""
        # Landing pad center is (0.0, 0.0)
        # Pad on screen is positioned at height - 120
        screen_x = int(self.width / 2.0 + x * (self.width / 2.2))
        screen_y = int((self.height - 120) - y * (self.height / 2.4))
        return screen_x, screen_y

    def render_frame(
        self,
        obs: np.ndarray,
        action: int,
        step: int,
        score: float,
        terrain_points: Optional[List[Tuple[float, float]]] = None,
        is_crashed: bool = False
    ) -> np.ndarray:
        """Render a single high-fidelity Apollo frame with HUD."""
        self.surface.fill(COLOR_SPACE_BLACK)

        # 1. Background stars (sparse, sharp white dots in vacuum)
        np.random.seed(42)
        for _ in range(60):
            sx = int(np.random.uniform(0, self.width))
            sy = int(np.random.uniform(0, self.height - 130))
            self.surface.set_at((sx, sy), (200, 200, 210))

        # 2. Lunar Surface (Cratered Regolith)
        ground_y = self.height - 120
        # Draw lunar horizon
        points = [(0, self.height), (0, ground_y + 10)]
        for i in range(0, self.width + 20, 20):
            # Gentle natural undulation
            undulation = int(math.sin(i * 0.015) * 6 + math.cos(i * 0.04) * 4)
            points.append((i, ground_y + undulation))
        points.append((self.width, self.height))
        pygame.draw.polygon(self.surface, COLOR_LUNAR_REGOLITH, points)

        # Draw craters on surface
        for cx, cy, rad in [(150, ground_y + 40, 35), (420, ground_y + 60, 45), (680, ground_y + 35, 25)]:
            pygame.draw.ellipse(self.surface, COLOR_LUNAR_CRATER, (cx - rad, cy - rad // 2, rad * 2, rad))

        # 3. Landing Zone: Boundary survey poles on sides only
        pad_x1, pad_y = self._world_to_screen(-0.25, 0.0)
        pad_x2, _ = self._world_to_screen(0.25, 0.0)

        # Clean vertical survey poles on the sides
        for pole_x in [pad_x1, pad_x2]:
            pygame.draw.line(self.surface, (200, 205, 215), (pole_x, pad_y), (pole_x, pad_y - 24), 2)
            pygame.draw.circle(self.surface, (240, 245, 255), (pole_x, pad_y - 24), 3)

        # 4. Lunar Module Geometry
        lm_x, lm_y = self._world_to_screen(obs[0], obs[1])
        angle = obs[4]  # in radians

        # Compute exact physical G-force (proper acceleration) from velocity delta
        current_vel = np.array([obs[2], obs[3]])
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        if self.prev_vel is not None:
            # dv/dt in simulation coordinates (dt = 1/50 sec)
            dv = (current_vel - self.prev_vel) * 50.0
            # Lunar gravity acceleration vector in Box2D environment is [0.0, -1.285]
            # Proper acceleration = Coordinate acceleration - Gravity vector
            a_proper_x = dv[0]
            a_proper_y = dv[1] - (-1.285)

            # Rotate into spacecraft body frame (axial along thrust axis, lateral perpendicular)
            # In these units, 1.285 units/s^2 = 0.166 Earth Gs (1.0 Earth G = 7.74 units/s^2)
            raw_g_axial = (a_proper_x * sin_a + a_proper_y * cos_a) / 7.74
            raw_g_lateral = (a_proper_x * cos_a - a_proper_y * sin_a) / 7.74
        else:
            raw_g_axial = 0.0
            raw_g_lateral = 0.0

        self.prev_vel = current_vel.copy()

        # If resting firmly on surface, ground reaction normal force balances lunar gravity (0.166 G)
        if (obs[6] == 1.0 or obs[7] == 1.0) and abs(obs[3]) < 0.05:
            raw_g_axial = max(raw_g_axial, 0.166)

        # Exponential moving average filter mimics flight computer accelerometer damping
        alpha = 0.35
        self.filtered_g_axial = (1 - alpha) * self.filtered_g_axial + alpha * raw_g_axial
        self.filtered_g_lateral = (1 - alpha) * self.filtered_g_lateral + alpha * raw_g_lateral

        self._draw_lunar_module(lm_x, lm_y, angle, action, obs, is_crashed=is_crashed)

        # 5. Flight Telemetry HUD & G-Force Mini-Map
        self._draw_telemetry_hud(obs, action, step, score, is_crashed=is_crashed)
        self._draw_gforce_map(obs, action, self.filtered_g_axial, self.filtered_g_lateral)

        # Convert pygame surface to numpy RGB array
        frame_rgb = pygame.surfarray.array3d(self.surface)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        return frame_rgb

    def render_swarm_frame(
        self,
        ships: List[Dict[str, Any]],
        step: int,
        active_count: int,
        landed_count: int,
        crashed_count: int
    ) -> np.ndarray:
        """Render a full fleet/swarm of Apollo Lunar Modules on lunar regolith."""
        self.surface.fill(COLOR_SPACE_BLACK)

        # 1. Background stars
        np.random.seed(42)
        for _ in range(60):
            sx = int(np.random.uniform(0, self.width))
            sy = int(np.random.uniform(0, self.height - 130))
            self.surface.set_at((sx, sy), (200, 200, 210))

        # 2. Lunar Surface
        ground_y = self.height - 120
        points = [(0, self.height), (0, ground_y + 10)]
        for i in range(0, self.width + 20, 20):
            undulation = int(math.sin(i * 0.015) * 6 + math.cos(i * 0.04) * 4)
            points.append((i, ground_y + undulation))
        points.append((self.width, self.height))
        pygame.draw.polygon(self.surface, COLOR_LUNAR_REGOLITH, points)

        for cx, cy, rad in [(150, ground_y + 40, 35), (420, ground_y + 60, 45), (680, ground_y + 35, 25)]:
            pygame.draw.ellipse(self.surface, COLOR_LUNAR_CRATER, (cx - rad, cy - rad // 2, rad * 2, rad))

        # 3. Landing Zone: Boundary survey poles on sides only
        pad_x1, pad_y = self._world_to_screen(-0.25, 0.0)
        pad_x2, _ = self._world_to_screen(0.25, 0.0)
        for pole_x in [pad_x1, pad_x2]:
            pygame.draw.line(self.surface, (200, 205, 215), (pole_x, pad_y), (pole_x, pad_y - 24), 2)
            pygame.draw.circle(self.surface, (240, 245, 255), (pole_x, pad_y - 24), 3)

        # 4. Draw all fleet ships
        for ship in ships:
            obs = ship["obs"]
            lm_x, lm_y = self._world_to_screen(obs[0], obs[1])
            self._draw_lunar_module(
                lm_x, lm_y,
                angle=obs[4],
                action=ship["action"],
                obs=obs,
                is_crashed=ship.get("is_crashed", False),
                scale=ship.get("scale", 0.75),
                label=ship.get("name", "")
            )

        # 5. Swarm Telemetry HUD
        header = self.title_font.render("LUNAR FLEET AUTONOMOUS DESCENT CONTROL | PGNCS SWARM", True, COLOR_HUD_GREEN)
        self.surface.blit(header, (25, 18))
        stats_line = f"FLEET SIZE: {len(ships)} SHIPS | IN-FLIGHT: {active_count:02d} | TOUCHDOWN: {landed_count:02d} | CRASH: {crashed_count:02d} | MET: T+{step/50.0:04.1f}S"
        self.surface.blit(self.sub_font.render(stats_line, True, COLOR_HUD_DIM), (25, 38))
        pygame.draw.line(self.surface, COLOR_HUD_DIM, (25, 54), (self.width - 25, 54), 1)

        # Convert to numpy RGB
        frame_rgb = pygame.surfarray.array3d(self.surface)
        return np.transpose(frame_rgb, (1, 0, 2))

    def _draw_lunar_module(self, cx: int, cy: int, angle: float, action: int, obs: np.ndarray, is_crashed: bool = False, scale: float = 1.2, label: str = ""):
        """Draw authentic two-stage Apollo Lunar Module (Descent Stage + Ascent Stage)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        def rotate_point(px: float, py: float) -> Tuple[int, int]:
            # Rotate around center of mass
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            return int(cx + rx * scale), int(cy - ry * scale)

        # --- A. VACUUM ENGINE PLUME (Faint hypergolic Aerozine-50/N2O4 expansion) ---
        if action == 2:  # Main descent engine firing
            plume_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            p_base_l = rotate_point(-6, -14)
            p_base_r = rotate_point(6, -14)
            p_tip_l = rotate_point(-16, -42)
            p_tip_r = rotate_point(16, -42)
            p_center = rotate_point(0, -48)

            pygame.draw.polygon(
                plume_surface,
                (245, 200, 110, 110),
                [p_base_l, p_base_r, p_tip_r, p_center, p_tip_l]
            )
            # Core inner shock diamond
            inner_l = rotate_point(-4, -16)
            inner_r = rotate_point(4, -16)
            inner_tip = rotate_point(0, -30)
            pygame.draw.polygon(
                plume_surface,
                (255, 245, 210, 180),
                [inner_l, inner_r, inner_tip]
            )
            self.surface.blit(plume_surface, (0, 0))

        # RCS Reaction Control System lateral thruster puffs
        if action == 1:  # Left thruster fires (pushes right)
            rcs_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            p1 = rotate_point(-14, 10)
            p2 = rotate_point(-28, 12)
            p3 = rotate_point(-28, 8)
            pygame.draw.polygon(rcs_surf, (220, 235, 255, 140), [p1, p2, p3])
            self.surface.blit(rcs_surf, (0, 0))
        elif action == 3:  # Right thruster fires (pushes left)
            rcs_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            p1 = rotate_point(14, 10)
            p2 = rotate_point(28, 12)
            p3 = rotate_point(28, 8)
            pygame.draw.polygon(rcs_surf, (220, 235, 255, 140), [p1, p2, p3])
            self.surface.blit(rcs_surf, (0, 0))

        # --- B. DESCENT ENGINE NOZZLE (Gimbaled bell) ---
        nozzle_poly = [
            rotate_point(-4, -8),
            rotate_point(4, -8),
            rotate_point(7, -14),
            rotate_point(-7, -14)
        ]
        pygame.draw.polygon(self.surface, COLOR_CABIN_DARK, nozzle_poly)

        # --- C. DESCENT STAGE (Octagonal Prism with Gold Kapton Foil) ---
        descent_stage = [
            rotate_point(-14, -8),
            rotate_point(14, -8),
            rotate_point(16, 2),
            rotate_point(12, 6),
            rotate_point(-12, 6),
            rotate_point(-16, 2)
        ]
        pygame.draw.polygon(self.surface, COLOR_GOLD_FOIL, descent_stage)
        pygame.draw.polygon(self.surface, COLOR_GOLD_DARK, descent_stage, 2)

        # Thermal blanket cross-quilting lines
        q1 = rotate_point(-8, -8)
        q2 = rotate_point(8, 6)
        pygame.draw.line(self.surface, COLOR_GOLD_DARK, q1, q2, 1)

        # --- D. ASCENT STAGE (Cabin, Docking Window, Thermal Panels) ---
        cabin_poly = [
            rotate_point(-11, 6),
            rotate_point(11, 6),
            rotate_point(10, 18),
            rotate_point(6, 22),
            rotate_point(-6, 22),
            rotate_point(-10, 18)
        ]
        pygame.draw.polygon(self.surface, COLOR_CABIN_ALUMINUM, cabin_poly)
        pygame.draw.polygon(self.surface, COLOR_CABIN_DARK, cabin_poly, 1)

        # Triangular Cockpit Pilot Windows
        win_l = [rotate_point(-6, 14), rotate_point(-2, 16), rotate_point(-3, 12)]
        win_r = [rotate_point(6, 14), rotate_point(2, 16), rotate_point(3, 12)]
        pygame.draw.polygon(self.surface, (20, 35, 55), win_l)
        pygame.draw.polygon(self.surface, (20, 35, 55), win_r)

        # Forward Rendezvous Radar Antenna & Docking Cone
        pygame.draw.line(self.surface, COLOR_STRUT_STEEL, rotate_point(0, 22), rotate_point(0, 27), 2)
        pygame.draw.circle(self.surface, COLOR_CABIN_ALUMINUM, rotate_point(0, 28), 3)

        # RCS Thruster Quads (4-way cross clusters)
        for rcs_x in [-14, 14]:
            rcs_pt = rotate_point(rcs_x, 12)
            pygame.draw.circle(self.surface, COLOR_CABIN_DARK, rcs_pt, 2)

        # --- E. 4 LANDING GEAR LEGS & SURFACE CONTACT SENSORS ---
        # Left main strut
        pygame.draw.line(self.surface, COLOR_STRUT_STEEL, rotate_point(-12, -2), rotate_point(-24, -16), 2)
        # Left footpad
        pad_l = rotate_point(-24, -16)
        pygame.draw.ellipse(self.surface, COLOR_FOOTPAD, (pad_l[0] - 5, pad_l[1] - 2, 10, 4))
        # Surface contact probe (thin rod extending beneath pad)
        pygame.draw.line(self.surface, (180, 140, 50), pad_l, (pad_l[0], pad_l[1] + 8), 1)

        # Right main strut
        pygame.draw.line(self.surface, COLOR_STRUT_STEEL, rotate_point(12, -2), rotate_point(24, -16), 2)
        # Right footpad
        pad_r = rotate_point(24, -16)
        pygame.draw.ellipse(self.surface, COLOR_FOOTPAD, (pad_r[0] - 5, pad_r[1] - 2, 10, 4))
        # Surface contact probe
        pygame.draw.line(self.surface, (180, 140, 50), pad_r, (pad_r[0], pad_r[1] + 8), 1)

        # Impact debris if crashed
        if is_crashed:
            rng = np.random.RandomState(int(abs(cx * 17 + cy * 31)) % 10000)
            for _ in range(16):
                ox = int(rng.uniform(-28, 28))
                oy = int(rng.uniform(-12, 14))
                rad = int(rng.uniform(2, 5))
                color = (255, rng.randint(90, 200), 40) if rng.rand() > 0.4 else (190, 190, 195)
                pygame.draw.circle(self.surface, color, (cx + ox, cy + oy), rad)

        # Call-sign label
        if label:
            lbl_surf = self.sub_font.render(label, True, (200, 225, 240))
            self.surface.blit(lbl_surf, (cx - lbl_surf.get_width() // 2, cy - int(34 * scale)))

    def _draw_telemetry_hud(self, obs: np.ndarray, action: int, step: int, score: float, is_crashed: bool = False):
        """Draw authentic Guidance Computer (DSKY) and Mission Control flight telemetry."""
        # Top banner - Neutral aerospace mission header (no NASA Apollo 11 text)
        header = self.title_font.render("LUNAR DESCENT TRAJECTORY CONTROL | PGNCS GUIDANCE", True, COLOR_HUD_GREEN)
        site = self.sub_font.render("TARGET: LZ-1 BASE ALPHA (SURVEY COORDINATES: 0.674° N, 23.473° E)", True, COLOR_HUD_DIM)
        self.surface.blit(header, (25, 18))
        self.surface.blit(site, (25, 38))

        # Horizontal telemetry divider
        pygame.draw.line(self.surface, COLOR_HUD_DIM, (25, 54), (self.width - 25, 54), 1)

        # Physical units conversion from LunarLander normalized coordinates
        altitude_m = max(0.0, float(obs[1]) * 100.0)
        v_vertical = float(obs[3]) * 10.0
        v_horizontal = float(obs[2]) * 10.0
        pitch_deg = math.degrees(float(obs[4]))

        # Telemetry columns
        col1_x = 25
        col2_x = 240
        col3_x = 440
        col4_x = 645
        row_y = 62

        # Column 1: Altitude & Rates
        self.surface.blit(self.sub_font.render("RADAR ALTITUDE:", True, COLOR_HUD_DIM), (col1_x, row_y))
        alt_color = COLOR_HUD_GREEN if altitude_m > 15.0 else COLOR_AMBER_WARN
        self.surface.blit(self.font.render(f"{altitude_m:05.1f} M", True, alt_color), (col1_x, row_y + 14))

        self.surface.blit(self.sub_font.render("DESCENT (H-DOT):", True, COLOR_HUD_DIM), (col1_x, row_y + 36))
        v_color = COLOR_HUD_GREEN if abs(v_vertical) < 3.0 else COLOR_AMBER_WARN
        self.surface.blit(self.font.render(f"{v_vertical:+05.2f} M/S", True, v_color), (col1_x, row_y + 50))

        # Column 2: Forward/Lateral & Attitude
        self.surface.blit(self.sub_font.render("LATERAL (X-DOT):", True, COLOR_HUD_DIM), (col2_x, row_y))
        self.surface.blit(self.font.render(f"{v_horizontal:+05.2f} M/S", True, COLOR_HUD_GREEN), (col2_x, row_y + 14))

        self.surface.blit(self.sub_font.render("PITCH ATTITUDE:", True, COLOR_HUD_DIM), (col2_x, row_y + 36))
        pitch_color = COLOR_HUD_GREEN if abs(pitch_deg) < 10.0 else COLOR_AMBER_WARN
        self.surface.blit(self.font.render(f"{pitch_deg:+05.1f} DEG", True, pitch_color), (col2_x, row_y + 50))

        # Column 3: Propulsion & Actuator Status
        eng_status = "FIRING (100%)" if action == 2 else "IDLE"
        eng_color = COLOR_AMBER_WARN if action == 2 else COLOR_HUD_DIM
        self.surface.blit(self.sub_font.render("DPS DESCENT ENGINE:", True, COLOR_HUD_DIM), (col3_x, row_y))
        self.surface.blit(self.font.render(f"{eng_status}", True, eng_color), (col3_x, row_y + 14))

        rcs_str = "PULSE LEFT" if action == 1 else ("PULSE RIGHT" if action == 3 else "OFF")
        self.surface.blit(self.sub_font.render("RCS ATTITUDE JETS:", True, COLOR_HUD_DIM), (col3_x, row_y + 36))
        self.surface.blit(self.font.render(f"{rcs_str}", True, COLOR_HUD_GREEN), (col3_x, row_y + 50))

        # Column 4: Contact Light & Mission Score
        left_contact = obs[6] == 1.0
        right_contact = obs[7] == 1.0
        contact_active = left_contact or right_contact

        contact_box = pygame.Rect(col4_x, row_y, 130, 26)
        if contact_active:
            pygame.draw.rect(self.surface, COLOR_CONTACT_BLUE, contact_box)
            c_text = self.font.render("LUNAR CONTACT", True, (0, 20, 50))
        else:
            pygame.draw.rect(self.surface, (20, 25, 30), contact_box)
            pygame.draw.rect(self.surface, COLOR_HUD_DIM, contact_box, 1)
            c_text = self.font.render("CONTACT LIGHT", True, COLOR_HUD_DIM)
        self.surface.blit(c_text, (col4_x + 6, row_y + 5))

        mission_time = step / 50.0
        self.surface.blit(self.sub_font.render(f"MET: T+{mission_time:04.1f}S | SCR: {score:+05.1f}", True, COLOR_HUD_DIM), (col4_x, row_y + 36))
        
        # Flight status detection
        if is_crashed or (obs[1] < 0.05 and (abs(pitch_deg) > 35.0 or abs(v_vertical) > 4.5)) or (score < -90.0 and (left_contact or right_contact)):
            status_text = "CRITICAL / CRASH"
            s_color = (255, 75, 75)
        elif left_contact and right_contact and abs(v_vertical) < 2.0:
            status_text = "TOUCHDOWN CONFIRMED"
            s_color = COLOR_CONTACT_BLUE
        elif left_contact or right_contact:
            status_text = "SURFACE CONTACT"
            s_color = COLOR_CONTACT_BLUE
        elif step > 400 and altitude_m > 15.0 and abs(v_vertical) < 0.6:
            status_text = "HOLDING PATTERN (HOVER)"
            s_color = COLOR_AMBER_WARN
        else:
            status_text = "DESCENT NOMINAL"
            s_color = COLOR_HUD_GREEN
        self.surface.blit(self.sub_font.render(f"STATUS: {status_text}", True, s_color), (col4_x, row_y + 50))

    def _draw_gforce_map(self, obs: np.ndarray, action: int, g_axial: float, g_lateral: float):
        """Draw an aerospace G-Force accelerometer mini-map (G-G reticle)."""
        box_x = self.width - 155
        box_y = 125
        box_w = 135
        box_h = 135

        # Background panel
        bg_rect = pygame.Rect(box_x, box_y, box_w, box_h)
        pygame.draw.rect(self.surface, (15, 18, 22), bg_rect)
        pygame.draw.rect(self.surface, COLOR_HUD_DIM, bg_rect, 1)

        # Title
        title = self.sub_font.render("G-ACCEL VECTOR", True, COLOR_HUD_GREEN)
        self.surface.blit(title, (box_x + 12, box_y + 6))

        # Circular G-Meter Reticle
        reticle_cx = box_x + box_w // 2
        reticle_cy = box_y + 58
        max_radius = 36  # represents 1.0 Earth G

        # Concentric reference rings (0.5 G, 1.0 G)
        pygame.draw.circle(self.surface, (30, 45, 38), (reticle_cx, reticle_cy), max_radius, 1)
        pygame.draw.circle(self.surface, (25, 38, 30), (reticle_cx, reticle_cy), max_radius // 2, 1)
        # Lunar gravity marker ring (0.166 G)
        pygame.draw.circle(self.surface, (20, 50, 40), (reticle_cx, reticle_cy), int(max_radius * 0.166), 1)
        # Crosshairs
        pygame.draw.line(self.surface, (30, 45, 38), (reticle_cx - max_radius, reticle_cy), (reticle_cx + max_radius, reticle_cy), 1)
        pygame.draw.line(self.surface, (30, 45, 38), (reticle_cx, reticle_cy - max_radius), (reticle_cx, reticle_cy + max_radius), 1)

        total_g = math.sqrt(g_axial**2 + g_lateral**2)

        # Plot G-Vector point in reticle
        # +g_axial points upward on screen (-y), +g_lateral points right (+x)
        # Clamp visual blip to reticle boundary so extreme touchdown shock doesn't leave the box
        clamped_axial = max(-1.2, min(1.2, g_axial))
        clamped_lateral = max(-1.2, min(1.2, g_lateral))
        blip_x = int(reticle_cx + clamped_lateral * max_radius)
        blip_y = int(reticle_cy - clamped_axial * max_radius)

        # Vector line from origin
        g_color = COLOR_HUD_GREEN if total_g < 0.6 else COLOR_AMBER_WARN
        pygame.draw.line(self.surface, g_color, (reticle_cx, reticle_cy), (blip_x, blip_y), 2)
        pygame.draw.circle(self.surface, g_color, (blip_x, blip_y), 4)
        pygame.draw.circle(self.surface, (255, 255, 255), (blip_x, blip_y), 2)

        # Digital G-Force readouts
        g_text = self.font.render(f"{total_g:4.2f} G", True, g_color)
        self.surface.blit(g_text, (box_x + 14, box_y + 102))

        sub_label = self.sub_font.render(f"AX:{g_axial:+.2f} LT:{g_lateral:+.2f}", True, COLOR_HUD_DIM)
        self.surface.blit(sub_label, (box_x + 10, box_y + 118))
