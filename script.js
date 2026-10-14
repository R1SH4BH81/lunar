/**
 * Lunar RL: Research Showcase Interactive Logic
 * Features:
 * 1. Deep Space Starfield & Parallax Canvas
 * 2. In-Browser 2D Lunar Lander Physics Simulator (Manual + Neural Autopilot)
 * 3. Multi-Agent Fleet Swarm Switcher
 * 4. Regime Filters & HUD Telemetry Tickers
 */

document.addEventListener("DOMContentLoaded", () => {
    initStarfield();
    initRegimeTabs();
    initSwarmSelector();
    initHudTelemetry();
    initLunarSimulator();
});

/* ==========================================================================
   1. STARFIELD & COSMIC DUST CANVAS
   ========================================================================== */
function initStarfield() {
    const canvas = document.getElementById("starfield-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    window.addEventListener("resize", () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const numStars = 140;
    const stars = [];

    for (let i = 0; i < numStars; i++) {
        stars.push({
            x: Math.random() * width,
            y: Math.random() * height,
            radius: Math.random() * 1.2 + 0.3,
            alpha: Math.random() * 0.8 + 0.2,
            twinkleSpeed: (Math.random() - 0.5) * 0.015,
            driftY: (Math.random() * 0.1) + 0.02
        });
    }

    function render() {
        ctx.clearRect(0, 0, width, height);

        for (let star of stars) {
            star.alpha += star.twinkleSpeed;
            if (star.alpha <= 0.1 || star.alpha >= 0.95) {
                star.twinkleSpeed = -star.twinkleSpeed;
            }

            star.y += star.driftY;
            if (star.y > height) {
                star.y = 0;
                star.x = Math.random() * width;
            }

            ctx.beginPath();
            ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(240, 246, 255, ${Math.max(0, Math.min(1, star.alpha))})`;
            ctx.fill();
        }

        requestAnimationFrame(render);
    }
    render();
}

/* ==========================================================================
   2. REGIME TABS FILTERING
   ========================================================================== */
function initRegimeTabs() {
    const buttons = document.querySelectorAll(".regime-tab-btn");
    const cards = document.querySelectorAll(".regime-card");

    buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
            buttons.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            const filter = btn.getAttribute("data-regime");
            cards.forEach((card) => {
                const category = card.getAttribute("data-category");
                if (filter === "all" || category === filter) {
                    card.style.display = "flex";
                } else {
                    card.style.display = "none";
                }
            });
        });
    });
}

/* ==========================================================================
   3. SWARM FLEET SELECTOR
   ========================================================================== */
function initSwarmSelector() {
    const fleetButtons = document.querySelectorAll(".fleet-btn");
    const swarmImg = document.getElementById("swarm-active-img");
    const badgeLabel = document.getElementById("swarm-badge-label");

    if (!swarmImg || !fleetButtons.length) return;

    fleetButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            fleetButtons.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            const count = btn.getAttribute("data-ships");
            const src = btn.getAttribute("data-src");

            // Smooth cross-fade
            swarmImg.style.opacity = "0.3";
            setTimeout(() => {
                swarmImg.src = src;
                swarmImg.style.opacity = "1";
            }, 150);

            if (badgeLabel) {
                badgeLabel.textContent = `${count} Ships \u2022 Autonomous Formation`;
            }
        });
    });
}

/* ==========================================================================
   4. HUD TELEMETRY COUNTER
   ========================================================================== */
function initHudTelemetry() {
    const altEl = document.getElementById("hud-alt");
    const vdotEl = document.getElementById("hud-vdot");
    const xdotEl = document.getElementById("hud-xdot");
    const thetaEl = document.getElementById("hud-theta");

    if (!altEl) return;

    let time = 0;
    setInterval(() => {
        time += 0.05;
        // Periodic descent cycle
        const progress = (Math.sin(time * 0.4) + 1) / 2;
        const currentAlt = (progress * 18.5).toFixed(2);
        const currentVdot = (-Math.abs(Math.cos(time * 0.4)) * 1.8).toFixed(2);
        const currentXdot = (Math.sin(time * 0.8) * 0.35).toFixed(2);
        const currentTheta = (Math.sin(time * 0.5) * 1.2).toFixed(1);

        altEl.textContent = `${currentAlt} M`;
        vdotEl.textContent = `${currentVdot} M/S`;
        xdotEl.textContent = `${currentXdot} M/S`;
        thetaEl.textContent = `${currentTheta} DEG`;
    }, 200);
}

/* ==========================================================================
   5. IN-BROWSER LUNAR LANDER PHYSICS SIMULATOR
   ========================================================================== */
function initLunarSimulator() {
    const canvas = document.getElementById("lunar-sim-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    // UI elements
    const altDisplay = document.getElementById("sim-alt");
    const vyDisplay = document.getElementById("sim-vy");
    const vxDisplay = document.getElementById("sim-vx");
    const angleDisplay = document.getElementById("sim-angle");
    const fuelDisplay = document.getElementById("sim-fuel");
    const banner = document.getElementById("sim-touchdown-banner");
    const bannerText = document.getElementById("banner-text");

    const modeManualBtn = document.getElementById("mode-manual");
    const modeAutopilotBtn = document.getElementById("mode-autopilot");

    const btnMain = document.getElementById("btn-fire-main");
    const btnLeft = document.getElementById("btn-fire-left");
    const btnRight = document.getElementById("btn-fire-right");
    const btnReset = document.getElementById("btn-reset-sim");

    let isAutopilot = false;

    // Simulation state
    const gravity = 0.035; // Lunar gravity scale
    const mainThrustPower = 0.095;
    const sideThrustPower = 0.0025;

    let state = {
        x: canvas.width / 2,
        y: 60,
        vx: (Math.random() - 0.5) * 1.2,
        vy: 0.2,
        angle: (Math.random() - 0.5) * 0.3, // Radians
        angularVel: 0,
        fuel: 100,
        touchdown: false,
        crashed: false
    };

    let keys = {
        main: false,
        left: false,
        right: false
    };

    let particles = [];

    function resetSimulation() {
        state = {
            x: canvas.width / 2 + (Math.random() - 0.5) * 200,
            y: 50,
            vx: (Math.random() - 0.5) * 1.4,
            vy: 0.1,
            angle: (Math.random() - 0.5) * 0.2,
            angularVel: 0,
            fuel: 100,
            touchdown: false,
            crashed: false
        };
        particles = [];
        if (banner) banner.classList.add("hidden");
    }

    // Input Handlers
    window.addEventListener("keydown", (e) => {
        if (["ArrowUp", "KeyW"].includes(e.code)) keys.main = true;
        if (["ArrowLeft", "KeyA"].includes(e.code)) keys.left = true;
        if (["ArrowRight", "KeyD"].includes(e.code)) keys.right = true;
    });

    window.addEventListener("keyup", (e) => {
        if (["ArrowUp", "KeyW"].includes(e.code)) keys.main = false;
        if (["ArrowLeft", "KeyA"].includes(e.code)) keys.left = false;
        if (["ArrowRight", "KeyD"].includes(e.code)) keys.right = false;
    });

    // Touch / Button controls
    if (btnMain) {
        btnMain.addEventListener("mousedown", () => (keys.main = true));
        btnMain.addEventListener("mouseup", () => (keys.main = false));
        btnMain.addEventListener("touchstart", (e) => { e.preventDefault(); keys.main = true; });
        btnMain.addEventListener("touchend", () => (keys.main = false));
    }
    if (btnLeft) {
        btnLeft.addEventListener("mousedown", () => (keys.left = true));
        btnLeft.addEventListener("mouseup", () => (keys.left = false));
        btnLeft.addEventListener("touchstart", (e) => { e.preventDefault(); keys.left = true; });
        btnLeft.addEventListener("touchend", () => (keys.left = false));
    }
    if (btnRight) {
        btnRight.addEventListener("mousedown", () => (keys.right = true));
        btnRight.addEventListener("mouseup", () => (keys.right = false));
        btnRight.addEventListener("touchstart", (e) => { e.preventDefault(); keys.right = true; });
        btnRight.addEventListener("touchend", () => (keys.right = false));
    }
    if (btnReset) btnReset.addEventListener("click", resetSimulation);

    // Mode Toggle
    if (modeManualBtn && modeAutopilotBtn) {
        modeManualBtn.addEventListener("click", () => {
            isAutopilot = false;
            modeManualBtn.classList.add("active");
            modeAutopilotBtn.classList.remove("active");
        });
        modeAutopilotBtn.addEventListener("click", () => {
            isAutopilot = true;
            modeAutopilotBtn.classList.add("active");
            modeManualBtn.classList.remove("active");
        });
    }

    // Target Landing Pad (Centered at canvas bottom)
    const padWidth = 120;
    const padY = canvas.height - 40;
    const padX1 = canvas.width / 2 - padWidth / 2;
    const padX2 = canvas.width / 2 + padWidth / 2;

    // Simulation loop
    function updatePhysics() {
        if (state.touchdown || state.crashed) return;

        // Neural Autopilot Controller (PPO emulation logic)
        if (isAutopilot && state.fuel > 0) {
            const padCenterX = canvas.width / 2;
            const targetXOffset = (state.x - padCenterX) / 100;

            // Desired tilt to steer toward center
            const desiredAngle = Math.max(-0.35, Math.min(0.35, targetXOffset * 0.4 + state.vx * 0.45));
            const angleError = state.angle - desiredAngle;

            // Attitude control
            if (angleError > 0.05) {
                keys.left = true;
                keys.right = false;
            } else if (angleError < -0.05) {
                keys.right = true;
                keys.left = false;
            } else {
                keys.left = false;
                keys.right = false;
            }

            // Descent velocity control (soft touchdown target)
            const targetVy = 0.45;
            if (state.vy > targetVy || (state.y > padY - 140 && state.vy > 0.25)) {
                keys.main = true;
            } else {
                keys.main = false;
            }
        }

        // Apply Thruster Dynamics
        if (keys.main && state.fuel > 0) {
            state.vx += Math.sin(state.angle) * mainThrustPower;
            state.vy -= Math.cos(state.angle) * mainThrustPower;
            state.fuel = Math.max(0, state.fuel - 0.12);

            // Spawn exhaust particles
            for (let i = 0; i < 3; i++) {
                particles.push({
                    x: state.x - Math.sin(state.angle) * 16 + (Math.random() - 0.5) * 4,
                    y: state.y + Math.cos(state.angle) * 16,
                    vx: -Math.sin(state.angle) * (2 + Math.random() * 3) + (Math.random() - 0.5) * 1.5,
                    vy: Math.cos(state.angle) * (2 + Math.random() * 3),
                    radius: Math.random() * 3.5 + 1.5,
                    color: Math.random() > 0.4 ? "#38BDF8" : "#F59E0B",
                    life: 1.0
                });
            }
        }

        if (keys.left && state.fuel > 0) {
            state.angularVel += sideThrustPower;
            state.fuel = Math.max(0, state.fuel - 0.03);
        }
        if (keys.right && state.fuel > 0) {
            state.angularVel -= sideThrustPower;
            state.fuel = Math.max(0, state.fuel - 0.03);
        }

        // Gravity & Inertia
        state.vy += gravity;
        state.x += state.vx;
        state.y += state.vy;

        state.angle += state.angularVel;
        state.angularVel *= 0.985; // Natural angular damping

        // Ground Collision Check
        if (state.y >= padY - 16) {
            state.y = padY - 16;
            const onPad = state.x >= padX1 && state.x <= padX2;
            const safeSpeed = Math.abs(state.vy) < 0.95 && Math.abs(state.vx) < 0.6;
            const safeAngle = Math.abs(state.angle) < 0.25;

            if (onPad && safeSpeed && safeAngle) {
                state.touchdown = true;
                if (banner && bannerText) {
                    bannerText.textContent = "TOUCHDOWN ON LZ-1 CONFIRMED! (+250 PTS)";
                    banner.style.background = "rgba(16, 185, 129, 0.9)";
                    banner.classList.remove("hidden");
                }
            } else {
                state.crashed = true;
                if (banner && bannerText) {
                    bannerText.textContent = onPad ? "HARD LANDING / EXCESSIVE VELOCITY! (-100 PTS)" : "CRASHED OFF LANDING PAD (-100 PTS)";
                    banner.style.background = "rgba(239, 68, 68, 0.9)";
                    banner.classList.remove("hidden");
                }
            }
        }
    }

    function renderSimulator() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Lunar Sky Backdrop & Earth
        const earthGrad = ctx.createRadialGradient(80, 70, 5, 80, 70, 35);
        earthGrad.addColorStop(0, "#38BDF8");
        earthGrad.addColorStop(0.7, "#1E3A8A");
        earthGrad.addColorStop(1, "rgba(7, 10, 19, 0)");
        ctx.fillStyle = earthGrad;
        ctx.beginPath();
        ctx.arc(80, 70, 35, 0, Math.PI * 2);
        ctx.fill();

        // Draw Lunar Surface Regolith
        ctx.fillStyle = "#1E293B";
        ctx.beginPath();
        ctx.moveTo(0, padY);
        ctx.lineTo(padX1, padY);
        ctx.lineTo(padX1, padY + 8);
        ctx.lineTo(padX2, padY + 8);
        ctx.lineTo(padX2, padY);
        ctx.lineTo(canvas.width, padY);
        ctx.lineTo(canvas.width, canvas.height);
        ctx.lineTo(0, canvas.height);
        ctx.closePath();
        ctx.fill();

        // Draw LZ-1 Landing Pad
        ctx.fillStyle = "#0F172A";
        ctx.fillRect(padX1, padY, padWidth, 6);
        ctx.strokeStyle = "#38BDF8";
        ctx.lineWidth = 2;
        ctx.strokeRect(padX1, padY, padWidth, 6);

        // Landing Pad Target Striping
        ctx.fillStyle = "#38BDF8";
        ctx.font = "bold 9px 'JetBrains Mono', monospace";
        ctx.fillText("LZ-1 ALPHA", padX1 + 32, padY + 16);

        // Flags on both sides of pad
        ctx.fillStyle = "#F59E0B";
        ctx.fillRect(padX1 - 2, padY - 24, 2, 24);
        ctx.fillRect(padX2, padY - 24, 2, 24);

        ctx.fillStyle = "#EF4444";
        ctx.beginPath();
        ctx.moveTo(padX1, padY - 24);
        ctx.lineTo(padX1 + 10, padY - 19);
        ctx.lineTo(padX1, padY - 14);
        ctx.fill();

        ctx.beginPath();
        ctx.moveTo(padX2 + 2, padY - 24);
        ctx.lineTo(padX2 + 12, padY - 19);
        ctx.lineTo(padX2 + 2, padY - 14);
        ctx.fill();

        // Render Particles
        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.life -= 0.035;

            if (p.life <= 0) {
                particles.splice(i, 1);
                continue;
            }

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius * p.life, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = p.life;
            ctx.fill();
            ctx.globalAlpha = 1.0;
        }

        // Render Apollo Lunar Module Lander
        ctx.save();
        ctx.translate(state.x, state.y);
        ctx.rotate(state.angle);

        // Gold Foil Descent Stage
        ctx.fillStyle = "#D97706";
        ctx.fillRect(-12, 0, 24, 12);
        ctx.strokeStyle = "#F59E0B";
        ctx.lineWidth = 1;
        ctx.strokeRect(-12, 0, 24, 12);

        // Silver / White Ascent Stage (Cabin)
        ctx.fillStyle = "#F1F5F9";
        ctx.beginPath();
        ctx.moveTo(-10, 0);
        ctx.lineTo(-7, -12);
        ctx.lineTo(7, -12);
        ctx.lineTo(10, 0);
        ctx.closePath();
        ctx.fill();

        // Cabin Windows
        ctx.fillStyle = "#0F172A";
        ctx.fillRect(-5, -9, 4, 3);
        ctx.fillRect(1, -9, 4, 3);

        // Landing Gear Legs & Pads
        ctx.strokeStyle = "#94A3B8";
        ctx.lineWidth = 2;

        // Left Leg
        ctx.beginPath();
        ctx.moveTo(-10, 8);
        ctx.lineTo(-16, 16);
        ctx.stroke();

        ctx.fillStyle = "#64748B";
        ctx.fillRect(-19, 15, 6, 2);

        // Right Leg
        ctx.beginPath();
        ctx.moveTo(10, 8);
        ctx.lineTo(16, 16);
        ctx.stroke();

        ctx.fillRect(13, 15, 6, 2);

        ctx.restore();

        // Update UI Telemetry
        if (altDisplay) altDisplay.textContent = `${Math.max(0, (padY - state.y - 16) * 0.4).toFixed(1)} m`;
        if (vyDisplay) vyDisplay.textContent = `${(-state.vy * 5.0).toFixed(2)} m/s`;
        if (vxDisplay) vxDisplay.textContent = `${(state.vx * 5.0).toFixed(2)} m/s`;
        if (angleDisplay) angleDisplay.textContent = `${(-state.angle * (180 / Math.PI)).toFixed(1)}°`;
        if (fuelDisplay) fuelDisplay.textContent = `${Math.round(state.fuel)}%`;
    }

    function loop() {
        updatePhysics();
        renderSimulator();
        requestAnimationFrame(loop);
    }

    loop();
}
