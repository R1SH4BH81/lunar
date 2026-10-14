"""Interactive Streamlit Dashboard for Lunar Lander RL & Reward Shaping Experiments.

Allows users to:
1. Replay and visualize recordings (Random Crash, Solved Baseline, Hover Hack, Potential Shaped)
2. View detailed telemetry breakdown of the 8 observation numbers
3. Analyze comparative reward curves, fuel usage, and landing success rates
4. Run live evaluations directly from the browser UI
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import OBSERVATION_DESCRIPTIONS, ACTION_NAMES

st.set_page_config(
    page_title="Lunar Lander RL: Reward Shaping Lab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Lunar Lander AI: Reinforcement Learning & Reward Shaping</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Train an AI to land a rocket, witness reward hacking, and engineer mathematical fixes with PPO.</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🚀 Multi-Agent Lunar Swarm",
    "🎮 Flight Replays & Regimes",
    "🧠 How RL & Reward Shaping Works",
    "📊 Comparative Experiments",
    "🔬 Telemetry & Physics Breakdown"
])

with tabs[0]:
    st.header("🌌 Autonomous Lunar Swarm (Multi-Agent Fleet Descent)")
    st.caption("Watch a coordinated fleet of Apollo Lunar Modules descend simultaneously onto Base Alpha in real time.")

    # State management for single primary display screen
    if "active_swarm_gif" not in st.session_state:
        st.session_state["active_swarm_gif"] = "recordings/lunar_swarm_fleet.gif"
    if "active_swarm_caption" not in st.session_state:
        st.session_state["active_swarm_caption"] = "Real-Time Lunar Swarm: PGNCS Aces, Hover Exploiters, and Trainee Explorers descending together onto Base Alpha."
    if "active_fleet_size" not in st.session_state:
        st.session_state["active_fleet_size"] = 12

    # Primary top screen display
    swarm_gif = st.session_state["active_swarm_gif"]
    if os.path.exists(swarm_gif):
        st.image(
            swarm_gif,
            caption=st.session_state["active_swarm_caption"],
            use_container_width=True
        )
    else:
        st.info("Generating lunar swarm fleet recording...")

    # Fleet telemetry metrics tied to active simulation
    curr_size = st.session_state["active_fleet_size"]
    aces = int(curr_size * 0.5)
    hacks = int(curr_size * 0.25)
    trains = curr_size - aces - hacks

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric(label="Active Fleet Size", value=f"{curr_size} Ships", delta="Full Swarm")
    with mcol2:
        st.metric(label="PGNCS Aces (Solved)", value=f"{aces} Units", delta="Precision Landing")
    with mcol3:
        st.metric(label="Hover Exploiters", value=f"{hacks} Units", delta="Reward Hacked", delta_color="inverse")
    with mcol4:
        st.metric(label="Trainee Units", value=f"{trains} Units", delta="Exploratory Drift", delta_color="off")

    st.markdown("---")
    st.subheader("🛠️ Launch a Custom Real-Time Swarm Mission")
    st.write("Configure and simulate a fresh fleet descent (updates the top screen):")

    scol1, scol2 = st.columns([2, 1])
    with scol1:
        fleet_size = st.slider("Select Fleet Size (Lander Count):", min_value=4, max_value=16, value=curr_size, step=2)
    with scol2:
        st.write("")
        st.write("")
        launch_btn = st.button("🚀 Launch Live Swarm Simulation", use_container_width=True)

    if launch_btn:
        with st.spinner(f"Simulating {fleet_size}-ship swarm descent with Box2D physics & PGNCS neural guidance..."):
            from src.swarm_simulator import record_swarm_mission
            custom_gif = f"recordings/swarm_custom_{fleet_size}.gif"
            record_swarm_mission(n_ships=fleet_size, max_steps=320, output_gif=custom_gif)
            st.session_state["active_swarm_gif"] = custom_gif
            st.session_state["active_swarm_caption"] = f"Active Custom Swarm ({fleet_size} Ships) | Real-time descent onto Base Alpha"
            st.session_state["active_fleet_size"] = fleet_size
            st.rerun()

with tabs[1]:
    st.header("Visual Flight Recordings")
    st.write("Compare the lander's behavior across different reward regimes:")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Random Policy (Watch it Fail)")
        st.caption("Unguided actions tumble and crash into the lunar surface.")
        random_gif = "recordings/random_agent_crash.gif"
        if os.path.exists(random_gif):
            st.image(random_gif, caption="Random Policy: Tumbling & Crash (-130 pts)", use_container_width=True)
        else:
            st.info("Run `python scripts/random_agent.py` to generate this recording.")

        st.subheader("3. Hover Hacking (The Exploit)")
        st.caption("Rewarding +1 per frame for upright slow flight creates an AI that refuses to land.")
        hover_gif = "recordings/hover_hack_flight.gif"
        if os.path.exists(hover_gif):
            st.image(hover_gif, caption="Reward Hacked: Hovering perpetually until 1,000 frame cutoff (+665 pts)", use_container_width=True)
        else:
            st.info("Hover hack model training or evaluation pending.")

    with col2:
        st.subheader("2. Baseline PPO Landing (Solved)")
        st.caption("Trained with standard Gymnasium reward function; touches down softly between flags.")
        baseline_gif = "recordings/baseline_landing.gif"
        if os.path.exists(baseline_gif):
            st.image(baseline_gif, caption="PPO Baseline: Smooth landing (>200 pts)", use_container_width=True)
        else:
            st.info("Training or evaluation in progress. Refresh once completed.")

        st.subheader("4. Potential-Based Shaping (The Fix)")
        st.caption("PBRS (Ng et al., 1999) + small time fee fixes hovering while accelerating convergence.")
        potential_gif = "recordings/potential_shaped_landing.gif"
        if os.path.exists(potential_gif):
            st.image(potential_gif, caption="Potential-Shaped: Firm, safe, and fuel-efficient landing", use_container_width=True)
        else:
            st.info("Potential-shaped model training or evaluation pending.")

    # Technical Lunar Landing Simulation & Telemetry HUD
    st.markdown("---")
    st.subheader("🛰️ Precision Lunar Descent & Telemetry HUD (Base Alpha)")
    st.caption("Engineered landing platform on lunar regolith with real-time aerospace G-Force accelerometer map & PGNCS flight data.")
    mission_gif = "recordings/lunar_mission_landing.gif"
    if os.path.exists(mission_gif):
        st.image(mission_gif, caption="Precision Lunar Landing: Base Alpha Platform, G-Force Accelerometer Map, H-DOT/X-DOT Rates, Surface Contact Light (Score: +265 pts)", use_container_width=True)
    else:
        st.info("Run `python scripts/record_nasa_mission.py` to generate the simulation.")

with tabs[2]:
    st.header("The Reinforcement Learning Mechanics")
    st.markdown("""
    ### 1. The 8 Observation Numbers
    Every frame (50 times per second), the environment feeds 8 numbers to the neural network:
    """)

    obs_table_data = [
        {"Index": 0, "Variable": "x position", "Range / Meaning": "[-1.0, 1.0]: 0.0 is center of landing pad"},
        {"Index": 1, "Variable": "y position", "Range / Meaning": "[0.0, 1.4]: Height above pad surface"},
        {"Index": 2, "Variable": "x velocity (vx)", "Range / Meaning": "Horizontal drift speed"},
        {"Index": 3, "Variable": "y velocity (vy)", "Range / Meaning": "Vertical speed (negative when falling)"},
        {"Index": 4, "Variable": "Angle (tilt)", "Range / Meaning": "Radians [-pi, pi]: 0.0 is straight up"},
        {"Index": 5, "Variable": "Angular velocity", "Range / Meaning": "Spinning rate in radians/second"},
        {"Index": 6, "Variable": "Left leg contact", "Range / Meaning": "1.0 if touching lunar ground, else 0.0"},
        {"Index": 7, "Variable": "Right leg contact", "Range / Meaning": "1.0 if touching lunar ground, else 0.0"},
    ]
    st.table(pd.DataFrame(obs_table_data))

    st.markdown("""
    ### 2. The 4 Actions
    - **0**: Do nothing (drift on gravity & momentum)
    - **1**: Fire Left Orientation Engine (pushes lander right / tilts clockwise)
    - **2**: Fire Main Engine (counters gravity, burns ~0.3 pts of fuel per tick)
    - **3**: Fire Right Orientation Engine (pushes lander left / tilts counter-clockwise)

    ### 3. Reward Hacking & Why It Happens
    - When you award a static bonus, e.g. **+1 point per frame** for being slow and upright:
      $$\\text{Total Reward} = \\sum_{t=1}^{T} \\left( R_{\\text{env}}(t) + 1.0 \\right)$$
    - Landing **ends the episode**. If the agent lands at frame 200, it receives terminal bonus +100, totaling ~250 points.
    - If the agent **never lands**, it hovers until the environment timeout (1,000 frames), accumulating $1000 \\times 1.0 - \\text{fuel} \\approx 700\\text{ points}$!
    - The AI mathematically exploits this flaw: **Hovering beats landing.**

    ### 4. The Mathematical Fix: Potential-Based Reward Shaping (PBRS)
    - Theorem (Ng, Harada, & Russell, 1999): A shaping reward of the form:
      $$F(s, a, s') = \\gamma \\Phi(s') - \\Phi(s)$$
      **strictly preserves the optimal policy!**
    - When an agent hovers in place, $s' \\approx s$, so $\\Phi(s') - \\Phi(s) \\approx 0$. No bonus points can be farmed by lingering.
    - Combined with a small per-frame fee ($-0.15$), hovering is heavily penalized, while safe continuous descent is rewarded.
    """)

with tabs[3]:
    st.header("Comparative Performance Analysis")
    chart_path = "recordings/comparison_chart.png"
    csv_path = "recordings/comparison_chart.csv"

    if os.path.exists(chart_path):
        st.image(chart_path, caption="Comprehensive Comparison of Reward Regimes", use_container_width=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.subheader("Performance Metrics Table")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run `python scripts/compare_experiments.py` after models finish training to see the comparison matrix.")

with tabs[4]:
    st.header("Telemetry Inspector")
    st.write("Examine flight physics and observation variables live.")
    st.code("""
# Observation vector breakdown:
obs[0] = x_pos          # Horizontal offset from pad
obs[1] = y_pos          # Altitude
obs[2] = v_x            # Lateral speed
obs[3] = v_y            # Vertical speed
obs[4] = angle          # Tilt (radians)
obs[5] = angular_vel    # Rotational speed
obs[6] = left_leg_down  # Ground sensor
obs[7] = right_leg_down # Ground sensor
    """, language="python")
