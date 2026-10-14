"""Interactive Streamlit Dashboard for Lunar Lander RL & Reward Shaping Experiments.

Provides evaluation, telemetry inspection, multi-agent fleet simulation,
and comparative analysis across PPO reward regimes.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
# Neutralize torch.classes path inspection by Streamlit's local sources watcher
torch.classes.__path__ = []
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import OBSERVATION_DESCRIPTIONS, ACTION_NAMES

st.set_page_config(
    page_title="Lunar Lander RL Lab",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Refined UI styles: precision aerospace dark-console theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    code, pre, .mono-data {
        font-family: 'JetBrains Mono', Consolas, monospace !important;
        font-variant-numeric: tabular-nums;
    }

    /* Header styling */
    .app-header {
        padding: 0.5rem 0 1.25rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 0.35rem;
        line-height: 1.4;
    }

    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem 1.25rem;
    }
    .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-caption {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Regime badges */
    .badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .badge-baseline { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-hack { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-pbrs { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-random { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Clean card wrappers */
    .regime-card {
        background: #0B1120;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }
    .regime-title {
        font-size: 1rem;
        font-weight: 600;
        color: #F1F5F9;
        margin: 0.25rem 0 0.5rem 0;
    }
    .regime-desc {
        font-size: 0.825rem;
        color: #94A3B8;
        line-height: 1.4;
        margin-bottom: 0.75rem;
        min-height: 2.4rem;
    }

    /* Tabs refinement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.9rem;
        color: #94A3B8;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #F8FAFC !important;
        font-weight: 600;
        border-bottom: 2px solid #38BDF8 !important;
    }

    /* Math and callout blocks */
    .math-callout {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Lunar Lander RL: Reward Shaping Lab</h1>
    <div class="app-subtitle">Autonomous descent control, reward hacking dynamics, and potential-based policy invariance under PPO.</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Fleet Simulation",
    "Flight Regimes",
    "Reward Mechanics",
    "Benchmark Matrix",
    "Telemetry Inspector"
])

# -----------------------------------------------------------------------------
# TAB 1: FLEET SIMULATION
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Autonomous Fleet Descent")
    st.caption("Simultaneous multi-agent trajectory execution under distinct policy checkpoints.")

    # State management
    if "active_swarm_gif" not in st.session_state:
        st.session_state["active_swarm_gif"] = "recordings/lunar_swarm_fleet.gif"
    if "active_fleet_size" not in st.session_state:
        st.session_state["active_fleet_size"] = 12

    curr_size = st.session_state["active_fleet_size"]
    aces = int(curr_size * 0.5)
    hacks = int(curr_size * 0.25)
    trains = curr_size - aces - hacks

    # Active fleet metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(label="Active Fleet Size", value=f"{curr_size} Ships")
    with col_m2:
        st.metric(label="Baseline Policy (Aces)", value=f"{aces} Units")
    with col_m3:
        st.metric(label="Hover Exploiters", value=f"{hacks} Units")
    with col_m4:
        st.metric(label="Trainee Units", value=f"{trains} Units")

    swarm_gif = st.session_state["active_swarm_gif"]
    if os.path.exists(swarm_gif):
        st.image(swarm_gif, use_container_width=True)
    else:
        st.info("No recorded fleet trajectory found. Generate one using the controls below.")

    st.markdown("---")
    st.subheader("Simulate Custom Fleet")

    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        fleet_size = st.slider("Fleet size (lander count):", min_value=4, max_value=16, value=curr_size, step=2)
    with ctrl_col2:
        st.write("")
        st.write("")
        launch_btn = st.button("Run Simulation", use_container_width=True)

    if launch_btn:
        with st.spinner(f"Simulating {fleet_size}-agent Box2D dynamics and PPO guidance..."):
            from src.swarm_simulator import record_swarm_mission
            custom_gif = f"recordings/swarm_custom_{fleet_size}.gif"
            record_swarm_mission(n_ships=fleet_size, max_steps=320, output_gif=custom_gif)
            st.session_state["active_swarm_gif"] = custom_gif
            st.session_state["active_fleet_size"] = fleet_size
            st.rerun()

# -----------------------------------------------------------------------------
# TAB 2: FLIGHT REGIMES
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Flight Regimes Across Reward Specifications")
    st.caption("Side-by-side comparison of learned behavior under standard, flawed, and shaped reward formulations.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="regime-card">
            <span class="badge badge-random">Random Baseline</span>
            <div class="regime-title">Unguided Policy</div>
            <div class="regime-desc">Stochastic action selection produces uncontrolled tumbling and ground impact (-130 mean score).</div>
        </div>
        """, unsafe_allow_html=True)
        random_gif = "recordings/random_agent_crash.gif"
        if os.path.exists(random_gif):
            st.image(random_gif, use_container_width=True)
        else:
            st.info("Run scripts/random_agent.py to generate recording.")

        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="regime-card">
            <span class="badge badge-hack">Reward Hacked</span>
            <div class="regime-title">Hovering Exploitation</div>
            <div class="regime-desc">Static +1 bonus per upright frame incentivizes perpetual hovering to step cutoff (+665 score).</div>
        </div>
        """, unsafe_allow_html=True)
        hover_gif = "recordings/hover_hack_flight.gif"
        if os.path.exists(hover_gif):
            st.image(hover_gif, use_container_width=True)
        else:
            st.info("Hover hack model recording pending.")

    with col2:
        st.markdown("""
        <div class="regime-card">
            <span class="badge badge-baseline">Standard Env</span>
            <div class="regime-title">Baseline PPO Policy</div>
            <div class="regime-desc">Standard Gymnasium reward function converges to soft touchdown between designated landing flags (>200 score).</div>
        </div>
        """, unsafe_allow_html=True)
        baseline_gif = "recordings/baseline_landing.gif"
        if os.path.exists(baseline_gif):
            st.image(baseline_gif, use_container_width=True)
        else:
            st.info("Baseline model recording pending.")

        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="regime-card">
            <span class="badge badge-pbrs">Mathematically Shaped</span>
            <div class="regime-title">Potential-Based Reward Shaping</div>
            <div class="regime-desc">PBRS preserves optimal policy invariance while step penalties penalize lingering in mid-air.</div>
        </div>
        """, unsafe_allow_html=True)
        potential_gif = "recordings/potential_shaped_landing.gif"
        if os.path.exists(potential_gif):
            st.image(potential_gif, use_container_width=True)
        else:
            st.info("Potential-shaped recording pending.")

    st.markdown("---")
    st.subheader("Full Apollo Descent Telemetry HUD")
    st.caption("Touchdown on LZ-1 with real-time velocity vectoring, vertical rate (H-DOT), and contact light.")
    mission_gif = "recordings/lunar_mission_landing.gif"
    if os.path.exists(mission_gif):
        st.image(mission_gif, use_container_width=True)
    else:
        st.info("Run scripts/record_nasa_mission.py to generate recording.")

# -----------------------------------------------------------------------------
# TAB 3: REWARD MECHANICS
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Observation Space and MDP Specification")

    st.markdown("""
    The `LunarLander-v3` environment feeds an 8-dimensional continuous state vector $s_t \\in \\mathbb{R}^8$ at 50 Hz:
    """)

    obs_data = [
        {"Index": 0, "Variable": "x", "Type": "Continuous", "Range": "[-1.0, 1.0]", "Description": "Horizontal displacement from pad center"},
        {"Index": 1, "Variable": "y", "Type": "Continuous", "Range": "[0.0, 1.4]", "Description": "Altitude above lunar surface"},
        {"Index": 2, "Variable": "v_x", "Type": "Continuous", "Range": "[-Inf, Inf]", "Description": "Horizontal velocity"},
        {"Index": 3, "Variable": "v_y", "Type": "Continuous", "Range": "[-Inf, Inf]", "Description": "Vertical velocity (negative during descent)"},
        {"Index": 4, "Variable": "theta", "Type": "Continuous", "Range": "[-pi, pi]", "Description": "Lander angle in radians (0.0 is upright)"},
        {"Index": 5, "Variable": "omega", "Type": "Continuous", "Range": "[-Inf, Inf]", "Description": "Angular velocity (radians/sec)"},
        {"Index": 6, "Variable": "contact_left", "Type": "Binary", "Range": "{0.0, 1.0}", "Description": "Left landing leg surface contact sensor"},
        {"Index": 7, "Variable": "contact_right", "Type": "Binary", "Range": "{0.0, 1.0}", "Description": "Right landing leg surface contact sensor"}
    ]
    st.dataframe(pd.DataFrame(obs_data), use_container_width=True, hide_index=True)

    st.subheader("Action Space")
    action_data = [
        {"Action": 0, "Name": "No-op", "Effect": "Ballistic drift under lunar gravity (-10.0 m/s^2 equivalent)"},
        {"Action": 1, "Name": "Fire Left Engine", "Effect": "Clockwise torque + slight rightward acceleration"},
        {"Action": 2, "Name": "Fire Main Engine", "Effect": "Upward thrust countering gravity; expends ~0.3 reward per step"},
        {"Action": 3, "Name": "Fire Right Engine", "Effect": "Counter-clockwise torque + slight leftward acceleration"}
    ]
    st.dataframe(pd.DataFrame(action_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Reward Hacking Analysis")
    st.markdown("""
    When training with a naive shaping bonus for upright posture and gentle vertical descent:
    $$r_{\\text{shaped}}(s, a) = r_{\\text{env}}(s, a) + 1.0 \\cdot \\mathbb{I}_{\\text{upright}}$$

    An agent that lands terminates the episode at step $T \\approx 200$, collecting a terminal landing bonus of $+100$ and cumulative reward $R \\approx 250$.
    An agent that never lands maintains altitude and hovers until environment timeout at $T_{\\max} = 1000$:
    $$R_{\\text{hover}} = \\sum_{t=1}^{1000} (1.0 - \\text{fuel cost}) \\approx 650 - 700$$

    Because $R_{\\text{hover}} > R_{\\text{landing}}$, policy gradient algorithms converge to perpetual hovering.
    """)

    st.subheader("Mathematical Correction: Potential-Based Reward Shaping")
    st.markdown("""
    According to Ng, Harada, and Russell (1999), policy invariance is guaranteed if and only if the auxiliary shaping function is expressed as the difference of a potential function:
    $$F(s, a, s') = \\gamma \\Phi(s') - \\Phi(s)$$

    Under this formulation:
    1. **Cycle invariance**: The sum of shaping rewards along any closed loop or stationary hover is zero:
       $$\\sum_{t=1}^{T} \\gamma^{t-1} F(s_t, a_t, s_{t+1}) = -\\Phi(s_0) + \\gamma^T \\Phi(s_T)$$
    2. **Hover prevention**: When hovering in place, $s_{t+1} \\approx s_t$, making $F(s, a, s') \\approx (\\gamma - 1) \\Phi(s) \\le 0$. Lingering yields zero gain.
    3. Coupled with a constant step fee ($-0.15$), fuel conservation and timely touchdown remain strictly optimal.
    """)

# -----------------------------------------------------------------------------
# TAB 4: BENCHMARK MATRIX
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Empirical Benchmark Results")
    st.caption("Quantitative metrics across trained PPO models evaluated over 15 deterministic episodes.")

    chart_path = "recordings/comparison_chart.png"
    csv_path = "recordings/comparison_chart.csv"

    if os.path.exists(chart_path):
        st.image(chart_path, use_container_width=True)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Format table for scanability
        display_df = df.copy()
        if "mean_score" in display_df.columns:
            display_df["mean_score"] = display_df["mean_score"].map(lambda x: f"{x:.1f}")
        if "std_score" in display_df.columns:
            display_df["std_score"] = display_df["std_score"].map(lambda x: f"{x:.1f}")
        if "solved_pct" in display_df.columns:
            display_df["solved_pct"] = display_df["solved_pct"].map(lambda x: f"{x:.0f}%")
        if "landed_pct" in display_df.columns:
            display_df["landed_pct"] = display_df["landed_pct"].map(lambda x: f"{x:.0f}%")
        if "mean_length" in display_df.columns:
            display_df["mean_length"] = display_df["mean_length"].map(lambda x: f"{x:.0f} steps")

        st.subheader("Performance Comparison Table")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run scripts/compare_experiments.py to generate benchmark results.")

# -----------------------------------------------------------------------------
# TAB 5: TELEMETRY INSPECTOR
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("State Vector Specification & Signal Ranges")
    st.caption("Reference sensor values for policy evaluation and Box2D simulation diagnostics.")

    st.code("""
# Observation vector indexing:
obs[0] = pos_x        # Range: [-1.0, 1.0]  (0.0 = pad center)
obs[1] = pos_y        # Range: [ 0.0, 1.4]  (0.0 = ground level)
obs[2] = vel_x        # Range: [-Inf, Inf]  (positive = moving right)
obs[3] = vel_y        # Range: [-Inf, Inf]  (negative = falling)
obs[4] = angle        # Range: [-pi, pi]    (0.0 = upright, positive = counter-clockwise)
obs[5] = angular_vel  # Range: [-Inf, Inf]  (spin rate in rad/s)
obs[6] = left_leg     # Range: {0.0, 1.0}   (1.0 = ground contact confirmed)
obs[7] = right_leg    # Range: {0.0, 1.0}   (1.0 = ground contact confirmed)
    """, language="python")

    st.subheader("Reward Formulations in Code")
    st.code("""
# 1. Baseline Gymnasium reward
# r = 100 * (1 - |x|) + 100 * (1 - |y|) + 100 * (1 - |v_x|) + 100 * (1 - |v_y|)
#     - 100 * |angle| + 10 * left_leg + 10 * right_leg - 0.3 * main_engine - 0.03 * side_engine

# 2. Hover Hack reward (Naive shaping)
r_hack = r_env + 1.0 if (abs(obs[4]) < 0.25 and abs(obs[3]) < 0.35) else r_env

# 3. Potential-Based Reward Shaping (Ng et al. 1999)
phi_current = -np.sqrt(obs[0]**2 + obs[1]**2) - np.sqrt(obs[2]**2 + obs[3]**2) - abs(obs[4])
f_shaping = gamma * phi_next - phi_current - 0.15
r_pbrs = r_env + f_shaping
    """, language="python")
