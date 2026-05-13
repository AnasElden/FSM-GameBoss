"""
Boss AI — FSM Simulator (Streamlit)
Graph-Based Discrete Modeling of an Adaptive Game Boss
Authors: Youssef Mohamed Abdelatif & Anas Abduallah Mostafa
MSA University, Giza, Egypt
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Boss AI — FSM Simulator",
    page_icon="⚔",
    layout="wide",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

/* dark background */
.stApp { background-color: #0a0a0f; color: #e8e8f0; }
.stApp > header { background-color: #0a0a0f; }

/* metric cards */
[data-testid="metric-container"] {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label { color: #6b6b88 !important; font-size: 12px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8e8f0; font-family: 'Space Mono', monospace; font-size: 28px;
}

/* buttons */
.stButton > button {
    background: #13131a;
    border: 1px solid #2a2a3a;
    color: #e8e8f0;
    border-radius: 8px;
    font-size: 14px;
    transition: all 0.15s;
    width: 100%;
}
.stButton > button:hover {
    border-color: #42f5c8;
    color: #42f5c8;
    background: rgba(66,245,200,0.06);
}

/* dataframe */
[data-testid="stDataFrame"] { background: #13131a; border-radius: 8px; }

/* sidebar */
[data-testid="stSidebar"] { background: #0d0d14; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #6b6b88; font-size: 13px; }

/* progress bar override */
.stProgress > div > div { background: #42f5c8; }

/* headers */
h1, h2, h3 { color: #e8e8f0; }
h1 { font-size: 22px; font-weight: 700; }

/* state banner */
.state-banner {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 1rem;
    border-left: 4px solid;
}
</style>
""", unsafe_allow_html=True)

# ─── FSM DATA ────────────────────────────────────────────────
STATES = {
    "s0": {"name": "Idle",       "desc": "Boss observes. Waiting for trigger.",          "color": "#888888", "emoji": "👁"},
    "s1": {"name": "Patrol",     "desc": "Traversing arena. Controlling space.",         "color": "#42f5c8", "emoji": "🚶"},
    "s2": {"name": "Aggressive", "desc": "High-frequency attack chain. Boss is angry.",  "color": "#f54242", "emoji": "⚔"},
    "s3": {"name": "Defensive",  "desc": "Shield up. Reducing exposure.",               "color": "#5599ff", "emoji": "🛡"},
    "s4": {"name": "Berserk",    "desc": "HP ≤ 20%. Unbounded aggression triggered.",   "color": "#f5a742", "emoji": "💢"},
    "s5": {"name": "Recovery",   "desc": "Post-special cooldown. Momentarily vulnerable.", "color": "#c8f542", "emoji": "💚"},
    "s6": {"name": "Defeated",   "desc": "Terminal absorbing state. Encounter over.",   "color": "#444444", "emoji": "💀"},
}

ACTIONS = {
    "⚔ Melee Attack":   {"key": "attack_melee",  "dmg": 8,  "desc": "−8 HP · Close zone"},
    "🏹 Ranged Attack":  {"key": "attack_ranged", "dmg": 5,  "desc": "−5 HP · Mid zone"},
    "✨ Magic Attack":   {"key": "attack_magic",  "dmg": 12, "desc": "−12 HP · Any zone"},
    "💨 Dodge":          {"key": "dodge",          "dmg": 0,  "desc": "0 HP · Evasive move"},
    "🛡 Block":          {"key": "block",          "dmg": 0,  "desc": "0 HP · Defensive stance"},
    "⏸ Wait / Idle":    {"key": "idle",           "dmg": 0,  "desc": "0 HP · Observe"},
}

ADJACENCY = np.array([
    [0,1,1,0,0,0,0],
    [0,0,1,1,0,0,0],
    [0,0,0,1,1,1,1],
    [0,1,1,0,0,0,0],
    [0,0,0,0,0,1,1],
    [0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0],
], dtype=int)
STATE_KEYS = list(STATES.keys())

PLAYER_SCRIPTS = {
    "Aggressive (Table 5)": [
        "attack_melee","attack_melee","attack_melee","attack_melee",   # t1-t4
        "attack_melee","attack_melee","attack_melee","attack_melee",   # t5-t8
        "attack_ranged","attack_ranged","attack_magic",                # t9-t11 → s4
        "attack_melee","idle","idle","idle",                           # t12-t15 → s5
        "idle","idle",                                                 # t16-t17 → s1
        "attack_melee","attack_melee",                                 # t18-t19 → s6
    ],
    "Evasive (Table 6)": [
        "idle","idle","block","idle","idle","idle",
        "attack_magic","dodge","dodge","dodge",
        "idle","idle","attack_magic",
        "attack_magic","attack_magic","attack_magic","attack_magic","attack_magic","attack_magic","attack_magic",
    ],
}

# ─── SESSION STATE INIT ──────────────────────────────────────
def init_sim():
    st.session_state.sim = {
        "state": "s0",
        "hp": 100,
        "tick": 0,
        "dodge_streak": 0,
        "idle_ticks": 0,
        "cooldown_tick": 0,
        "recovery_tick": 0,
        "states_visited": ["s0"],
        "log": [],
        "done": False,
    }

if "sim" not in st.session_state:
    init_sim()

sim = st.session_state.sim

# ─── TRANSITION FUNCTION δ ───────────────────────────────────
def delta(s, action, sim):
    hp_ratio = sim["hp"] / 100

    if s == "s0":
        if action in ("attack_melee", "attack_ranged") and hp_ratio > 0.5:
            return "s2"
        if action == "idle":
            return "s1"
        return "s0"

    if s == "s1":
        if action == "block":
            return "s3"
        if action in ("attack_melee", "attack_ranged", "attack_magic"):
            return "s2"
        return "s1"

    if s == "s2":
        if sim["hp"] <= 0:
            return "s6"
        if hp_ratio <= 0.20:
            return "s4"
        if sim["dodge_streak"] >= 3:
            return "s3"
        return "s2"

    if s == "s3":
        if action == "attack_magic":
            return "s2"
        if sim["idle_ticks"] >= 3:
            return "s1"
        return "s3"

    if s == "s4":
        if sim["hp"] <= 0:
            return "s6"
        if sim["cooldown_tick"] >= 4:
            return "s5"
        return "s4"

    if s == "s5":
        if sim["recovery_tick"] >= 2:
            return "s1"
        return "s5"

    return s  # s6 absorbing

# ─── DO ACTION ───────────────────────────────────────────────
def do_action(action_key):
    s = st.session_state.sim
    if s["done"]:
        return

    dmg_map = {"attack_melee": 8, "attack_ranged": 5, "attack_magic": 12,
               "dodge": 0, "block": 0, "idle": 0}
    dmg = dmg_map.get(action_key, 0)

    s["tick"] += 1
    s["hp"] = max(0, s["hp"] - dmg)

    # counters
    if action_key == "dodge":
        s["dodge_streak"] += 1
    else:
        s["dodge_streak"] = 0

    if s["state"] == "s3" and action_key == "idle":
        s["idle_ticks"] += 1
    elif s["state"] != "s3":
        s["idle_ticks"] = 0

    if s["state"] == "s4":
        s["cooldown_tick"] += 1
    else:
        s["cooldown_tick"] = 0

    if s["state"] == "s5":
        s["recovery_tick"] += 1
    else:
        s["recovery_tick"] = 0

    prev = s["state"]
    nxt  = delta(s["state"], action_key, s)
    s["state"] = nxt

    if nxt not in s["states_visited"]:
        s["states_visited"].append(nxt)

    condition = f"HP={s['hp']}%"
    if s["dodge_streak"] >= 3:
        condition += " · 3× dodge"
    if s["cooldown_tick"] >= 4:
        condition += " · cooldown elapsed"
    if s["recovery_tick"] >= 2:
        condition += " · recovery elapsed"

    s["log"].append({
        "Tick":        s["tick"],
        "Action":      action_key.replace("_", " "),
        "Condition":   condition,
        "δ Output":    f"{prev} → {nxt}",
        "New State":   f"{nxt} {STATES[nxt]['name']}",
        "HP":          s["hp"],
        "Damage":      dmg,
    })

    if nxt == "s6":
        s["done"] = True

# ─── AUTO RUN ────────────────────────────────────────────────
def auto_run(profile_name):
    init_sim()
    script = PLAYER_SCRIPTS[profile_name]
    for action_key in script:
        do_action(action_key)
        if st.session_state.sim["done"]:
            break

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚔ Boss FSM Simulator")
    st.caption("Graph-Based Adaptive Boss AI\nMSA University · Youssef & Anas")
    st.divider()

    st.markdown("### Auto Simulation")
    st.caption("Replay exact traces from the paper")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Aggressive\n(Table 5)", use_container_width=True):
            auto_run("Aggressive (Table 5)")
            st.rerun()
    with col2:
        if st.button("▶ Evasive\n(Table 6)", use_container_width=True):
            auto_run("Evasive (Table 6)")
            st.rerun()

    st.divider()

    st.markdown("### Manual Action")
    selected_action = st.selectbox("Choose player action:", list(ACTIONS.keys()))
    action_info = ACTIONS[selected_action]
    st.caption(action_info["desc"])

    if st.button("⚡ Execute Action", use_container_width=True, disabled=sim["done"]):
        do_action(action_info["key"])
        st.rerun()

    st.divider()

    if st.button("↺ Reset Encounter", use_container_width=True):
        init_sim()
        st.rerun()

    st.divider()
    st.markdown("### ⚠ FSM Note")
    st.caption(
        "This FSM has **no cross-session memory**. "
        "The boss adapts within a single encounter based on your actions, "
        "but resets fully on restart. This is a design-time adaptive model, "
        "not a learning AI."
    )

# ─── MAIN LAYOUT ────────────────────────────────────────────
st.markdown("# ⚔ Boss AI — FSM Simulator")
st.caption("Graph-Based Discrete Modeling of an Adaptive Game Boss · MSA University")

# ─── STATE BANNER ───────────────────────────────────────────
s = sim
state_info = STATES[s["state"]]
hp_pct = s["hp"]
hp_color = "#42f5c8" if hp_pct > 50 else "#f5a742" if hp_pct > 20 else "#f54242"

st.markdown(f"""
<div class="state-banner" style="border-left-color:{state_info['color']}">
  <div style="font-size:12px;color:#6b6b88;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">
    Current Boss State
  </div>
  <div style="font-size:28px;font-weight:700;color:{state_info['color']}">
    {state_info['emoji']} {s['state']} — {state_info['name']}
  </div>
  <div style="font-size:13px;color:#6b6b88;margin-top:4px;">
    {state_info['desc']}
  </div>
</div>
""", unsafe_allow_html=True)

# ─── HP BAR ─────────────────────────────────────────────────
st.markdown(f"**HP: {s['hp']} / 100**")
st.progress(s["hp"] / 100)

# ─── METRICS ────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ticks Elapsed", s["tick"])
c2.metric("States Visited", len(s["states_visited"]))
c3.metric("Dodge Streak", s["dodge_streak"])
c4.metric("Boss HP", f"{s['hp']}%")

st.divider()

# ─── DEFEAT MESSAGE ─────────────────────────────────────────
if s["done"]:
    st.success(f"🏆 Boss Defeated! Encounter ended in **{s['tick']} ticks** · "
               f"**{len(s['states_visited'])}** distinct states visited · "
               f"Path: {' → '.join(s['states_visited'])}")

# ─── TWO COLUMNS: LOG + GRAPH ───────────────────────────────
col_log, col_graph = st.columns([3, 2])

with col_log:
    st.markdown("### 📋 Encounter Log")
    if s["log"]:
        df = pd.DataFrame(s["log"])
        display_df = df[["Tick", "Action", "Condition", "δ Output", "New State", "HP", "Damage"]]
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 45 + len(df) * 35),
        )
    else:
        st.caption("No actions yet. Use the sidebar to play.")

with col_graph:
    st.markdown("### 🔵 FSM State Graph")

    # Draw adjacency matrix as a styled table
    state_labels = [f"{k} {STATES[k]['name']}" for k in STATE_KEYS]
    adj_df = pd.DataFrame(ADJACENCY, index=STATE_KEYS, columns=STATE_KEYS)
    st.dataframe(adj_df, use_container_width=True)
    st.caption("Adjacency matrix A — Table 2 from paper")

    st.divider()

    # States visited path
    st.markdown("### 🗺 State Path")
    path_html = ""
    for i, sid in enumerate(s["states_visited"]):
        color = STATES[sid]["color"]
        emoji = STATES[sid]["emoji"]
        name  = STATES[sid]["name"]
        arrow = " → " if i < len(s["states_visited"]) - 1 else ""
        path_html += f'<span style="color:{color};font-weight:700;">{emoji}{sid}</span>'
        if arrow:
            path_html += f'<span style="color:#444;"> → </span>'
    st.markdown(f'<div style="font-family:monospace;font-size:13px;line-height:2;">{path_html}</div>',
                unsafe_allow_html=True)

st.divider()

# ─── MARKOV MATRIX ──────────────────────────────────────────
with st.expander("📊 Markov Transition Matrix (Table 4 — HP > 50%)", expanded=False):
    markov = np.array([
        [0.00, 0.30, 0.70, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.60, 0.40, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.20, 0.05, 0.70, 0.05],
        [0.00, 0.50, 0.50, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.80, 0.20],
        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    ])
    m_df = pd.DataFrame(markov, index=STATE_KEYS, columns=STATE_KEYS)
    st.dataframe(m_df.style.format("{:.2f}"), use_container_width=True)
    st.caption("Row-stochastic matrix P. Each row sums to 1.00.")

# ─── TRANSITION TABLE ────────────────────────────────────────
with st.expander("📋 Transition Function δ — Table 3", expanded=False):
    transitions = [
        ["s0 Idle",        "attack_melee / ranged", "HP_boss > 50%",        "s2 Aggressive"],
        ["s0 Idle",        "idle",                  "Any",                  "s1 Patrol"],
        ["s1 Patrol",      "attack_melee",          "Player in Close zone", "s2 Aggressive"],
        ["s1 Patrol",      "block",                 "Any",                  "s3 Defensive"],
        ["s2 Aggressive",  "dodge × 3 consecutive", "Any",                  "s3 Defensive"],
        ["s2 Aggressive",  "Any",                   "HP_boss ≤ 20%",        "s4 Berserk"],
        ["s2 Aggressive",  "Any",                   "HP_boss = 0%",         "s6 Defeated"],
        ["s3 Defensive",   "idle ≥ 3 ticks",        "Any",                  "s1 Patrol"],
        ["s3 Defensive",   "attack_magic",          "Any",                  "s2 Aggressive"],
        ["s4 Berserk",     "Any",                   "Cooldown elapsed",     "s5 Recovery"],
        ["s5 Recovery",    "Any",                   "Recovery elapsed",     "s1 Patrol"],
    ]
    t_df = pd.DataFrame(transitions,
                        columns=["Current State", "Player Action", "Condition", "Next State δ(s,a)"])
    st.dataframe(t_df, use_container_width=True, hide_index=True)

# ─── COMPLEXITY ──────────────────────────────────────────────
with st.expander("⚙ Complexity Analysis (Section 8.1)", expanded=False):
    st.markdown("""
| Operation | Complexity |
|---|---|
| Transition lookup (hash table) | O(1) |
| BFS / DFS reachability | O(\\|S\\| + \\|E\\|) = O(7 + 11) |
| Adjacency matrix space | O(\\|S\\|²) = O(49) |
| Markov stationary distribution | O(\\|S\\|³) = O(343) |
| Expected hitting time | O(\\|S\\|³) = O(343) |
""")

st.divider()
st.caption("Youssef Mohamed Abdelatif & Anas Abduallah Mostafa · MSA University · MSA Engineering Journal")
